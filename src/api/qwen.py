import argparse
import logging
import time
import uvicorn
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from diffusers import QwenImageEditPipeline


logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen-Image-Edit"
# Supported resolution pairs (w, h)
SUPPORTED_DIMS = [
    (928, 1664),
    (1056, 1584),
    (1140, 1472),
    (1328, 1328),
    (1664, 928),
    (1584, 1056),
    (1472, 1140),
]
# Default resize size picked from qwen_local example
DEFAULT_SIZE = (1664, 928)

PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "prompts" / "qwen_preprocessor.md"
)

try:
    pipeline: QwenImageEditPipeline = QwenImageEditPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # computation in bf16 if GPU supports
        device_map="balanced",  # spread modules if needed
    )
    # Enable performance knobs
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("QwenImageEditPipeline loaded successfully with flash attention and tf32 enabled")
except Exception as e:
    logger.exception("Failed to load QwenImageEditPipeline: %s", e)
    raise

class _EditRequest(BaseModel):
    image_path: Path = Field(..., description="Path to the input image on disk")
    degradation_type: str = Field(..., description="Type of degradation (e.g. haze, noise)")
    severity: str = Field(..., description="Severity level (Very Low, Low, Medium, High, Very High)")
    inference_steps: int = Field(20, description="Number of diffusion inference steps")
    true_cfg_scale: float = Field(9.0, description="Classifier-free guidance scale")
    negative_prompt: Optional[str] = Field(" ", description="Negative prompt")
    output_path: Path = Field(..., description="Path to the output directory")

class _EditResponse(BaseModel):
    duration: float = Field(..., description="Inference time in seconds")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Qwen Image Edit API",
        description="Image restoration via Qwen-Image-Edit")

    @app.post("/qwen_edit", response_model=_EditResponse)
    async def qwen_edit(req: _EditRequest):
        """Run Qwen-Image-Edit pipeline on the provided *image_path*.

        The image is resized to a supported resolution (default 1664×928) before editing.
        The processed image is cropped to 1280×675 (centre crop) for consistency.
        """
        if not req.image_path.exists():
            raise HTTPException(status_code=400, detail=f"Image not found: {req.image_path}")

        try:
            image = Image.open(req.image_path).convert("RGB")
        except Exception as e:
            logger.exception("Failed to open image: %s", e)
            raise HTTPException(status_code=400, detail=f"Unable to open image: {e}")

        # Resize to supported resolution
        target_size = DEFAULT_SIZE if image.size not in SUPPORTED_DIMS else image.size
        resized_image = image.resize(target_size, Image.LANCZOS)
        logger.debug("Resized image to %s", image.size)

        # Format prompt
        try:
            prompt_template = PROMPT_TEMPLATE_PATH.read_text()
        except FileNotFoundError:
            logger.warning("Prompt template not found at %s; using raw degradation/severity", PROMPT_TEMPLATE_PATH)
            prompt_template = "Remove {degradation_type} at {severity} severity while preserving details."

        formatted_prompt = prompt_template.format(
            degradation_type=req.degradation_type, severity=req.severity
        )

        inputs = {
            "image": resized_image,
            "prompt": formatted_prompt,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": req.true_cfg_scale,
            "negative_prompt": req.negative_prompt,
            "num_inference_steps": req.inference_steps
        }

        start = time.time()
        try:
            with torch.inference_mode():
                result = pipeline(**inputs)
                output_image = result.images[0]
        except Exception as e:
            logger.exception("Pipeline inference failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
        duration = time.time() - start

        # Centre crop to 1280x675
        w, h = output_image.size
        crop_w, crop_h = image.size
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        output_image = output_image.crop((left, top, left + crop_w, top + crop_h))
        logger.info(f"Image processed in {duration} seconds")

        out_path = req.output_path / f"{req.image_path.stem}-{req.degradation_type}-{req.severity}.jpg"
        output_image.save(out_path)

        # Return a dictionary matching _EditResponse schema
        return {"duration": duration}
    
    return app

def parse_args():
    parser = argparse.ArgumentParser(description="Serve qwen_edit via FastAPI")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5005, help="Port to bind (default: 5005)")
    parser.add_argument("--reload", action="store_true", help="Enable live reload (uvicorn --reload)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, reload=args.reload)