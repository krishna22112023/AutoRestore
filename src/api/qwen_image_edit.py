import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from functools import lru_cache
from pathlib import Path
import argparse

@lru_cache(maxsize=4)
def load_model(image_model: str):
    if image_model == "qwen-image-edit":
        from diffusers import QwenImageEditPipeline
        model = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16, device_map="balanced")
        return model
    else:
        raise HTTPException(status_code=400, detail="Invalid image model")

class EditRequest(BaseModel):
    input_image_path: str
    output_image_path: str
    image_model: str
    prompt: str
    num_inference_steps: int
    classifier_free_guidance: float
    seed: int

class EditResponse(BaseModel):
    success: bool 

app = FastAPI(
        title="Qwen Image Edit API",
        description="Image restoration via Qwen-Image-Edit")

@app.post("/qwen_image_edit_hf", response_model=EditResponse)
async def qwen_image_edit_hf(request: EditRequest):
    input_path = Path(request.input_image_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Input image not found: {request.input_image_path}")
    
    input_image = Image.open(input_path)
    
    try:
        model = load_model(request.image_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    
    model.set_progress_bar_config(disable=None)
    
    inputs = {
            "image": input_image,
            "prompt": request.prompt,
            "generator": torch.manual_seed(request.seed),
            "true_cfg_scale": request.classifier_free_guidance,
            "negative_prompt": " ",
            "num_inference_steps": request.num_inference_steps,
        }
    try:
        with torch.inference_mode():
            output = model(**inputs)
            output_image = output.images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")
    
    try:
        output_path = Path(request.output_image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_image.save(output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")
    
    return {"success": True}

def parse_args():
    parser = argparse.ArgumentParser(description="Serve qwen_edit via FastAPI")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5005, help="Port to bind (default: 5005)")
    parser.add_argument("--reload", action="store_true", help="Enable live reload (uvicorn --reload)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
