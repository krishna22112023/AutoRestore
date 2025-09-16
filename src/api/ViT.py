import argparse
import sys
import pyprojroot
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoImageProcessor, AutoModel
import uvicorn

root = pyprojroot.find_root(pyprojroot.has_dir("src"))
sys.path.append(str(root))

from config import logger

def _device() -> torch.device:
    """Return CUDA device if available else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _ImagePair(BaseModel):
    raw_image: str = Field(..., description="Path to raw/original image")
    processed_image: str = Field(..., description="Path to processed/modified image")
    model: str = Field(
        default="google/vit-base-patch16-224",
        description="HuggingFace model name or local path",
    )


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    """Load *model_name* and corresponding processor, cache the result."""
    logger.info("Loading model '%s' …", model_name)
    device = _device()
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return processor, model


@torch.inference_mode()
def _get_embedding(model: torch.nn.Module, processor, image_path: Path, device: torch.device):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # Prefer pooled output if available
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        emb = outputs.pooler_output  # (1, hidden_dim)
    else:
        emb = outputs.last_hidden_state[:, 0, :]  # CLS token
    return emb.squeeze(0)  # (hidden_dim,)


def _cosine_similarity(raw_path: Path, proc_path: Path, model_name: str) -> float:
    processor, model = _load_model(model_name)
    device = next(model.parameters()).device
    emb_raw = _get_embedding(model, processor, raw_path, device)
    emb_proc = _get_embedding(model, processor, proc_path, device)
    cos_sim = F.cosine_similarity(emb_raw.unsqueeze(0), emb_proc.unsqueeze(0)).item()
    return float(cos_sim)


def create_app() -> FastAPI:
    app = FastAPI(
        title="ViT Cosine Similarity API",
        description="Compute cosine similarity between embeddings of two images using a pre-trained Vision Transformer (ViT).",
    )

    # Log every incoming request and response
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info("Incoming %s %s", request.method, request.url.path)
        try:
            response = await call_next(request)
            logger.info("Completed %s %s -> %s", request.method, request.url.path, response.status_code)
            return response
        except Exception as exc:
            logger.exception("Unhandled error processing request: %s", exc)
            raise

    # Custom exception handlers for better logging & JSON responses
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning("HTTPException: %s - detail: %s", exc.status_code, exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("Validation error: %s", exc)
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.post("/vit_similarity")
    async def vit_similarity_endpoint(payload: _ImagePair):
        raw_path = Path(payload.raw_image)
        proc_path = Path(payload.processed_image)
        if not raw_path.exists():
            logger.warning("Raw image not found: %s", raw_path)
            raise HTTPException(status_code=400, detail=f"Raw image not found: {raw_path}")
        if not proc_path.exists():
            logger.warning("Processed image not found: %s", proc_path)
            raise HTTPException(status_code=400, detail=f"Processed image not found: {proc_path}")
        try:
            score = _cosine_similarity(raw_path, proc_path, payload.model)
        except Exception as e:
            logger.exception("vit_similarity failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
        return {"score": score}

    return app


def _parse_args():
    parser = argparse.ArgumentParser(description="Serve ViT cosine similarity via FastAPI")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5004, help="Port to bind (default: 5004)")
    parser.add_argument("--reload", action="store_true", help="Enable live reload (uvicorn --reload)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, reload=args.reload)
