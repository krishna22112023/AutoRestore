import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
import pyiqa
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class IQARequest(BaseModel):
    """Payload model for IQA requests.

    raw_image_path: Path to the original image.
    proc_image_path: Optional path to the processed / restored image. Pass an empty
        string or omit for no-reference metrics.
    metric: Name of the metric to compute (e.g. ``lpips`` or ``qalign``).
    """

    raw_image_path: str
    proc_image_path: str
    metric: str


def score(raw_image, proc_image, metric: str) -> float:
    """Compute the requested IQA *metric* for the given image(s).

    For full-reference metrics (e.g. LPIPS) both *raw_image* and
    *proc_image* are required.  For no-reference metrics (e.g. Q-Align,
    BRISQUE) only *raw_image* is used.
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    metric_lower = metric.lower()

    try:
        iqa_metric = pyiqa.create_metric(metric_lower, device=device)
    except Exception as e:  # pragma: no cover – let the caller handle the 400
        raise ValueError(f"Unsupported metric: {metric}") from e

    if metric_lower in {"lpips", "dists"}:  # full-reference metrics that need two images
        if not proc_image:
            raise ValueError("proc_image_path must be provided for full-reference metrics like LPIPS/DISTS")
        return float(iqa_metric(raw_image, proc_image).item())
    else:  # no-reference metrics
        return float(iqa_metric(raw_image).item())


def create_app() -> FastAPI:
    app = FastAPI(title="Image quality analysis API", description="Exposes metrics as an HTTP endpoint")

    @app.post("/iqa")
    async def iqa_endpoint(payload: IQARequest):
        """Compute the requested IQA metric and return it as JSON."""

        raw_path = payload.raw_image_path
        raw_image = Image.open(raw_path)
        proc_path = payload.proc_image_path
        if proc_path:
            proc_image = Image.open(proc_path)
            if raw_image.size != proc_image.size:
                raw_image = raw_image.resize(proc_image.size, Image.LANCZOS)
        else:
            proc_image = None
        metric = payload.metric.lower()

        logger.info("Incoming IQA request | metric=%s, raw='%s', proc='%s'", metric, raw_path, proc_path)
        # Validate required arguments based on metric type
        if metric in {"lpips", "dists"} and not proc_image:
            raise HTTPException(status_code=400, detail="proc_image_path is required for full-reference metrics like lpips/dists")

        try:
            result = score(raw_image, proc_image, metric)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("IQA computation failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

        logger.info("IQA result | metric=%s → %.4f", metric, result)
        return {"score": result}

    return app


def _parse_args():
    parser = argparse.ArgumentParser(description="Serve IQA score via FastAPI")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5003, help="Port to bind (default: 5003)")
    parser.add_argument("--reload", action="store_true", help="Enable live reload (uvicorn --reload)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, reload=args.reload)
