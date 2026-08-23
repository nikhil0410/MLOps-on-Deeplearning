"""FastAPI inference service for Cats-vs-Dogs classification (M2 + M5).

Endpoints:
* ``GET  /``        - service metadata
* ``GET  /health``  - health check (liveness/readiness)
* ``POST /predict`` - accepts an image upload, returns label + probabilities
* ``GET  /metrics`` - Prometheus metrics

Also wires up structured JSON request/response logging and Prometheus
request-count / latency metrics (M5). Raw image bytes are never logged.
"""
from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pythonjsonlogger import jsonlogger

from src.api.metrics import (
    INFERENCE_LATENCY,
    PREDICTION_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.api.schemas import HealthResponse, PredictionResponse
from src.config import load_params
from src.data.preprocess import preprocess_image
from src.models.inference import load_model, predict

SERVICE_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Structured JSON logging (M5) - metadata only, never raw request payloads.
# --------------------------------------------------------------------------- #
logger = logging.getLogger("inference_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


# --------------------------------------------------------------------------- #
# Model state, loaded once at startup.
# --------------------------------------------------------------------------- #
class ModelState:
    model: Optional[torch.nn.Module] = None
    class_names: list[str] = []
    image_size: int = 224
    architecture: str = "SimpleCNN"


state = ModelState()


def _resolve_model_path() -> Path:
    params = load_params()
    default_path = Path(params["train"]["model_dir"]) / params["train"]["model_name"]
    return Path(os.getenv("MODEL_PATH", default_path))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; degrade gracefully if it's missing."""
    model_path = _resolve_model_path()
    try:
        model, class_names, image_size = load_model(model_path, device="cpu")
        state.model = model
        state.class_names = class_names
        state.image_size = image_size
        logger.info(
            "model_loaded",
            extra={"model_path": str(model_path), "classes": class_names},
        )
    except FileNotFoundError:
        logger.warning(
            "model_not_found", extra={"model_path": str(model_path)}
        )
    yield
    # nothing to clean up


app = FastAPI(
    title="Cats vs Dogs Inference Service",
    description="Baseline CNN image classifier served with FastAPI.",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Middleware: request/response logging + Prometheus metrics (M5).
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint).observe(elapsed)

    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": endpoint,
            "status_code": response.status_code,
            "latency_ms": round(elapsed * 1000, 2),
            "client": request.client.host if request.client else None,
        },
    )
    response.headers["X-Process-Time-ms"] = str(round(elapsed * 1000, 2))
    return response


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "cats-vs-dogs-inference",
        "version": SERVICE_VERSION,
        "endpoints": ["/health", "/predict", "/metrics", "/docs"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness/readiness probe. Reports whether a model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        model_name=state.architecture,
        version=SERVICE_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict_endpoint(file: UploadFile = File(...)):
    """Accept an uploaded image and return the predicted class + probabilities."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not decode image")

    tensor = preprocess_image(image, image_size=state.image_size)

    infer_start = time.perf_counter()
    label, probabilities = predict(state.model, tensor, state.class_names)
    infer_elapsed = time.perf_counter() - infer_start
    INFERENCE_LATENCY.observe(infer_elapsed)
    PREDICTION_COUNT.labels(label).inc()

    # Log prediction metadata only (no raw image bytes).
    # NOTE: avoid reserved LogRecord keys like 'filename' in `extra`.
    logger.info(
        "prediction",
        extra={
            "upload_filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(raw),
            "label": label,
            "confidence": round(probabilities[label], 4),
            "inference_ms": round(infer_elapsed * 1000, 2),
        },
    )

    return PredictionResponse(
        label=label,
        confidence=probabilities[label],
        probabilities=probabilities,
        inference_time_ms=round(infer_elapsed * 1000, 2),
    )


@app.get("/metrics", tags=["meta"])
def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_error", extra={"path": request.url.path, "error": str(exc)})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
