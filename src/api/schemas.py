"""Pydantic response models for the inference API."""
from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    # Fields start with "model_", which pydantic treats as a protected
    # namespace by default; opt out to silence the warning.
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., examples=["ok"])
    model_loaded: bool = Field(..., description="Whether a trained model is loaded")
    model_name: str = Field(..., description="Architecture name")
    version: str = Field(..., description="Service version")


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Probability of the predicted class")
    probabilities: Dict[str, float] = Field(..., description="Per-class probabilities")
    inference_time_ms: float = Field(..., description="Model inference latency in ms")


class ErrorResponse(BaseModel):
    detail: str
