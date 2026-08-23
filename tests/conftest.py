"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.model import build_model


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """A random 300x200 RGB image (non-square, to exercise resizing)."""
    arr = (np.random.rand(200, 300, 3) * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    """A random grayscale image to test RGB conversion."""
    arr = (np.random.rand(120, 120) * 255).astype("uint8")
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def untrained_model() -> torch.nn.Module:
    """A freshly initialized model (no training needed for logic tests)."""
    model = build_model(num_classes=2)
    model.eval()
    return model


@pytest.fixture
def class_names() -> list[str]:
    return ["cats", "dogs"]
