"""Unit tests for model + inference utilities (M3).

Covers the ``predict`` inference helper, the model forward pass, and the
``load_model`` checkpoint round-trip.
"""
from __future__ import annotations

import math

import torch

from src.data.preprocess import preprocess_image
from src.models.inference import load_model, predict
from src.models.model import build_model


def test_model_forward_output_shape(untrained_model):
    """Model must output one logit vector per class for a batch."""
    x = torch.randn(4, 3, 224, 224)
    out = untrained_model(x)
    assert out.shape == (4, 2)


def test_predict_returns_valid_label(untrained_model, class_names, sample_rgb_image):
    tensor = preprocess_image(sample_rgb_image, image_size=224)
    label, probs = predict(untrained_model, tensor, class_names)
    assert label in class_names
    assert set(probs.keys()) == set(class_names)


def test_predict_probabilities_sum_to_one(untrained_model, class_names, sample_rgb_image):
    """Softmax probabilities must sum to 1 and lie in [0, 1]."""
    tensor = preprocess_image(sample_rgb_image, image_size=224)
    _, probs = predict(untrained_model, tensor, class_names)
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-5)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_predict_label_matches_argmax(untrained_model, class_names, sample_rgb_image):
    """The returned label must correspond to the highest probability."""
    tensor = preprocess_image(sample_rgb_image, image_size=224)
    label, probs = predict(untrained_model, tensor, class_names)
    assert label == max(probs, key=probs.get)


def test_load_model_roundtrip(tmp_path, class_names):
    """A saved checkpoint must load back with matching metadata and weights."""
    model = build_model(num_classes=2)
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": 2,
            "class_names": class_names,
            "image_size": 224,
            "architecture": "SimpleCNN",
        },
        ckpt_path,
    )

    loaded, loaded_classes, image_size = load_model(ckpt_path, device="cpu")
    assert loaded_classes == class_names
    assert image_size == 224

    # Same input -> identical output for original and reloaded model.
    x = torch.randn(1, 3, 224, 224)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)
