"""Reusable inference utilities (model loading + prediction).

Kept separate from the FastAPI app so the prediction logic can be unit tested
(M3) and reused anywhere. ``predict`` returns a label plus per-class
probabilities that always sum to 1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.models.model import build_model


def load_model(
    model_path: str | Path,
    num_classes: int = 2,
    device: str | torch.device = "cpu",
) -> Tuple[torch.nn.Module, List[str], int]:
    """Load a trained model checkpoint.

    Supports both a plain ``state_dict`` and the richer checkpoint dict saved by
    ``train.py`` (which also stores ``class_names`` and ``image_size``).

    Returns ``(model, class_names, image_size)``.
    """
    device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device)

    class_names: List[str] = [str(i) for i in range(num_classes)]
    image_size = 224

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        num_classes = checkpoint.get("num_classes", num_classes)
        class_names = checkpoint.get("class_names", class_names)
        image_size = checkpoint.get("image_size", image_size)
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint  # a bare state_dict

    model = build_model(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names, image_size


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    class_names: List[str],
    device: str | torch.device = "cpu",
) -> Tuple[str, Dict[str, float]]:
    """Run a forward pass and return ``(label, {class_name: probability})``.

    ``tensor`` must be shaped ``(1, 3, H, W)`` (already preprocessed).
    """
    device = torch.device(device)
    model.eval()
    logits = model(tensor.to(device))
    probs = F.softmax(logits, dim=1).squeeze(0)

    probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    predicted_idx = int(torch.argmax(probs).item())
    label = class_names[predicted_idx]
    return label, probabilities
