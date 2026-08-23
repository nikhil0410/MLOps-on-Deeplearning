"""Configuration loading utilities.

Loads ``params.yaml`` once and exposes a typed-ish config object that the rest
of the codebase (training, DVC stages, inference service) can share. Keeping a
single source of truth avoids drift between training and serving.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = two levels up from this file (src/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@lru_cache(maxsize=1)
def load_params(params_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load and cache the ``params.yaml`` file.

    The path can be overridden with the ``PARAMS_PATH`` environment variable,
    which is handy inside containers where the working directory may differ.
    """
    path = Path(params_path or os.getenv("PARAMS_PATH", DEFAULT_PARAMS_PATH))
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_class_names() -> list[str]:
    """Return the ordered list of class names (index position == label id)."""
    return list(load_params()["classes"])
