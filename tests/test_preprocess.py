"""Unit tests for data preprocessing (M3).

Covers the ``preprocess_image`` function used by the inference service and the
``split_and_process`` dataset-splitting utility.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.preprocess import (
    IMAGENET_MEAN,
    get_eval_transform,
    get_train_transform,
    preprocess_image,
    split_and_process,
)


def test_preprocess_image_shape(sample_rgb_image):
    """Output must be a batched (1, 3, 224, 224) tensor regardless of input size."""
    tensor = preprocess_image(sample_rgb_image, image_size=224)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_image_custom_size(sample_rgb_image):
    tensor = preprocess_image(sample_rgb_image, image_size=128)
    assert tensor.shape == (1, 3, 128, 128)


def test_preprocess_image_converts_grayscale(sample_grayscale_image):
    """A single-channel image must be converted to 3 channels."""
    tensor = preprocess_image(sample_grayscale_image, image_size=224)
    assert tensor.shape[1] == 3  # channels


def test_preprocess_image_is_normalized(sample_rgb_image):
    """After ImageNet normalization, values should be centered (not 0-1)."""
    tensor = preprocess_image(sample_rgb_image, image_size=224)
    # With mean ~0.45 and std ~0.22, normalized values commonly fall in [-3, 3].
    assert tensor.min() >= -3.5
    assert tensor.max() <= 3.5
    # It must NOT still be raw [0, 1] pixels.
    assert tensor.min() < 0.0


def test_train_and_eval_transforms_produce_same_shape(sample_rgb_image):
    train_t = get_train_transform(224)(sample_rgb_image)
    eval_t = get_eval_transform(224)(sample_rgb_image)
    assert train_t.shape == eval_t.shape == (3, 224, 224)


def _make_raw_dataset(root: Path, classes, n_per_class):
    """Helper: write tiny jpgs for each class."""
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            Image.new("RGB", (64, 64), (i, i, i)).save(d / f"{cls}_{i}.jpg")


def test_split_and_process_counts(tmp_path):
    """Splitting 10 images/class 80/10/10 must preserve the total and resize."""
    classes = ["cats", "dogs"]
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    _make_raw_dataset(raw, classes, n_per_class=10)

    counts = split_and_process(
        raw_dir=raw,
        processed_dir=processed,
        image_size=224,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        seed=42,
        classes=classes,
    )

    # 10 per class -> 8 train, 1 val, 1 test per class -> x2 classes
    assert counts["train"] == 16
    assert counts["val"] == 2
    assert counts["test"] == 2
    assert sum(counts.values()) == 20

    # Processed images must be resized to 224x224.
    a_train_image = next((processed / "train" / "cats").glob("*.jpg"))
    with Image.open(a_train_image) as img:
        assert img.size == (224, 224)
