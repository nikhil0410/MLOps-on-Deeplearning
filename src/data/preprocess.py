"""Image preprocessing and dataset utilities.

Contains:
* ``preprocess_image`` - pure, testable function used by the inference service
  to turn a raw ``PIL.Image`` into a normalized model-ready tensor.
* Training / evaluation transforms (with data augmentation for training).
* ``split_and_process`` - splits raw images into train/val/test and resizes
  them to 224x224 (the DVC ``preprocess`` stage).
* ``build_dataloaders`` - builds PyTorch dataloaders from processed data.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Standard ImageNet normalization constants (good defaults for CNNs).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Training transform with data augmentation for better generalization."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic transform for validation / test / inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Convert a raw PIL image into a normalized, batched tensor.

    Returns a tensor of shape ``(1, 3, image_size, image_size)`` ready to be fed
    straight into the model. This is the function exercised by the unit tests
    and reused by the FastAPI service so training and serving stay consistent.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = get_eval_transform(image_size)(image)
    return tensor.unsqueeze(0)  # add batch dimension


def split_and_process(
    raw_dir: str | Path,
    processed_dir: str | Path,
    image_size: int,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
    classes: List[str],
) -> Dict[str, int]:
    """Split raw images into train/val/test and resize them to ``image_size``.

    Returns a dict with the number of images written to each split.
    """
    import random

    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "splits must sum to 1"

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    if processed_path.exists():
        shutil.rmtree(processed_path)

    rng = random.Random(seed)
    counts = {"train": 0, "val": 0, "test": 0}

    for class_name in classes:
        images = sorted((raw_path / class_name).glob("*.jpg"))
        rng.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)

        split_map = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split_name, split_images in split_map.items():
            out_dir = processed_path / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                with Image.open(img_path) as img:
                    img = img.convert("RGB").resize((image_size, image_size))
                    img.save(out_dir / img_path.name, quality=90)
                counts[split_name] += 1

    return counts


def build_dataloaders(
    processed_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train/val/test dataloaders from processed image folders."""
    processed_path = Path(processed_dir)

    train_ds = ImageFolder(processed_path / "train", transform=get_train_transform(image_size))
    val_ds = ImageFolder(processed_path / "val", transform=get_eval_transform(image_size))
    test_ds = ImageFolder(processed_path / "test", transform=get_eval_transform(image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes


def main() -> None:
    """DVC ``preprocess`` stage entry point."""
    from src.config import load_params

    params = load_params()
    d = params["data"]
    counts = split_and_process(
        raw_dir=d["raw_dir"],
        processed_dir=d["processed_dir"],
        image_size=d["image_size"],
        train_split=d["train_split"],
        val_split=d["val_split"],
        test_split=d["test_split"],
        seed=d["seed"],
        classes=params["classes"],
    )
    print(f"Preprocessing complete. Split counts: {counts}")


if __name__ == "__main__":
    main()
