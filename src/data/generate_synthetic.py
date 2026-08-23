"""Generate a small synthetic Cats-vs-Dogs dataset.

This is a stand-in for the Kaggle dataset so the *entire* pipeline
(versioning -> training -> serving -> CI/CD) runs end-to-end without a multi-GB
download. The two classes are made deliberately separable so the baseline CNN
actually learns something:

* ``cats`` -> warm-toned background (reddish/orange) with bright circles
* ``dogs`` -> cool-toned background (bluish/green) with bright rectangles

To use the *real* Kaggle dataset instead, drop the images into
``data/raw/cats`` and ``data/raw/dogs`` and skip this stage.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.config import load_params


def _make_cat_image(size: int, rng: random.Random) -> Image.Image:
    """Warm background + circles."""
    base = (
        rng.randint(150, 255),  # high red
        rng.randint(60, 140),   # medium green
        rng.randint(20, 90),    # low blue
    )
    img = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(3, 6)):
        r = rng.randint(size // 12, size // 5)
        cx, cy = rng.randint(0, size), rng.randint(0, size)
        fill = (rng.randint(200, 255), rng.randint(150, 220), rng.randint(120, 200))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    return img


def _make_dog_image(size: int, rng: random.Random) -> Image.Image:
    """Cool background + rectangles."""
    base = (
        rng.randint(20, 90),    # low red
        rng.randint(80, 160),   # medium green
        rng.randint(150, 255),  # high blue
    )
    img = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(3, 6)):
        w = rng.randint(size // 8, size // 3)
        h = rng.randint(size // 8, size // 3)
        x, y = rng.randint(0, size - w), rng.randint(0, size - h)
        fill = (rng.randint(120, 200), rng.randint(150, 220), rng.randint(200, 255))
        draw.rectangle([x, y, x + w, y + h], fill=fill)
    return img


def generate(raw_dir: str | Path, samples_per_class: int, image_size: int, seed: int) -> Path:
    """Generate the synthetic dataset under ``raw_dir/<class>/*.jpg``."""
    rng = random.Random(seed)
    np.random.seed(seed)
    raw_path = Path(raw_dir)

    generators = {"cats": _make_cat_image, "dogs": _make_dog_image}
    for class_name, gen_fn in generators.items():
        class_dir = raw_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img = gen_fn(image_size, rng)
            img.save(class_dir / f"{class_name}_{i:04d}.jpg", quality=90)
        print(f"  generated {samples_per_class} '{class_name}' images -> {class_dir}")

    return raw_path


def main() -> None:
    params = load_params()
    d = params["data"]

    parser = argparse.ArgumentParser(description="Generate synthetic cats/dogs data.")
    parser.add_argument("--raw-dir", default=d["raw_dir"])
    parser.add_argument("--samples-per-class", type=int, default=d["samples_per_class"])
    parser.add_argument("--image-size", type=int, default=d["image_size"])
    parser.add_argument("--seed", type=int, default=d["seed"])
    args = parser.parse_args()

    print("Generating synthetic dataset...")
    generate(args.raw_dir, args.samples_per_class, args.image_size, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
