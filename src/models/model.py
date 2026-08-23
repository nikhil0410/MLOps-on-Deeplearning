"""Baseline CNN architecture for Cats-vs-Dogs binary classification.

A deliberately small convolutional network: three conv blocks followed by
global average pooling and a linear classifier. Global pooling keeps the
parameter count low and makes the model robust to the input resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A simple 3-block CNN baseline."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 112 -> 56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 56 -> 28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int = 2) -> SimpleCNN:
    """Factory helper so callers don't import the class directly."""
    return SimpleCNN(num_classes=num_classes)
