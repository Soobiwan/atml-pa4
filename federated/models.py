"""Model definitions used across the federated learning experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Baseline CNN for CIFAR-10 experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _ensure_parent(path: str | os.PathLike[str]) -> None:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def save_weights(model: nn.Module, path: str = "CNN_weights/init_weights.pt") -> None:
    """Persist the model state_dict for reuse between experiments."""

    _ensure_parent(path)
    torch.save(model.state_dict(), path)


def load_weights(
    model: nn.Module,
    path: str = "CNN_weights/init_weights.pt",
    map_location: Optional[torch.device | str] = None,
) -> None:
    """Load weights saved via :func:`save_weights` into ``model``."""

    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)


__all__ = ["SimpleCNN", "save_weights", "load_weights"]
