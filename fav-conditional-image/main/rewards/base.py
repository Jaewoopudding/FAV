"""Reward abstract interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Reward(ABC):
    """Differentiable scalar reward."""

    name: str = "abstract"

    def set_labels(self, labels: Any) -> None:
        """Optional hook for class-conditional rewards. Default: no-op."""

    @abstractmethod
    def __call__(self, images_or_features: Any) -> Any:
        """Return reward scalar of shape ``(B,)``."""
