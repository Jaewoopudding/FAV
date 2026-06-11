"""Reward interface for text-to-image alignment.

A reward is a frozen model that extracts a differentiable image feature
(``encode_images``) and scores it (``forward``); the shared module makes
``nabla_r = d(reward)/d(features)`` well-defined for the FAV SVGD loss.
Text-conditioned rewards hold a per-prompt text-feature table; ``set_labels``
selects which prompt to score against.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Reward(ABC, nn.Module):
    """Differentiable reward: image features → scalar score."""

    name: str = "abstract"
    is_text_conditioned: bool = False
    feature_dim: int = 0  # informational

    def set_labels(self, labels: torch.Tensor) -> None:
        """Select per-sample text conditioning (text-conditioned rewards only)."""

    def build_labels(self, prompts: list[str]) -> None:
        """Build the per-prompt text-feature table (text-conditioned rewards only)."""

    @abstractmethod
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) pixels in [-1, 1] → (B, D) differentiable features."""

    @abstractmethod
    def forward(self, images_or_features: torch.Tensor) -> torch.Tensor:
        """(B, D) features OR (B, 3, H, W) images → (B,) scores."""


class NegatedReward(nn.Module):
    """Wraps a reward to flip its sign (e.g. minimise NSFW probability)."""

    def __init__(self, reward: Reward) -> None:
        super().__init__()
        self.reward = reward

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.reward.encode_images(images)

    def set_labels(self, labels: torch.Tensor) -> None:
        self.reward.set_labels(labels)

    def forward(self, images_or_features: torch.Tensor) -> torch.Tensor:
        return -self.reward(images_or_features)


def composed_reward_fn(reward):
    """Return a callable images → score (encode then score in one shot)."""
    def _fn(images: torch.Tensor) -> torch.Tensor:
        features = reward.encode_images(images)
        return reward(features)
    return _fn
