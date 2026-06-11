"""DRaFT loss: direct policy gradient (pixel-space reward maximization). Reward must be differentiable."""
from __future__ import annotations

from typing import Callable

import torch


def draft_loss(
    images_gen: torch.Tensor,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    reward_multiplier: float = 1.0,
) -> torch.Tensor:
    """Direct policy gradient: ``-E[r(images_gen)] * reward_multiplier``."""
    rewards = reward_fn(images_gen)
    return -rewards.mean() * reward_multiplier
