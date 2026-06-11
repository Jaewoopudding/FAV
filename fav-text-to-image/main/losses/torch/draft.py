"""DRaFT loss — direct reward backprop: ``L = -reward_multiplier * E[r(image)]``."""
from __future__ import annotations

from typing import Callable, Optional

import torch


def draft_loss(
    images_gen: torch.Tensor,
    r_x: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    reward_multiplier: float = 1.0,
) -> torch.Tensor:
    """Reward-maximization loss on decoded pixel images."""
    rewards = r_x(images_gen)
    return -rewards.mean() * reward_multiplier
