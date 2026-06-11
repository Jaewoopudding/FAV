"""Reward models for text-to-image alignment (HPSv2, NSFW)."""
from __future__ import annotations

import torch

from .base import NegatedReward, Reward, composed_reward_fn
from .hpsv2 import HPSv2Reward
from .nsfw import NSFWReward

__all__ = [
    "Reward",
    "NegatedReward",
    "composed_reward_fn",
    "HPSv2Reward",
    "NSFWReward",
    "build_reward",
]


def build_reward(reward_cfg, *, prompts: list[str], device, dtype: torch.dtype = torch.float32):
    """Construct a reward; returns ``(train_reward, raw_reward)`` where
    ``train_reward`` is possibly sign-flipped for training."""
    name = reward_cfg.name
    if name == "hpsv2":
        reward: Reward = HPSv2Reward(
            dtype=dtype, device=device,
            use_raw_prompt=bool(reward_cfg.get("hps_use_raw_prompt", False)),
        )
    elif name == "nsfw":
        reward = NSFWReward(dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown reward {name!r} (supported: hpsv2, nsfw)")

    reward.requires_grad_(False)
    reward.eval()
    if reward.is_text_conditioned:
        reward.build_labels(prompts)

    # Default: negate NSFW (minimize), keep HPSv2 (maximize).
    negate = bool(reward_cfg.get("negate", name == "nsfw"))
    train_reward = NegatedReward(reward) if negate else reward
    return train_reward, reward
