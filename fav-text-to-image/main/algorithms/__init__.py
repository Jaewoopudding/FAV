"""Alignment trainers (FAV, DRaFT)."""
from __future__ import annotations

from .base import Trainer
from .draft import DRaFTTrainer
from .fav import FAVTrainer

__all__ = ["Trainer", "FAVTrainer", "DRaFTTrainer", "build_trainer"]

_TRAINERS = {
    "fav": FAVTrainer,
    "draft": DRaFTTrainer,
}


def build_trainer(cfg, *, backbone, train_reward, raw_reward, accelerator,
                  train_prompts, eval_prompts) -> Trainer:
    name = cfg.algorithm.name
    if name not in _TRAINERS:
        raise NotImplementedError(f"Unknown algorithm {name!r} (have {list(_TRAINERS)})")
    return _TRAINERS[name](
        backbone=backbone, train_reward=train_reward, raw_reward=raw_reward,
        cfg=cfg, accelerator=accelerator,
        train_prompts=train_prompts, eval_prompts=eval_prompts,
    )
