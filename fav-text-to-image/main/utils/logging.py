"""Minimal wandb init helper."""
from __future__ import annotations

from typing import Optional


def init_wandb(*, project: str, run_name: str, config: dict, enabled: bool = True):
    if not enabled:
        return None
    import wandb
    wandb.init(project=project, name=run_name, config=config)
    return wandb.run
