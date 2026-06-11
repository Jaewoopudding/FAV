"""LoRA + optimizer-state checkpoint I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def save_ckpt(
    path: str | Path,
    *,
    lora_state: dict,
    optimizer_state: Optional[dict] = None,
    scheduler_state: Optional[dict] = None,
    step: int = 0,
    config: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lora_state": lora_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler_state,
            "step": step,
            "config": config,
        },
        path,
    )


def load_ckpt(path: str | Path, map_location: Any = "cpu") -> dict:
    return torch.load(path, map_location=map_location)
