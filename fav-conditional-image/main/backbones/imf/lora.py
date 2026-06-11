"""LoRA injection for the iMF backbone (MiT attention projections)."""
from __future__ import annotations

from typing import Iterable, Optional

import torch.nn as nn

from .model.mit import inject_lora as _mit_inject_lora
from .model.mit import get_lora_state_dict as _mit_get_lora_state_dict


def inject_imf_lora(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: Optional[int] = None,
    target_modules: Optional[Iterable[str]] = None,
    dropout: float = 0.0,
) -> tuple[nn.Module, int, int]:
    """Inject LoRA into iMF (MiT) attention projections. Returns ``(model, n_trainable, n_total)``."""
    return _mit_inject_lora(
        model,
        rank=rank,
        alpha=alpha,
        target_modules=list(target_modules) if target_modules is not None else None,
        dropout=dropout,
    )


def get_imf_lora_state(model: nn.Module) -> dict:
    return _mit_get_lora_state_dict(model)


def load_imf_lora_state(model: nn.Module, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    for name in unexpected:
        raise RuntimeError(f"Unexpected LoRA tensor in checkpoint: {name!r}")
