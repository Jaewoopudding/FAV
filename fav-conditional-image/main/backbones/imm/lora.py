"""LoRA injection for the IMM backbone. Imports are deferred to call-time so
the ``_compat`` sys.modules aliases are in place first."""
from __future__ import annotations

from typing import Iterable, Optional

import torch.nn as nn

from . import _compat


def _imm_dit_module():
    _compat.install_aliases()
    from training import dit
    return dit


def inject_imm_lora(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: Optional[float] = None,
    target_modules: Optional[Iterable[str]] = None,
    dropout: float = 0.0,
):
    """Inject LoRA into IMM DiT (qkv / proj / fc1 / fc2 by default)."""
    dit = _imm_dit_module()
    return dit.inject_lora(
        model,
        rank=rank,
        alpha=alpha,
        target_modules=list(target_modules) if target_modules is not None else None,
        dropout=dropout,
    )


def get_imm_lora_state(model: nn.Module) -> dict:
    return _imm_dit_module().get_lora_state_dict(model)


def load_imm_lora_state(model: nn.Module, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    for name in unexpected:
        raise RuntimeError(f"Unexpected LoRA tensor in checkpoint: {name!r}")
