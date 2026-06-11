"""LoRA injection for StyleGAN-XL via the vendored ``lora_utils.py``."""
from __future__ import annotations

from typing import Iterable, Optional

import torch.nn as nn

from . import _compat


def _lora_module():
    _compat.install_aliases()
    from . import _vendor
    from ._vendor import lora_utils
    return lora_utils


def inject_sgxl_lora(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: Optional[float] = None,
    target_modules: Optional[Iterable[str]] = None,
    dropout: float = 0.0,
):
    """Inject LoRA into StyleGAN-XL generator (FullyConnectedLayer only).

    ``target_modules`` is accepted for API symmetry but ignored; conv kernels
    stay frozen. ``alpha=None`` defaults to ``rank``.
    """
    if alpha is None:
        alpha = float(rank)
    return _lora_module().inject_lora(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


def get_sgxl_lora_state(model: nn.Module) -> dict:
    return _lora_module().get_lora_state_dict(model)


def load_sgxl_lora_state(model: nn.Module, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    for name in unexpected:
        raise RuntimeError(f"Unexpected LoRA tensor in checkpoint: {name!r}")
