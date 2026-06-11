"""``IMFBackbone`` — TorchBackbone wrapper around iMeanFlow."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from ..base import ForwardMode, TorchBackbone
from ...utils import lora_registry
from .flow import iMeanFlow
from .lora import inject_imf_lora, get_imf_lora_state, load_imf_lora_state


class IMFBackbone(TorchBackbone):
    """iMeanFlow backbone with unified forward dispatcher."""

    name = "imf"
    supports_sample_mode = True
    supports_velocity_mode = True

    def __init__(
        self,
        *,
        model_str: str = "MiT_XL_2",
        dtype: torch.dtype = torch.float32,
        img_size: int = 32,
        img_channels: int = 4,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.flow = iMeanFlow(
            model_str=model_str,
            dtype=dtype,
            img_size=img_size,
            img_channels=img_channels,
            num_classes=num_classes,
            eval_mode=True,
        )
        self.dtype = dtype
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

    def load_pretrained(self, ckpt_path: str | Path) -> None:
        """Load iMF weights into the underlying MiT network."""
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif isinstance(state, dict) and "net" in state and isinstance(state["net"], dict):
            state = state["net"]
        cleaned = {}
        for k, v in state.items():
            kk = k
            for prefix in ("module.", "net."):
                if kk.startswith(prefix):
                    kk = kk[len(prefix):]
            cleaned[kk] = v
        missing, unexpected = self.flow.net.load_state_dict(cleaned, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading iMF checkpoint: {sorted(unexpected)[:5]}..."
            )

    def inject_lora(
        self,
        rank: int = 4,
        *,
        alpha: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        dropout: float = 0.0,
    ) -> tuple[int, int]:
        """Inject LoRA into MiT attention projections. Returns (n_trainable, n_total)."""
        _, n_train, n_total = inject_imf_lora(
            self.flow,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            dropout=dropout,
        )
        return n_train, n_total

    def get_lora_state(self) -> dict:
        return get_imf_lora_state(self.flow)

    def load_lora_state(self, state: dict) -> None:
        load_imf_lora_state(self.flow, state)

    def _sample(
        self,
        *,
        n_sample: int,
        rng,
        num_steps: int,
        cfg_omega,
        interval_min,
        interval_max,
        labels: Optional[torch.Tensor] = None,
        **_unused: Any,
    ) -> torch.Tensor:
        """Full few-step generation. Differentiable when grad is enabled."""
        return self.flow.generate(
            n_sample=n_sample,
            rng=rng,
            num_steps=num_steps,
            omega=cfg_omega,
            t_min=interval_min,
            t_max=interval_max,
            labels=labels,
        )

    def _velocity(self, *, x, t, h, omega, t_min, t_max, y, **_unused: Any):
        """Single-step velocity prediction."""
        return self.flow.u_fn(x, t, h, omega, t_min, t_max, y)


lora_registry.register(
    "imf",
    inject=inject_imf_lora,
    get_state=get_imf_lora_state,
    load_state=load_imf_lora_state,
)
