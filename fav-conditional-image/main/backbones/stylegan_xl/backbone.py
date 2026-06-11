"""``StyleGANXLBackbone`` — TorchBackbone wrapper around StyleGAN-XL.

Single-pass pixel-space generator (no VAE, no iterative solver); ``_sample``
runs ``z -> mapping -> w -> synthesis -> image`` end-to-end.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F

from ..base import ForwardMode, TorchBackbone
from ...utils import lora_registry
from . import _compat
from .lora import inject_sgxl_lora, get_sgxl_lora_state, load_sgxl_lora_state


class StyleGANXLBackbone(TorchBackbone):
    """StyleGAN-XL pixel-space generator backbone."""

    name = "stylegan_xl"
    supports_sample_mode = True
    supports_velocity_mode = False

    def __init__(
        self,
        *,
        img_resolution: int = 256,
        label_dim: int = 1000,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        _compat.install_aliases()
        self.G: Optional[torch.nn.Module] = None
        self.dtype = dtype
        self.img_resolution = img_resolution
        self.label_dim = label_dim

    def load_pretrained(self, ckpt_path: str | Path) -> None:
        """Load the EMA generator from a StyleGAN-XL pickle."""
        _compat.install_aliases()
        import pickle
        import dnnlib  # type: ignore

        path = Path(ckpt_path)
        with dnnlib.util.open_url(str(path), verbose=False) as f:
            data = pickle.load(f)
        G_src = data.get("G_ema", data.get("G", None))
        if G_src is None:
            raise RuntimeError(
                f"No G_ema/G found in {path}; keys present: {sorted(data.keys())}"
            )
        G = copy.deepcopy(G_src)
        del data, G_src
        self.G = G
        self.img_resolution = int(getattr(G, "img_resolution", self.img_resolution))
        self.label_dim = int(getattr(G, "c_dim", self.label_dim))
        self.z_dim = int(getattr(G, "z_dim", 64))
        self.num_ws = int(getattr(G.mapping, "num_ws", 1))

    def inject_lora(
        self,
        rank: int = 4,
        *,
        alpha: Optional[float] = None,
        target_modules: Optional[Iterable[str]] = None,
        dropout: float = 0.0,
    ) -> tuple[int, int]:
        assert self.G is not None, "Call load_pretrained() first"
        # vendored inject_lora may return (model, n_train, n_total) or (n_train, n_total)
        result = inject_sgxl_lora(
            self.G, rank=rank, alpha=alpha,
            target_modules=target_modules, dropout=dropout,
        )
        if isinstance(result, tuple) and len(result) == 3:
            _, n_train, n_total = result
        elif isinstance(result, tuple) and len(result) == 2:
            n_train, n_total = result
        else:
            n_train = sum(p.numel() for p in self.G.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.G.parameters())
        return n_train, n_total

    def get_lora_state(self) -> dict:
        return get_sgxl_lora_state(self.G)

    def load_lora_state(self, state: dict) -> None:
        load_sgxl_lora_state(self.G, state)

    def _sample(
        self,
        *,
        n_sample: int,
        rng,
        labels: Optional[torch.Tensor] = None,
        truncation_psi: float = 1.0,
        noise_mode: str = "const",
        **_unused: Any,
    ) -> torch.Tensor:
        """``z -> mapping -> truncated w -> synthesis -> image`` in one shot.

        ``truncation_psi`` controls the truncation trick (not CFG; StyleGAN-XL
        has none). Returns a pixel-space tensor ``(B, 3, H, W)`` in ``[-1, 1]``.
        """
        assert self.G is not None, "Call load_pretrained() first"
        device = next(self.G.parameters()).device
        psi = float(truncation_psi)

        z = rng.randn((n_sample, self.z_dim)).to(self.dtype)

        # labels=None: use the average w across classes (no conditioning)
        if labels is not None and self.label_dim > 0:
            labels_long = labels.long().to(device)
            class_labels = F.one_hot(labels_long, self.label_dim).to(z.dtype)
            w_avg = self.G.mapping.w_avg.index_select(0, labels_long)
        else:
            class_labels = None
            w_avg = self.G.mapping.w_avg.unsqueeze(0).expand(n_sample, -1)

        w = self.G.mapping(z, class_labels)
        # truncation around per-class w_avg
        w_avg = w_avg.unsqueeze(1).repeat(1, self.num_ws, 1)
        w = w_avg + (w - w_avg) * psi
        return self.G.synthesis(w, noise_mode=noise_mode)


lora_registry.register(
    "stylegan_xl",
    inject=inject_sgxl_lora,
    get_state=get_sgxl_lora_state,
    load_state=load_sgxl_lora_state,
)
