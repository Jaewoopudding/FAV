"""``IMMBackbone`` — TorchBackbone wrapper around IMMPrecond/DiT."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from ..base import ForwardMode, TorchBackbone
from ...utils import lora_registry
from . import _compat
from .lora import inject_imm_lora, get_imm_lora_state, load_imm_lora_state
from .sampler import generator_fn


_VENDOR_CONFIG_DIR = Path(__file__).resolve().parent / "_vendor" / "configs"


def _load_config(config_name: str = "im256_generate_images.yaml"):
    from omegaconf import OmegaConf
    path = _VENDOR_CONFIG_DIR / config_name
    if not path.exists():
        raise FileNotFoundError(f"IMM config not found: {path}")
    return OmegaConf.load(path)


class IMMBackbone(TorchBackbone):
    """IMM (Inductive Moment Matching) backbone."""

    name = "imm"
    supports_sample_mode = True
    supports_velocity_mode = False

    def __init__(
        self,
        *,
        config_name: str = "im256_generate_images.yaml",
        temb_type: str = "identity",
        img_resolution: int = 32,
        img_channels: int = 4,
        label_dim: int = 1000,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        _compat.install_aliases()
        import dnnlib  # type: ignore

        from omegaconf import OmegaConf
        cfg = _load_config(config_name)
        if cfg.get("network") is not None:
            cfg.network.temb_type = temb_type
        # resolve interpolations before construct_class_by_name
        cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

        interface = dict(
            img_resolution=img_resolution,
            img_channels=img_channels,
            label_dim=label_dim,
        )
        self.net = dnnlib.util.construct_class_by_name(**cfg.network, **interface)
        self.cfg = cfg
        self.config_name = config_name
        self.dtype = dtype
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim

    def load_pretrained(self, ckpt_path: str | Path) -> None:
        """Load an EMA pickle and copy its parameters into ``self.net``."""
        _compat.install_aliases()
        import pickle
        import dnnlib  # type: ignore
        from torch_utils import misc  # type: ignore

        path = Path(ckpt_path)
        with dnnlib.util.open_url(str(path), verbose=False) as f:
            data = pickle.load(f)
        ema = data["ema"].cpu()
        misc.copy_params_and_buffers(src_module=ema, dst_module=self.net, require_all=True)
        del data, ema

    def inject_lora(
        self,
        rank: int = 4,
        *,
        alpha: Optional[float] = None,
        target_modules: Optional[Iterable[str]] = None,
        dropout: float = 0.0,
    ) -> tuple[int, int]:
        _, n_train, n_total = inject_imm_lora(
            self.net,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            dropout=dropout,
        )
        return n_train, n_total

    def get_lora_state(self) -> dict:
        return get_imm_lora_state(self.net)

    def load_lora_state(self, state: dict) -> None:
        load_imm_lora_state(self.net, state)

    def _sample(
        self,
        *,
        n_sample: int,
        rng,
        labels: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        cfg_omega: float = 1.5,
        discretization: str = "uniform",
        **_unused: Any,
    ) -> torch.Tensor:
        """Few-step IMM sampling. Differentiable when grad is enabled."""
        cfg_scale = float(cfg_omega)
        device = next(self.parameters()).device
        shape = (n_sample, self.img_channels, self.img_resolution, self.img_resolution)

        if hasattr(self.net, "get_init_noise"):
            latents = self.net.get_init_noise(shape, device, rng)
        else:
            latents = rng.randn(shape).to(self.dtype).to(device)

        if labels is not None and self.label_dim > 0:
            class_labels = torch.nn.functional.one_hot(
                labels.long().to(device), num_classes=self.label_dim,
            ).to(device)
        else:
            class_labels = None

        return generator_fn(
            self.net, latents, class_labels,
            name="pushforward_generator_fn",
            discretization=discretization,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
        )


lora_registry.register(
    "imm",
    inject=inject_imm_lora,
    get_state=get_imm_lora_state,
    load_state=load_imm_lora_state,
)
