"""Backbones for text-to-image alignment (add one by subclassing ``Backbone``)."""
from __future__ import annotations

from .base import Backbone

__all__ = ["Backbone", "build_backbone"]


def build_backbone(cfg_backbone) -> Backbone:
    """Construct a backbone from ``cfg.backbone`` (no weights loaded yet)."""
    name = cfg_backbone.name
    _inter = cfg_backbone.sample.intermediate_timesteps
    inter_ts = None if _inter is None else float(_inter)
    if name == "sana_sprint_hf":
        from .sana_sprint_hf import SanaSprintHFBackbone
        chi = cfg_backbone.get("complex_human_instruction", None)
        return SanaSprintHFBackbone(
            model_repo=cfg_backbone.model_repo,
            image_size=int(cfg_backbone.image_size),
            sample_steps=int(cfg_backbone.sample.sample_steps),
            max_timesteps=float(cfg_backbone.sample.max_timesteps),
            intermediate_timesteps=inter_ts,
            cfg_scale=float(cfg_backbone.sample.cfg_scale),
            weight_dtype=str(cfg_backbone.get("weight_dtype", "bfloat16")),
            vae_dtype=str(cfg_backbone.get("vae_dtype", "float32")),
            max_sequence_length=int(cfg_backbone.get("max_sequence_length", 300)),
            complex_human_instruction=(list(chi) if chi is not None else None),
            use_grad_checkpoint=bool(cfg_backbone.get("use_grad_checkpoint", True)),
        )
    raise NotImplementedError(f"Backbone {name!r} is not implemented.")
