"""Shared trainer infrastructure: builders, ref-model duplication, eval, ckpt I/O."""
from __future__ import annotations

import copy
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs

from ..backbones.base import ForwardMode, TorchBackbone
from ..rewards import (
    AestheticReward,
    JPEGCompressibilityReward,
    JPEGIncompressibilityReward,
    Reward,
    reward_gradient_estimator,
)
from ..utils.clip import CLIPEncoder
from ..utils.distributed import BatchGenerator
from ..utils.vae import VAEWrapper, IMF_STATS, IMM_STATS


def build_accelerator(*, gradient_accumulation_steps: int, mixed_precision: str = "fp32") -> Accelerator:
    """Construct an Accelerator with DDP find_unused_parameters=True (LoRA-friendly)."""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    mp = "no" if mixed_precision in ("fp32", "no", None) else mixed_precision
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mp,
        kwargs_handlers=[ddp_kwargs],
    )


def build_optimizer(
    params,
    *,
    type: str = "adamw",
    lr: float = 4e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    if type != "adamw":
        raise NotImplementedError(f"optimizer type {type!r} not yet supported")
    return torch.optim.AdamW(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    type: str = "constant",
    warmup_steps: int = 0,
    total_steps: Optional[int] = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear-warmup + (constant | cosine) scheduler."""
    if type not in ("constant", "cosine"):
        raise NotImplementedError(f"scheduler type {type!r} not yet supported")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if type == "constant":
            return 1.0
        if total_steps is None:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_reward(
    reward_cfg,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    shared_clip: Optional[CLIPEncoder] = None,
) -> tuple[Reward, Optional[CLIPEncoder]]:
    """Construct a reward model. Returns ``(reward, clip_encoder)``; CLIP is reused
    as a feature extractor for the SVGD loss."""
    name = reward_cfg.name
    if name == "aesthetic":
        clip_encoder = shared_clip or CLIPEncoder(
            model_name=reward_cfg.get("clip_model", "openai/clip-vit-large-patch14"),
            dtype=dtype,
        ).to(device)
        reward = AestheticReward(
            mlp_ckpt=reward_cfg.mlp_ckpt,
            dtype=dtype,
            clip_encoder=clip_encoder,
        ).to(device)
        reward.requires_grad_(False)
        reward.eval()
        return reward, clip_encoder
    if name == "jpeg_compress":
        return JPEGCompressibilityReward(quality=reward_cfg.get("quality", 95)), None
    if name == "jpeg_incompress":
        return JPEGIncompressibilityReward(quality=reward_cfg.get("quality", 95)), None
    raise ValueError(f"Unknown reward {name!r}")


def build_vae(
    backbone_name: str,
    *,
    decode_batch_size: int = 8,
    dtype: torch.dtype = torch.float32,
    compile_decode: bool = False,
) -> VAEWrapper:
    """Construct the SD-VAE wrapper for the named backbone. Each backbone has its own
    VAE checkpoint and latent normalization stats; do not share an instance across them."""
    if backbone_name == "imf":
        stats = IMF_STATS
    elif backbone_name == "imm":
        stats = IMM_STATS
    else:
        raise ValueError(
            f"build_vae: no VAE stats registered for backbone {backbone_name!r}. "
            "Pixel-space backbones (StyleGAN-XL) should pass vae=None instead."
        )
    vae = VAEWrapper(
        mean=stats["mean"],
        std=stats["std"],
        vae_type=stats["vae_type"],
        decode_batch_size=decode_batch_size,
        dtype=dtype,
        compile_decode=compile_decode,
    )
    if hasattr(vae, "vae"):
        vae.vae.enable_gradient_checkpointing()
        vae.vae.enable_slicing()
    return vae


def build_gradient_estimator(grad_cfg) -> Optional[Callable]:
    """NES gradient estimator if enabled, otherwise None."""
    if not grad_cfg or not grad_cfg.get("enabled", False):
        return None
    return partial(
        reward_gradient_estimator,
        sigma=grad_cfg.get("sigma", 0.01),
        n_samples=grad_cfg.get("num_samples", 16),
        chunk_size=grad_cfg.get("chunk_size", 1),
    )


def clone_for_reference(backbone: TorchBackbone, *, device: torch.device | str) -> TorchBackbone:
    """Deep-copy a backbone to serve as a frozen reference. Must be called BEFORE
    ``backbone.inject_lora(...)`` so the reference has no LoRA params."""
    ref = copy.deepcopy(backbone)
    ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.forward_mode = ForwardMode.SAMPLE
    return ref


class RoundRobinClassSampler:
    """Round-robin over a fixed label set with periodic reshuffle. Same seed across
    ranks so all GPUs see the same labels per step."""

    def __init__(self, labels: torch.Tensor, classes_per_step: int, *, seed: int = 0) -> None:
        if len(labels) % classes_per_step != 0:
            raise ValueError(
                f"len(labels)={len(labels)} must be divisible by classes_per_step={classes_per_step}"
            )
        self.labels = labels
        self.classes_per_step = classes_per_step
        self.n_chunks = len(labels) // classes_per_step
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self._shuffled = self.labels.clone()
        self._step = 0

    def next_step_labels(self) -> torch.Tensor:
        chunk_idx = self._step % self.n_chunks
        if chunk_idx == 0:
            perm = torch.randperm(len(self.labels), generator=self.rng)
            self._shuffled = self.labels[perm]
        out = self._shuffled[
            chunk_idx * self.classes_per_step : (chunk_idx + 1) * self.classes_per_step
        ]
        self._step += 1
        return out


def compute_seed_offsets(
    *,
    global_step: int,
    classes_per_step: int,
    grad_accum_steps: int,
    n_proc: int,
    micro_bsz: int,
    pairs_per_micro: int = 2,
) -> int:
    """Starting seed for one optimizer step. ``pairs_per_micro=2`` for FAV/DRaFT
    (ref + gen), ``=1`` for Flow-GRPO."""
    return global_step * classes_per_step * grad_accum_steps * n_proc * micro_bsz * pairs_per_micro


def evaluate_reward_per_label(
    *,
    backbone: TorchBackbone,
    accelerator: Accelerator,
    labels: torch.Tensor,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    vae: Optional[VAEWrapper],
    sample_kwargs: dict,
    eval_batch_size: int,
    global_seed: int,
) -> dict:
    """Per-label sampling + reward evaluation. Per-label seed is fixed so the same
    noise is used at every eval. Returns aggregated metrics + sample images."""
    rank = accelerator.process_index
    n_proc = accelerator.num_processes
    device = accelerator.device
    eval_per_gpu = max(1, eval_batch_size // n_proc)

    raw = accelerator.unwrap_model(backbone)
    raw.eval()

    all_rewards: list[torch.Tensor] = []
    sample_images: dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for label in labels:
            label_val = int(label.item())
            labels_eval = torch.tensor([label_val] * eval_per_gpu, dtype=torch.int32)

            seed_base = global_seed + label_val * 10_000 + rank * eval_per_gpu
            seed_idx = torch.arange(seed_base, seed_base + eval_per_gpu)
            rng = BatchGenerator(device=device, seeds=seed_idx)

            x_eval = raw._sample(
                n_sample=eval_per_gpu,
                rng=rng,
                labels=labels_eval,
                **sample_kwargs,
            )

            if vae is not None:
                images_eval = vae.decode(x_eval, enable_grad=False)
            else:
                images_eval = x_eval  # pixel-space backbones emit images directly

            rewards_eval = reward_fn(images_eval)
            if not torch.is_tensor(rewards_eval):
                rewards_eval = torch.as_tensor(rewards_eval, device=device, dtype=torch.float32)

            all_rewards.append(accelerator.gather(rewards_eval))

            if accelerator.is_main_process and eval_per_gpu > 0:
                sample_images[label_val] = images_eval[0].detach().cpu()

    raw.train()

    rewards_cat = torch.cat(all_rewards)
    return {
        "reward_mean": rewards_cat.mean().item(),
        "reward_std": rewards_cat.std().item(),
        "reward_max": rewards_cat.max().item(),
        "reward_min": rewards_cat.min().item(),
        "sample_images": sample_images,  # main-process only; empty on others
    }


def save_trainer_ckpt(
    path: str | Path,
    *,
    accelerator: Accelerator,
    backbone: TorchBackbone,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
    cfg: Any,
) -> None:
    """Save LoRA + optimizer + scheduler state on the main process."""
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    raw = accelerator.unwrap_model(backbone)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lora_state": raw.get_lora_state(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "global_step": global_step,
            "cfg": cfg,
        },
        path,
    )


def load_trainer_ckpt(
    path: str | Path,
    *,
    accelerator: Accelerator,
    backbone: TorchBackbone,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    map_location: Any = "cpu",
) -> int:
    """Restore LoRA + optimizer + scheduler. Returns the resumed global_step."""
    state = torch.load(str(path), map_location=map_location)
    raw = accelerator.unwrap_model(backbone)
    raw.load_lora_state(state["lora_state"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state.get("global_step", 0))
