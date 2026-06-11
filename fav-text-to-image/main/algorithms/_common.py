"""Shared trainer infrastructure: accelerator / optimizer / scheduler builders."""
from __future__ import annotations

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs


def mixed_precision_mode(weight_dtype: torch.dtype) -> str:
    """Map a torch dtype to accelerate's mixed-precision string."""
    return {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "no"}[weight_dtype]


def build_accelerator(*, gradient_accumulation_steps: int, mixed_precision: str = "no") -> Accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )


def build_optimizer(params, *, lr: float, betas=(0.9, 0.95), weight_decay: float = 0.01):
    return torch.optim.AdamW(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)


def build_scheduler(optimizer, *, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_n_gpu() -> int:
    import os
    return int(os.environ.get("WORLD_SIZE") or torch.cuda.device_count() or 1)
