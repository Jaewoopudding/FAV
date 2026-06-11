"""Per-sample seeded RNG and seeding helpers (DDP handled by Accelerator)."""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch


class BatchGenerator:
    """Per-sample seeded generator: noise stays reproducible across DDP ranks."""

    def __init__(self, device: torch.device | str, seeds: Sequence[int]) -> None:
        self.device = device
        self.generators = [
            torch.Generator("cpu").manual_seed(int(s) % (1 << 32)) for s in seeds
        ]

    def randn(self, size: tuple[int, ...], **kwargs) -> torch.Tensor:
        assert size[0] == len(self.generators), (
            f"size[0]={size[0]} but {len(self.generators)} seeds were given"
        )
        # Generators are CPU-resident; drop any device kwarg and move after sampling.
        kwargs.pop("device", None)
        return torch.stack([
            torch.randn(size[1:], generator=g, **kwargs).to(self.device)
            for g in self.generators
        ])

    def randn_like(self, x: torch.Tensor) -> torch.Tensor:
        return self.randn(tuple(x.shape), dtype=x.dtype, layout=x.layout)

    def randint(self, *args, size: tuple[int, ...], **kwargs) -> torch.Tensor:
        assert size[0] == len(self.generators)
        kwargs.pop("device", None)
        return torch.stack([
            torch.randint(*args, size=size[1:], generator=g, **kwargs).to(self.device)
            for g in self.generators
        ])


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
