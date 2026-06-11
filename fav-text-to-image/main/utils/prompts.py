"""Prompt-file loading + round-robin prompt selection."""
from __future__ import annotations

import torch


def load_prompts(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class RoundRobinPromptSampler:
    """Round-robin over ``n_train`` prompt indices with periodic reshuffle.

    Each ``next_step()`` returns ``prompts_per_step`` indices; an identical seed
    across ranks makes all GPUs see the same prompts per step.
    """

    def __init__(self, n_train: int, prompts_per_step: int, *, seed: int = 0) -> None:
        self.n_train = n_train
        self.prompts_per_step = min(prompts_per_step, n_train)
        self.n_chunks = max(n_train // self.prompts_per_step, 1)
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self._perm = torch.arange(n_train)
        self._step = 0

    def next_step(self) -> list[int]:
        chunk_idx = self._step % self.n_chunks
        if chunk_idx == 0:
            self._perm = torch.randperm(self.n_train, generator=self.rng)
        start = chunk_idx * self.prompts_per_step
        end = min(start + self.prompts_per_step, self.n_train)
        self._step += 1
        return self._perm[start:end].tolist()
