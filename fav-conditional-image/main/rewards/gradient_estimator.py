"""Antithetic NES zeroth-order gradient estimator for non-differentiable rewards."""
from __future__ import annotations

from typing import Callable

import torch


def reward_gradient_estimator(
    features: torch.Tensor,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    sigma: float = 0.01,
    n_samples: int = 16,
    chunk_size: int = 1,
) -> torch.Tensor:
    """Estimate ``∇ E[r(features + σ·noise)]`` by antithetic NES, returning ``(B, D)``."""
    B, D = features.shape
    nabla_r = torch.zeros_like(features)

    with torch.no_grad():
        for i in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - i)
            # antithetic Gaussian perturbations u ~ N(0, I)
            u = torch.randn(k, B, D, device=features.device, dtype=features.dtype)

            z_plus = (features.unsqueeze(0) + sigma * u).reshape(k * B, D)
            z_minus = (features.unsqueeze(0) - sigma * u).reshape(k * B, D)

            r_plus = reward_fn(z_plus).reshape(k, B)
            r_minus = reward_fn(z_minus).reshape(k, B)

            # central finite difference [r(x+σu) - r(x-σu)] / 2σ, projected onto u
            diff = ((r_plus - r_minus) / (2 * sigma)).unsqueeze(-1)
            nabla_r += (diff * u).sum(dim=0)

        nabla_r /= n_samples  # Monte-Carlo average over N perturbations

    return nabla_r
