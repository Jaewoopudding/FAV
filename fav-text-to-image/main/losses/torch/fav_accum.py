"""Chunked / gradient-accumulating FAV loss.

Same SVGD gradient as ``amortized_mle_loss``, but the O(N²) particle interaction
runs against a detached global set while the autograd graph is built for only a
``micro_batch_size`` slice of generator particles at a time. Summing the per-chunk
losses (with the trainer's normalization) reproduces the monolithic per-prompt
gradient.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from .fav import _ensure_list


def fav_total_score(
    gen_global: torch.Tensor,
    ref_global: torch.Tensor,
    r_x: Callable[[torch.Tensor], torch.Tensor],
    *,
    beta: float,
    temp_kde: float | Sequence[float],
) -> torch.Tensor:
    """Per-particle SVGD total score ``prior_score + nabla_r * beta`` over the
    full detached global set ``(K, D)``."""
    temps_kde = _ensure_list(temp_kde)

    gen_req = gen_global.detach().clone().requires_grad_(True)
    rewards = r_x(gen_req)
    nabla_r = torch.autograd.grad(rewards.sum(), gen_req, create_graph=False)[0]

    dist_pos = torch.cdist(gen_req, ref_global, p=2) ** 2
    score_numerator = 0
    Z_total = 0
    for t in temps_kde:
        k = (-dist_pos / t).exp()
        Z_t = k.sum(dim=-1, keepdim=True)
        Z_total = Z_total + Z_t
        score_numerator = score_numerator + 2 * (k @ ref_global - Z_t * gen_req) / t
    Z_total = Z_total.clamp_min(1e-6)
    prior_score = score_numerator / Z_total

    total_score = prior_score + nabla_r * beta
    return total_score.detach()


def fav_loss_chunked(
    gen_chunk: torch.Tensor,
    gen_global: torch.Tensor,
    total_score_global: torch.Tensor,
    *,
    temp_stein: float | Sequence[float],
    n_global: int,
) -> torch.Tensor:
    """SVGD pseudo-loss for an ``(m, D)`` slice of live generator particles
    against the detached global set; ``n_global`` is the SVGD normalizer ``K``.
    Gradient matches the corresponding rows of ``amortized_mle_loss``."""
    assert gen_global.shape[0] == n_global, (
        f"n_global={n_global} but gen_global has {gen_global.shape[0]} rows"
    )
    temps_stein = _ensure_list(temp_stein)

    gen_req = gen_chunk.detach().clone().requires_grad_(True)
    dist_gen = torch.cdist(gen_req, gen_global, p=2) ** 2
    kernel_matrix = sum((-dist_gen / t).exp() for t in temps_stein)

    driving_term = (kernel_matrix @ total_score_global) / n_global

    kernel_sum = kernel_matrix.sum()
    grad_k = torch.autograd.grad(kernel_sum, gen_req)[0]
    repulsive_term = -grad_k / (2 * n_global)

    stein_velocity = driving_term + repulsive_term
    return F.mse_loss(gen_chunk, (gen_chunk + stein_velocity).detach())
