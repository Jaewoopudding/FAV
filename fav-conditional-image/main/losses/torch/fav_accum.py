"""Chunked / gradient-accumulating FAV loss.

Computes the same Stein-VGD gradient as ``fav_loss``, but the O(K^2) particle
interaction runs against a detached global set of K particles while the autograd
graph is built for only one ``micro_batch_size`` chunk of generator particles at
a time. With the trainer's per-chunk normalisation, the summed chunk losses
reproduce the monolithic ``fav_loss`` gradient to floating-point tolerance.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from .fav import _ensure_list


def fav_total_score(
    gen_global: torch.Tensor,
    ref_global: torch.Tensor,
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    beta: float,
    temp_kde: float | Sequence[float],
    gradient_estimator: Optional[Callable] = None,
) -> torch.Tensor:
    """Per-particle ``prior_score + beta * reward_grad`` (K, D) for the detached global set."""
    temps_kde = _ensure_list(temp_kde)

    gen_req = gen_global.detach().clone().requires_grad_(True)

    # Reward gradient ∇r(gen): first-order, or NES. Per-particle independent, so
    # chunking does not change any particle's gradient.
    if gradient_estimator is not None:
        nabla_r = gradient_estimator(gen_req.detach(), reward_fn)
    else:
        rewards = reward_fn(gen_req)
        nabla_r = torch.autograd.grad(rewards.sum(), gen_req, create_graph=False)[0]

    # Multi-scale KDE prior score: ∇_x log Σ_s p̂_{t_s}(x) over the reference set.
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
    """SVGD pseudo-loss for one live ``(m, D)`` chunk against the detached global set.
    ``n_global`` (= K) is the kernel normaliser; the trainer rescales by m/K."""
    assert gen_global.shape[0] == n_global, (
        f"n_global={n_global} but gen_global has {gen_global.shape[0]} rows"
    )
    temps_stein = _ensure_list(temp_stein)

    # Velocity is a detached target; only the final MSE carries grad to the backbone.
    gen_req = gen_chunk.detach().clone().requires_grad_(True)

    dist_gen = torch.cdist(gen_req, gen_global, p=2) ** 2          # (m, K)
    kernel_matrix = sum((-dist_gen / t).exp() for t in temps_stein)

    driving_term = (kernel_matrix @ total_score_global) / n_global  # (m, D)

    kernel_sum = kernel_matrix.sum()
    grad_k = torch.autograd.grad(kernel_sum, gen_req)[0]            # (m, D)
    repulsive_term = -grad_k / (2 * n_global)

    stein_velocity = driving_term + repulsive_term
    return F.mse_loss(gen_chunk, (gen_chunk + stein_velocity).detach())
