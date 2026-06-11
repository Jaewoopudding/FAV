"""FAV loss — amortized MLE via Stein-Variational Gradient Descent.

Pseudo-loss ``MSE(gen, stop_grad(gen + stein_velocity))`` so that
``∇_θ L = -stein_velocity / batch`` pushes generator features along the SVGD
direction. Reward gradient enters as ``nabla_r * beta``.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F


def _ensure_list(temp) -> list[float]:
    if isinstance(temp, (int, float)):
        return [float(temp)]
    return list(temp)


def amortized_mle_loss(
    gen: torch.Tensor,
    ref: torch.Tensor,
    r_x: Callable[[torch.Tensor], torch.Tensor],
    beta: float,
    temp_kde: float | Sequence[float],
    temp_stein: float | Sequence[float],
    accelerator,
) -> torch.Tensor:
    """Multi-scale Stein-VGD amortized MLE loss over the gathered global set."""
    temps_kde = _ensure_list(temp_kde)
    temps_stein = _ensure_list(temp_stein)

    global_ref = accelerator.gather(ref).detach()
    global_gen = accelerator.gather(gen).detach()
    n_global = global_gen.shape[0]

    gen_req = gen.detach().clone().requires_grad_(True)

    rewards = r_x(gen_req)
    nabla_r = torch.autograd.grad(rewards.sum(), gen_req, create_graph=False)[0]

    dist_pos = torch.cdist(gen_req, global_ref, p=2) ** 2
    score_numerator = 0
    Z_total = 0
    for t in temps_kde:
        k = (-dist_pos / t).exp()
        Z_t = k.sum(dim=-1, keepdim=True)
        Z_total = Z_total + Z_t
        score_numerator = score_numerator + 2 * (k @ global_ref - Z_t * gen_req) / t
    Z_total = Z_total.clamp_min(1e-6)
    prior_score_local = score_numerator / Z_total

    total_score_local = prior_score_local + nabla_r * beta
    global_total_score = accelerator.gather(total_score_local).detach()

    dist_gen = torch.cdist(gen_req, global_gen, p=2) ** 2
    kernel_matrix = sum((-dist_gen / t).exp() for t in temps_stein)

    driving_term = (kernel_matrix @ global_total_score) / n_global

    kernel_sum = kernel_matrix.sum()
    grad_k = torch.autograd.grad(kernel_sum, gen_req)[0]
    repulsive_term = -grad_k / (2 * n_global)

    stein_velocity = driving_term + repulsive_term
    return F.mse_loss(gen, (gen + stein_velocity).detach())
