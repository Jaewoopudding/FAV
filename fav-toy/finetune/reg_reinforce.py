"""Regularized REINFORCE / VarGrad-trajectory-balance fine-tuning for MeanFlow.

Target: pi(x) ~ p_ref(x) * r(x)^beta. Sample an SDE trajectory under the current
policy, compute REINFORCE coefficients from a VarGrad log-Z estimate against the
frozen reference, then minimize E[coef * sum_t log p_pi(x_{t-1}|x_t)].
"""
import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
from typing import Callable, Optional

from utils_2d import (
    evaluate_metrics_finetune,
    save_metrics_csv,
)

_T_CLAMP_MAX = 0.9999

def _noise_std(t: torch.Tensor, noise_level: float) -> torch.Tensor:
    t_safe = t.clamp(max=_T_CLAMP_MAX)
    return noise_level * (t_safe / (1.0 - t_safe)).sqrt()

def _sde_step_with_logprob(
    v_t:        torch.Tensor,
    x:          torch.Tensor,
    t:          torch.Tensor,
    dt:         torch.Tensor,
    noise_level: float,
    prev_x:     Optional[torch.Tensor] = None,
) -> tuple:
    """One reverse SDE step. If prev_x given, evaluate log p(prev_x|x) instead of sampling."""
    v_t = v_t.float();  x = x.float()
    if prev_x is not None:
        prev_x = prev_x.float()

    sigma    = _noise_std(t, noise_level)
    sqrt_dt  = dt.abs().sqrt()
    std_step = sigma * sqrt_dt
    mean     = x - v_t * dt
    D        = x.shape[-1]

    if prev_x is None:
        noise  = torch.randn_like(x)
        x_next = mean + std_step * noise
    else:
        x_next = prev_x
        noise  = (prev_x - mean) / std_step.clamp_min(1e-8)

    log_prob = (
        -0.5 * (noise ** 2).sum(dim=-1)
        - D * (std_step.squeeze(-1).log() + 0.5 * math.log(2 * math.pi))
    )
    return x_next, log_prob, mean, sigma

@torch.no_grad()
def _sample_trajectory(
    model,
    n:          int,
    device:     torch.device,
    n_steps:    int,
    noise_level: float,
) -> tuple:
    t_steps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    x = torch.randn(n, model.data_dim, device=device)

    latents_list, next_latents_list, log_probs_list = [], [], []

    for i in range(n_steps):
        t_ = torch.full((n, 1), t_steps[i].item(),     device=device)
        r_ = torch.full((n, 1), t_steps[i+1].item(),   device=device)
        dt_ = t_ - r_

        v_t = model.net(x, r_, t_)
        x_next, log_prob, _, _ = _sde_step_with_logprob(v_t, x, t_, dt_, noise_level)

        latents_list.append(x)
        next_latents_list.append(x_next)
        log_probs_list.append(log_prob)
        x = x_next

    return (
        x,
        torch.stack(log_probs_list,    dim=1),
        torch.stack(latents_list,      dim=1),
        torch.stack(next_latents_list, dim=1),
    )

@torch.no_grad()
def _compute_ref_logprobs(
    model_ref,
    latents:      torch.Tensor,
    next_latents: torch.Tensor,
    n_steps:      int,
    noise_level:  float,
    device:       torch.device,
) -> torch.Tensor:
    B, T, _ = latents.shape
    t_steps  = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    log_probs = []

    for j in range(T):
        t_ = torch.full((B, 1), t_steps[j].item(),   device=device)
        r_ = torch.full((B, 1), t_steps[j+1].item(), device=device)
        dt_ = t_ - r_

        v_t = model_ref.net(latents[:, j].detach(), r_, t_)
        _, log_prob, _, _ = _sde_step_with_logprob(
            v_t, latents[:, j].detach(), t_, dt_, noise_level,
            prev_x=next_latents[:, j].detach(),
        )
        log_probs.append(log_prob)

    return torch.stack(log_probs, dim=1)

def _recompute_logprobs(
    model,
    latents:      torch.Tensor,
    next_latents: torch.Tensor,
    n_steps:      int,
    noise_level:  float,
    device:       torch.device,
) -> torch.Tensor:
    """Re-evaluate log p_pi(x_{t-1}|x_t) WITH gradients for loss backprop."""
    B, T, _ = latents.shape
    t_steps  = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    log_probs = []

    for j in range(T):
        t_ = torch.full((B, 1), t_steps[j].item(),   device=device)
        r_ = torch.full((B, 1), t_steps[j+1].item(), device=device)
        dt_ = t_ - r_

        v_t = model.net(latents[:, j].detach(), r_, t_)
        _, log_prob, _, _ = _sde_step_with_logprob(
            v_t, latents[:, j].detach(), t_, dt_, noise_level,
            prev_x=next_latents[:, j].detach(),
        )
        log_probs.append(log_prob)

    return torch.stack(log_probs, dim=1)

def finetune(
    model,
    device:         torch.device,
    energy_fn:      Callable,
    *,
    num_steps:      int   = 5000,
    batch_size:     int   = 256,
    lr:             float = 1e-4,
    beta:           float = 1.0,
    n_steps:        int   = 4,
    noise_level:    float = 0.1,
    max_grad_norm:  float = 1.0,
    metric_every:   int   = 50,
    metric_samples: int   = 4096,
    save_dir:       str   = "results/finetune/reg_reinforce",
    dataset_name:   str   = "data",
    ref_data:       Optional[torch.Tensor] = None,
) -> object:
    assert hasattr(model, "net") and hasattr(model, "data_dim"), \
        "Regularized Reinforce requires MeanFlowModel (.net, .data_dim)."
    assert n_steps > 1, "n_steps must be > 1 for trajectory-based methods."

    os.makedirs(save_dir, exist_ok=True)

    model_ref = copy.deepcopy(model)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    print(
        f"[Reg. Reinforce] n_steps={n_steps}  noise_level={noise_level}  "
        f"beta={beta}  batch={batch_size}"
    )

    losses, m_steps, kl_list, mmd_list, reward_list = [], [], [], [], []
    ema_loss = None
    last_kl, last_mmd2, last_reward = None, None, None
    EMA_ALPHA = 0.3
    ema_mmd2 = None
    best_ema_mmd2 = float("inf")
    best_state = None
    best_step = -1
    pbar = tqdm(range(num_steps), desc="Reg. Reinforce")

    for step in pbar:

        # PHASE 1: SAMPLING  (no grad)
        model.eval()
        with torch.no_grad():
            x_final, log_probs_pi, latents, next_latents = _sample_trajectory(
                model, batch_size, device, n_steps, noise_level,
            )
            log_pf_pi = log_probs_pi.sum(dim=1)        # (B,)

            log_probs_ref = _compute_ref_logprobs(
                model_ref, latents, next_latents, n_steps, noise_level, device,
            )
            log_pf_ref = log_probs_ref.sum(dim=1)      # (B,)

            # Reward: β · log r(x_final)
            log_r = beta * energy_fn(x_final).clamp_min(1e-10).log()   # (B,)

        # VarGrad log-Z and coefficient (all detached)
        with torch.no_grad():
            # Monte Carlo estimate of log Z
            log_Z = (log_pf_pi + log_r - log_pf_ref).mean()            # scalar

            # Residual from balance condition → REINFORCE advantage
            coef = (log_pf_pi + log_Z - log_pf_ref - log_r).detach()  # (B,)

            # Batch-normalize (variance reduction)
            coef = (coef - coef.mean()) / (coef.std() + 1e-8)         # (B,)

        # PHASE 2: TRAINING  (gradient through log_pf_pi_new)
        model.train()
        opt.zero_grad()

        log_probs_new = _recompute_logprobs(
            model, latents, next_latents, n_steps, noise_level, device,
        )                                           # (B, T)
        log_pf_pi_new = log_probs_new.sum(dim=1)   # (B,)

        # Regularized Reinforce loss:  E[ coef · log_pf_pi_new ]
        #   coef > 0  →  suppress (too-probable / low-reward trajectory)
        #   coef < 0  →  amplify  (under-probable / high-reward trajectory)
        loss = (coef * log_pf_pi_new).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

        # ── Logging ───────────────────────────────────────────────────
        loss_val = loss.item()
        losses.append(loss_val)
        ema_loss = loss_val if ema_loss is None else 0.96 * ema_loss + 0.04 * loss_val
        pbar.set_postfix(
            loss=f"{ema_loss:.3e}",
            logZ=f"{log_Z.item():.2f}",
            r=f"{log_r.mean().item():.3f}",
        )

        if (step % metric_every == 0 or step == num_steps - 1) and ref_data is not None:
            kl, mmd2, reward_mean = evaluate_metrics_finetune(
                model, ref_data, energy_fn, device,
                n=metric_samples, beta=beta, n_steps=n_steps,
            )
            last_kl, last_mmd2, last_reward = kl, mmd2, reward_mean
            m_steps.append(step)
            kl_list.append(kl); mmd_list.append(mmd2); reward_list.append(reward_mean)
            ema_mmd2 = mmd2 if ema_mmd2 is None else EMA_ALPHA * mmd2 + (1 - EMA_ALPHA) * ema_mmd2
            if ema_mmd2 < best_ema_mmd2:
                best_ema_mmd2 = ema_mmd2
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
            pbar.set_postfix(
                loss=f"{ema_loss:.3e}",
                KL=f"{kl:.3f}", MMD2=f"{mmd2:.4f}", r=f"{reward_mean:.3f}",
                bestEMA=f"{best_ema_mmd2:.4f}",
            )
    # Save final (last) model checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, "model_finetuned.pt"))
    # Save best-EMA-MMD² checkpoint separately
    if best_state is not None:
        torch.save(best_state, os.path.join(save_dir, "model_best.pt"))
        print(f"[Reg. Reinforce] best EMA-MMD² = {best_ema_mmd2:.6e} @ step "
              f"{best_step}  -> model_best.pt")

    if m_steps:
        save_metrics_csv(os.path.join(save_dir, "metrics_finetune.csv"),
                         m_steps, kl_list, mmd_list, reward_vals=reward_list)
    return model
