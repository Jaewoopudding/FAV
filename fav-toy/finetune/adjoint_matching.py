"""
Adjoint Matching fine-tuning for MeanFlowModel only.
(DriftModel is not compatible -- no continuous-time ODE structure.)

Algorithm (from adjoint_matching_meanflow.ipynb):
  1. Forward SDE from noise x1 → x_eps using finetuned model as drift
       b_fwd = -(2·u_θ(x, r, t) + κ(t_rev, h)·x)
       x ← x + h·b_fwd + √h·σ(t_rev, h)·ε
  2. Backward adjoint from reward gradient
       adj[-1] = -∇_x log r(x_final)
       adj[i]  = adj[i+1] + h · ∂/∂x[b_base · adj[i+1]]
  3. Loss over selected timesteps
       diff = (2/σ)·(u_θ - u_base) − σ·adj
       L    = E[‖diff‖²]

Note: num_timesteps must be > 1 (t=1 step is degenerate for this algorithm).

Helper functions (reversed-time SDE coefficients):
  η(t, h)  = (1 - t + h) / (t + h)
  σ(t, h)  = √(2η(t, h))
  κ(t, h)  = 1 / (t + h)
where t is the *reversed* time coordinate (small at the start of forward pass).
"""
import copy
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

# SDE coefficients (reversed-time parameterization)

def _eta(t: torch.Tensor, h: float) -> torch.Tensor:
    return (1.0 - t + h) / (t + h)

def _sigma(t: torch.Tensor, h: float) -> torch.Tensor:
    return (2.0 * _eta(t, h)).sqrt()

def _kappa(t: torch.Tensor, h: float) -> torch.Tensor:
    return 1.0 / (t + h)

def finetune(
    model,                    # MeanFlowModel — modified in-place
    device: torch.device,
    energy_fn: Callable,      # (N, D) -> (N,)  unnormalized reward r(x)
    *,
    num_steps: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-4,
    beta: float = 1.0,
    num_timesteps: int = 10,  # must be > 1
    max_grad_norm: float = 1.0,
    metric_every: int = 50,
    save_dir: str = "results/finetune/adjoint_matching",
    dataset_name: str = "data",
    ref_data: Optional[torch.Tensor] = None,
):
    assert num_timesteps > 1, (
        f"Adjoint Matching requires num_timesteps > 1, got {num_timesteps}. "
        "Use --n_steps ≥ 4."
    )
    os.makedirs(save_dir, exist_ok=True)

    # Frozen base model (pretrained weights)
    base_model = copy.deepcopy(model)
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.eval()

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))
    h = 1.0 / num_timesteps

    losses, m_steps, kl_list, mmd_list, reward_list = [], [], [], [], []
    ema = None
    last_kl, last_mmd2, last_reward = None, None, None
    # Best-checkpoint tracking (EMA-smoothed MMD² vs p·r target)
    EMA_ALPHA = 0.3
    ema_mmd2 = None
    best_ema_mmd2 = float("inf")
    best_state = None
    best_step = -1
    pbar = tqdm(range(num_steps), desc="AM Finetune")

    for step in pbar:
        # ── Time schedule ──────────────────────────────────────────
        # t_steps : [1.0, ..., eps, 0.0]  (len = num_timesteps + 2)
        # t_steps_reversed: [eps, ..., 1.0]  (reversed, len = num_timesteps + 1)
        t_steps = torch.linspace(1.0, 0.0, num_timesteps + 1, device=device)
        t_steps[-1] = 1e-5                                          # eps
        t_steps = torch.cat([t_steps, torch.zeros(1, device=device)])
        t_steps_rev = torch.flip(t_steps[:-1], dims=[0])

        bsz = batch_size
        x = torch.randn(bsz, model.data_dim, device=device)
        x_traj = [x]

        # ── Forward SDE (finetuned model, no grad) ─────────────────
        with torch.no_grad():
            for i in range(num_timesteps):
                t_ = torch.full((bsz, 1), t_steps[i].item(),     device=device)
                r_ = torch.full((bsz, 1), t_steps[i+1].item(),   device=device)
                tr = torch.full((bsz, 1), t_steps_rev[i].item(), device=device)

                vt    = model.net(x, r_, t_)
                b_fwd = -(2.0 * vt + _kappa(tr, h) * x)
                x     = x + h * b_fwd + h**0.5 * _sigma(tr, h) * torch.randn_like(x)
                x_traj.append(x.clone())

        # ── Adjoint backward ───────────────────────────────────────
        adj = [None] * (num_timesteps + 1)

        x_T = x_traj[-1].clone().requires_grad_(True)
        log_r = beta * torch.log(energy_fn(x_T).clamp_min(1e-10)).sum()
        adj[-1] = torch.autograd.grad(-log_r, x_T)[0].detach()

        for i in reversed(range(num_timesteps)):
            t_ = torch.full((bsz, 1), t_steps[i+1].item(),     device=device)
            r_ = torch.full((bsz, 1), t_steps[i+2].item(),     device=device)
            tr = torch.full((bsz, 1), t_steps_rev[i+1].item(), device=device)

            xi = x_traj[i+1].detach().requires_grad_(True)
            vt_base = base_model.net(xi, r_, t_)
            b_base  = -(2.0 * vt_base + _kappa(tr, h) * xi)

            Jv    = (b_base * adj[i+1]).sum()
            dJdx  = torch.autograd.grad(Jv, xi)[0]
            adj[i] = (adj[i+1] + h * dJdx).detach()

        # ── Timestep selection ─────────────────────────────────────
        # 25% random from early steps, all late steps (last 25%)
        n_75  = max(1, (num_timesteps * 3) // 4)
        n_sel = max(1, num_timesteps // 4)
        rand_part  = list(np.random.choice(n_75, min(n_sel, n_75), replace=False))
        fixed_part = list(range(n_75, num_timesteps))
        selected   = rand_part + fixed_part

        # ── Loss ───────────────────────────────────────────────────
        opt.zero_grad()
        total_loss = torch.zeros(1, device=device)

        for i in selected:
            t_ = torch.full((bsz, 1), t_steps[i].item(),     device=device)
            r_ = torch.full((bsz, 1), t_steps[i+1].item(),   device=device)
            tr = torch.full((bsz, 1), t_steps_rev[i].item(), device=device)

            xi = x_traj[i].detach()
            vt = model.net(xi, r_, t_)
            with torch.no_grad():
                vt_base = base_model.net(xi, r_, t_)

            sig        = _sigma(tr, h)
            diff       = (2.0 / sig) * (vt - vt_base) - sig * adj[i]
            total_loss = total_loss + (diff ** 2).sum(dim=-1).mean()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

        losses.append(total_loss.item())
        ema = total_loss.item() if ema is None else 0.96 * ema + 0.04 * total_loss.item()
        pbar.set_postfix(loss=f"{ema:.3e}")

        if step % metric_every == 0 and ref_data is not None:
            kl, mmd2, reward_mean = evaluate_metrics_finetune(
                model, ref_data, energy_fn, device, n=256, beta=beta, n_steps=num_timesteps,
            )
            last_kl, last_mmd2, last_reward = kl, mmd2, reward_mean
            m_steps.append(step); kl_list.append(kl); mmd_list.append(mmd2); reward_list.append(reward_mean)
            ema_mmd2 = mmd2 if ema_mmd2 is None else EMA_ALPHA * mmd2 + (1 - EMA_ALPHA) * ema_mmd2
            if ema_mmd2 < best_ema_mmd2:
                best_ema_mmd2 = ema_mmd2
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
            pbar.set_postfix(loss=f"{ema:.3e}", KL=f"{kl:.3f}", MMD2=f"{mmd2:.4f}",
                             r=f"{reward_mean:.3f}", bestEMA=f"{best_ema_mmd2:.4f}")
    if m_steps:
        save_metrics_csv(os.path.join(save_dir, "metrics_finetune.csv"),
                         m_steps, kl_list, mmd_list, reward_vals=reward_list)

    # Save best ckpt separately (model retains LAST weights on return).
    if best_state is not None:
        torch.save(best_state, os.path.join(save_dir, "model_best.pt"))
        print(f"[AM] best EMA-MMD² = {best_ema_mmd2:.6e} @ step {best_step}  "
              f"-> model_best.pt  (model returned with LAST weights)")
    return model
