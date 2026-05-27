"""
2D MeanFlow model — pre-training module.

Time convention: t=0 -> data, t=1 -> noise.
Final sampling uses n_steps=64 (set by pretrain.py).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
from tqdm import tqdm
from typing import Optional

from utils_2d import (
    BaseGenerator,
    evaluate_metrics,
    evaluate_metrics_fixed,
    save_metrics_csv,
)

def _sample_t_r(
    batch_size: int,
    fraction_equal: float = 0.75,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    raw  = torch.sigmoid(torch.normal(-0.4, 1.0, (batch_size, 2), device=device))
    t    = torch.max(raw[:, 0], raw[:, 1]).unsqueeze(1)
    r    = torch.min(raw[:, 0], raw[:, 1]).unsqueeze(1)
    mask = torch.rand(batch_size, 1, device=device) < fraction_equal
    return t, torch.where(mask, t, r)

def _pretrain_loss(
    net: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    adp_p: float = 1.0,
) -> torch.Tensor:
    """u_target = v_t - (t-r)*du/dt via JVP. `adp_p` is the adaptive-weighting
    power (loss / (loss.detach()+0.01)^adp_p); 0.0 disables it."""
    x_t = (1 - t) * x0 + t * x1
    v_t = x1 - x0
    _, dudt = jvp(
        func=net,
        inputs=(x_t, r, t),
        v=(v_t, torch.zeros_like(r), torch.ones_like(t)),
        create_graph=False,
    )
    u_target = (v_t - (t - r) * dudt).detach()
    loss = F.mse_loss(net(x_t, r, t), u_target)
    if adp_p > 0:
        loss = loss / ((loss.detach() + 0.01) ** adp_p)
    return loss

class _MeanFlowMLP(nn.Module):
    """u(x_t, r, t): [x_t (D) | t (1) | t-r (1)] -> u (D)."""

    def __init__(
        self,
        data_dim: int = 2,
        hidden: int = 256,
        depth: int = 3,
        activation: str = "silu",
    ):
        super().__init__()
        act = nn.SiLU if activation == "silu" else nn.ReLU
        layers = [nn.Linear(data_dim + 2, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, r, t):
        return self.net(torch.cat([z, t, t - r], dim=-1))

class MeanFlowModel(BaseGenerator):
    """MeanFlow ODE model: x_r = x_t - (t-r)*u(x_t, r, t), t=1->0."""

    def __init__(
        self,
        data_dim: int = 2,
        hidden: int = 256,
        depth: int = 3,
        activation: str = "silu",
    ):
        super().__init__()
        self.data_dim = data_dim
        self.net = _MeanFlowMLP(data_dim, hidden, depth=depth, activation=activation)

    def forward(self, z, r, t):
        return self.net(z, r, t)

    def sample(
        self,
        n: int,
        device: torch.device,
        n_steps: int = 10,
        t_steps: torch.Tensor = None,
    ) -> torch.Tensor:
        x = torch.randn(n, self.data_dim, device=device)
        if t_steps is None:
            t_steps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        for i in range(n_steps):
            t_ = torch.full((n, 1), t_steps[i].item(),     device=device)
            r_ = torch.full((n, 1), t_steps[i + 1].item(), device=device)
            x  = x - self.net(x, r_, t_) * (t_ - r_)
        return x

    def pretrain(
        self,
        data: torch.Tensor,
        device: torch.device,
        *,
        num_steps: int = 20_000,
        batch_size: int = 512,
        lr: float = 1e-4,
        fraction_equal: float = 0.75,
        adp_p: float = 1.0,
        max_grad_norm: float = 1.0,
        metric_every: int = 200,
        metric_samples: int = 4096,
        save_dir: str = "results/meanflow",
        dataset_name: str = "data",
        ref_data: Optional[torch.Tensor] = None,
        monitor_n_steps: int = 1,
    ) -> "MeanFlowModel":
        os.makedirs(save_dir, exist_ok=True)
        opt = torch.optim.AdamW(self.parameters(), lr=lr,
                                betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)

        losses, m_steps, kl_list, mmd_list = [], [], [], []
        ema  = None
        last_kl, last_mmd2 = None, None
        pbar = tqdm(range(num_steps), desc="MeanFlow Pretrain")

        for step in pbar:
            idx  = torch.randint(0, data.shape[0], (batch_size,), device=device)
            x0   = data[idx]
            x1   = torch.randn_like(x0)
            t, r = _sample_t_r(batch_size, fraction_equal, device)

            opt.zero_grad()
            loss = _pretrain_loss(self.net, x0, x1, t, r, adp_p=adp_p)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            opt.step()

            losses.append(loss.item())
            ema = loss.item() if ema is None else 0.96 * ema + 0.04 * loss.item()
            pbar.set_postfix(loss=f"{ema:.4f}")

            if step % metric_every == 0:
                if ref_data is not None:
                    kl, mmd2 = evaluate_metrics_fixed(self, ref_data, device,
                                                     n_steps=monitor_n_steps)
                else:
                    kl, mmd2 = evaluate_metrics(self, data, device, metric_samples)
                last_kl, last_mmd2 = kl, mmd2
                m_steps.append(step); kl_list.append(kl); mmd_list.append(mmd2)
                pbar.set_postfix(loss=f"{ema:.4f}", KL=f"{kl:.3f}", MMD2=f"{mmd2:.4f}")
        save_metrics_csv(os.path.join(save_dir, "metrics_pretrain.csv"), m_steps, kl_list, mmd_list)
        return self
