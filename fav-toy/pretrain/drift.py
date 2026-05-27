"""
2D Drift model — pre-training module.

compute_drift follows drifting_model_demo.ipynb exactly:
  - Joint kernel over [gen, pos] concatenated
  - Doubly-normalized kernel (row × col), then cross-product weighting
  - Fixed temp=0.05, lr=1e-3, in_dim=32, SiLU activation
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional

from utils_2d import (
    BaseGenerator,
    evaluate_metrics,
    evaluate_metrics_fixed,
    save_metrics_csv,
)

# Core algorithm  (verbatim from drifting_model_demo.ipynb)

def _compute_drift(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temp: float = 0.05,
) -> torch.Tensor:
    """
    Drift field V = V_pos - V_neg.

    Joint kernel over [gen, pos] with doubly-normalized (row × col) weights.
    Cross-product weighting: pos_coeff mixes pos-side with gen-side sum,
    neg_coeff mixes gen-side with pos-side sum.
    """
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]

    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)          # mask self-distances
    kernel = (-dist / temp).exp()

    normalizer = (kernel.sum(dim=-1, keepdim=True) *
                  kernel.sum(dim=-2, keepdim=True))
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    K = kernel / normalizer

    pos_coeff = K[:, G:] * K[:, :G].sum(dim=-1, keepdim=True)
    neg_coeff = K[:, :G] * K[:, G:].sum(dim=-1, keepdim=True)

    return pos_coeff @ targets[G:] - neg_coeff @ targets[:G]

def _drifting_loss(gen: torch.Tensor, pos: torch.Tensor, temp: float) -> torch.Tensor:
    with torch.no_grad():
        drift  = _compute_drift(gen, pos, temp=temp)
        target = (gen + drift).detach()
    return F.mse_loss(gen, target)

# Model  (in_dim=32, SiLU — matches notebook)

class DriftModel(BaseGenerator):
    """MLP: z (in_dim=32) -> x (data_dim=2).  Implements BaseGenerator."""

    def __init__(self, in_dim: int = 32, hidden: int = 256, data_dim: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.in_dim, device=device)
        return self.net(z)

    def pretrain(
        self,
        data: torch.Tensor,
        device: torch.device,
        *,
        num_steps: int = 5000,
        batch_size: int = 2048,
        lr: float = 1e-3,
        temp: float = 0.2,
        metric_every: int = 50,
        metric_samples: int = 4096,
        save_dir: str = "results/drift",
        dataset_name: str = "data",
        ref_data: Optional[torch.Tensor] = None,
    ) -> "DriftModel":
        os.makedirs(save_dir, exist_ok=True)
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        print(f"[Drift] temp={temp}  lr={lr}  batch={batch_size}  steps={num_steps}")

        losses, m_steps, kl_list, mmd_list = [], [], [], []
        ema  = None
        last_kl, last_mmd2 = None, None
        pbar = tqdm(range(num_steps), desc="Drift Pretrain")

        for step in pbar:
            idx = torch.randint(0, data.shape[0], (batch_size,), device=device)
            pos = data[idx]
            gen = self.sample(batch_size, device)

            loss = _drifting_loss(gen, pos, temp)
            opt.zero_grad(); loss.backward(); opt.step()

            losses.append(loss.item())
            ema = loss.item() if ema is None else 0.96 * ema + 0.04 * loss.item()
            pbar.set_postfix(loss=f"{ema:.3e}")

            if step % metric_every == 0:
                if ref_data is not None:
                    kl, mmd2 = evaluate_metrics_fixed(self, ref_data, device)
                else:
                    kl, mmd2 = evaluate_metrics(self, data, device, metric_samples)
                last_kl, last_mmd2 = kl, mmd2
                m_steps.append(step); kl_list.append(kl); mmd_list.append(mmd2)
                pbar.set_postfix(loss=f"{ema:.3e}", KL=f"{kl:.3f}", MMD2=f"{mmd2:.4f}")
        save_metrics_csv(os.path.join(save_dir, "metrics_pretrain.csv"), m_steps, kl_list, mmd_list)
        return self
