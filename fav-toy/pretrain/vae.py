"""
2D VAE model — pre-training module.

Encoder : x (data_dim=2) -> (mu, logvar)  in latent_dim
Decoder : z (latent_dim) -> x_hat (data_dim=2)
Loss    : recon (Gaussian NLL, fixed var) + beta * KL(q(z|x) || N(0, I))
Sampling: z ~ N(0, I), x = decoder(z)  (encoder unused at sample time)
"""
import copy
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

class _Encoder(nn.Module):
    """x -> (mu, logvar)."""

    def __init__(self, data_dim: int = 2, hidden: int = 256, latent_dim: int = 8):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.SiLU(),
            nn.Linear(hidden,   hidden), nn.SiLU(),
            nn.Linear(hidden,   hidden), nn.SiLU(),
        )
        self.mu     = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

class _Decoder(nn.Module):
    """z -> x_hat."""

    def __init__(self, latent_dim: int = 8, hidden: int = 256, data_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden,     hidden), nn.SiLU(),
            nn.Linear(hidden,     hidden), nn.SiLU(),
            nn.Linear(hidden,     data_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

def _vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gaussian NLL (fixed var=1, sum over dims) + beta * KL, averaged over batch."""
    recon = 0.5 * ((x - x_hat) ** 2).sum(dim=-1).mean()
    kl    = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    return recon + beta * kl, recon, kl

class VAEModel(BaseGenerator):
    """VAE with diagonal Gaussian posterior and N(0, I) prior.  Implements BaseGenerator."""

    def __init__(self, latent_dim: int = 8, hidden: int = 256, data_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = _Encoder(data_dim, hidden, latent_dim)
        self.decoder    = _Decoder(latent_dim, hidden, data_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z          = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return self.decoder(z), mu, logvar

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    def pretrain(
        self,
        data: torch.Tensor,
        device: torch.device,
        *,
        num_steps: int = 10_000,
        batch_size: int = 2048,
        lr: float = 1e-3,
        beta: float = 1.0,
        metric_every: int = 100,
        metric_samples: int = 4096,
        save_dir: str = "results/vae",
        dataset_name: str = "data",
        ref_data: Optional[torch.Tensor] = None,
    ) -> "VAEModel":
        os.makedirs(save_dir, exist_ok=True)
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        print(f"[VAE] latent={self.latent_dim}  beta={beta}  lr={lr}  batch={batch_size}  steps={num_steps}")

        losses, recon_log, kl_log        = [], [], []
        m_steps, kl_list, mmd_list       = [], [], []
        ema = None
        last_kl, last_mmd2 = None, None
        ema_mmd2  = None
        ema_alpha = 0.3
        best_ema_mmd2 = float("inf")
        best_state    = None
        best_step     = -1
        pbar = tqdm(range(num_steps), desc="VAE Pretrain")

        for step in pbar:
            idx = torch.randint(0, data.shape[0], (batch_size,), device=device)
            x   = data[idx]

            x_hat, mu, logvar = self(x)
            loss, recon, kl   = _vae_loss(x, x_hat, mu, logvar, beta)

            opt.zero_grad(); loss.backward(); opt.step()

            losses.append(loss.item())
            recon_log.append(recon.item())
            kl_log.append(kl.item())
            ema = loss.item() if ema is None else 0.96 * ema + 0.04 * loss.item()
            pbar.set_postfix(loss=f"{ema:.3e}", recon=f"{recon.item():.3e}", kl=f"{kl.item():.3e}")

            if step % metric_every == 0:
                if ref_data is not None:
                    kl_m, mmd2 = evaluate_metrics_fixed(self, ref_data, device)
                else:
                    kl_m, mmd2 = evaluate_metrics(self, data, device, metric_samples)
                last_kl, last_mmd2 = kl_m, mmd2
                m_steps.append(step); kl_list.append(kl_m); mmd_list.append(mmd2)

                ema_mmd2 = mmd2 if ema_mmd2 is None else ema_alpha * mmd2 + (1 - ema_alpha) * ema_mmd2
                if ema_mmd2 < best_ema_mmd2:
                    best_ema_mmd2 = ema_mmd2
                    best_state    = copy.deepcopy(self.state_dict())
                    best_step     = step
                pbar.set_postfix(loss=f"{ema:.3e}", KL=f"{kl_m:.3f}",
                                 MMD2=f"{mmd2:.4f}", bestEMA=f"{best_ema_mmd2:.4f}")
        if best_state is not None:
            torch.save(best_state, os.path.join(save_dir, "model_best_ema_mmd2.pt"))
            self.load_state_dict(best_state)
            print(f"  Loaded best checkpoint  best_ema_mmd2={best_ema_mmd2:.6e}  @ step {best_step}")

        save_metrics_csv(os.path.join(save_dir, "metrics_pretrain.csv"), m_steps, kl_list, mmd_list)
        return self
