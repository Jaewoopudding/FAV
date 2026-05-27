"""Shared utilities: 8gaussians dataset, energy, metrics, CSV logging."""
import csv
import os
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde

class BaseGenerator(ABC, nn.Module):
    @abstractmethod
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        ...

def get_toy_data(n: int) -> Tuple[np.ndarray, np.ndarray]:
    scale = 4.0
    centers = [
        (0, 1), (-1/np.sqrt(2), 1/np.sqrt(2)), (-1, 0), (-1/np.sqrt(2), -1/np.sqrt(2)),
        (0, -1), (1/np.sqrt(2), -1/np.sqrt(2)), (1, 0), (1/np.sqrt(2), 1/np.sqrt(2)),
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    dataset, labels = [], []
    for _ in range(n):
        idx = np.random.randint(8)
        point = np.random.randn(2) * 0.5 + np.array(centers[idx])
        dataset.append(point)
        labels.append(idx)
    data = np.array(dataset, dtype="float32") / np.sqrt(2)
    energy = (np.array(labels, dtype="float32") / 7.0)[:, None]
    return data, energy

def load_dataset(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    data_np, energy_np = get_toy_data(n)
    return (
        torch.tensor(data_np, dtype=torch.float32, device=device),
        torch.tensor(energy_np, dtype=torch.float32, device=device),
    )

class EightGaussiansEnergy:
    """Cluster k of the 8gaussians dataset gets reward k/7 (softmax-smoothed)."""
    _SCALE = 4.0
    _RAW = [(0,1),(-1/2**0.5,1/2**0.5),(-1,0),(-1/2**0.5,-1/2**0.5),
            (0,-1),(1/2**0.5,-1/2**0.5),(1,0),(1/2**0.5,1/2**0.5)]

    def __init__(self, temp: float = 1.0):
        centers_np = [(self._SCALE * x / 2**0.5, self._SCALE * y / 2**0.5)
                      for x, y in self._RAW]
        self.centers = torch.tensor(centers_np, dtype=torch.float32)
        self.temp    = temp

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c       = self.centers.to(x.device)
        rewards = torch.arange(8, dtype=torch.float32).to(x.device) / 7.0
        diff    = x.unsqueeze(1) - c.unsqueeze(0)
        dist_sq = (diff ** 2).sum(-1)
        weights = torch.softmax(-dist_sq / self.temp, dim=-1)
        return (weights * rewards.unsqueeze(0)).sum(-1)

def get_energy_fn():
    return EightGaussiansEnergy(temp=1.0)

def _median_pairwise_dist(xy: torch.Tensor, max_samples: int = 2048) -> float:
    if xy.shape[0] > max_samples:
        xy = xy[torch.randperm(xy.shape[0], device=xy.device)[:max_samples]]
    return float(torch.pdist(xy.float()).median().clamp_min(1e-8).item())

@torch.no_grad()
def mmd_squared_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: Optional[float] = None,
) -> float:
    if sigma is None:
        sigma = _median_pairwise_dist(torch.cat([x, y], 0))
    s2 = 2.0 * sigma ** 2
    k_xx = torch.exp(-torch.cdist(x, x).pow(2) / s2)
    k_yy = torch.exp(-torch.cdist(y, y).pow(2) / s2)
    k_xy = torch.exp(-torch.cdist(x, y).pow(2) / s2)
    return float((k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()).item())

def kl_divergence_kde_2d(
    real: np.ndarray,
    gen: np.ndarray,
    grid_size: int = 80,
    eps: float = 1e-12,
) -> float:
    real = real.reshape(-1, 2)
    gen  = gen.reshape(-1, 2)
    lo = np.minimum(real.min(0), gen.min(0)) - 0.5
    hi = np.maximum(real.max(0), gen.max(0)) + 0.5
    gx, gy = np.linspace(lo[0], hi[0], grid_size), np.linspace(lo[1], hi[1], grid_size)
    xx, yy = np.meshgrid(gx, gy)
    grid   = np.stack([xx.ravel(), yy.ravel()])
    p = np.clip(gaussian_kde(real.T)(grid), eps, None)
    q = np.clip(gaussian_kde(gen.T)(grid),  eps, None)
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

@torch.no_grad()
def sample_from_target(
    pool: torch.Tensor,
    energy_fn: Callable,
    n: int,
    beta: float = 1.0,
) -> torch.Tensor:
    """Importance-resample n points from p(x)*r(x)^beta given a p(x) pool."""
    r = energy_fn(pool).float().clamp_min(0.0)
    w = r.pow(beta)
    total = w.sum()
    if total < 1e-12:
        idx = torch.randint(0, pool.shape[0], (n,), device=pool.device)
    else:
        idx = torch.multinomial(w / total, n, replacement=True)
    return pool[idx]

@torch.no_grad()
def evaluate_metrics_finetune(
    model: BaseGenerator,
    pool: torch.Tensor,
    energy_fn: Callable,
    device: torch.device,
    n: int = 4096,
    beta: float = 1.0,
    **sample_kwargs,
) -> Tuple[float, float, float]:
    """KL, MMD2, mean reward of model samples vs IS-resampled p(x)*r(x)^beta target."""
    model.eval()
    gt  = sample_from_target(pool, energy_fn, n, beta=beta)
    gen = model.sample(n, device, **sample_kwargs)
    kl   = kl_divergence_kde_2d(gt.cpu().numpy(), gen.cpu().numpy())
    mmd2 = mmd_squared_rbf(gt.to(device), gen)
    with torch.no_grad():
        reward_mean = float(energy_fn(gen).mean().item())
    model.train()
    return kl, mmd2, reward_mean

@torch.no_grad()
def evaluate_metrics(
    model: BaseGenerator,
    real_data: torch.Tensor,
    device: torch.device,
    num_samples: int = 4096,
) -> Tuple[float, float]:
    model.eval()
    idx  = torch.randint(0, real_data.shape[0], (num_samples,), device=device)
    real = real_data[idx]
    gen  = model.sample(num_samples, device)
    kl   = kl_divergence_kde_2d(real.cpu().numpy(), gen.cpu().numpy())
    mmd2 = mmd_squared_rbf(real, gen)
    model.train()
    return kl, mmd2

@torch.no_grad()
def evaluate_metrics_fixed(
    model: BaseGenerator,
    ref_real: torch.Tensor,
    device: torch.device,
    n_gen: Optional[int] = None,
    **sample_kwargs,
) -> Tuple[float, float]:
    model.eval()
    n   = n_gen if n_gen is not None else ref_real.shape[0]
    gen = model.sample(n, device, **sample_kwargs)
    kl   = kl_divergence_kde_2d(ref_real.cpu().numpy(), gen.cpu().numpy())
    mmd2 = mmd_squared_rbf(ref_real.to(device), gen)
    model.train()
    return kl, mmd2

def save_metrics_csv(
    save_path: str,
    steps: List[int],
    kl_vals: List[float],
    mmd_vals: List[float],
    reward_vals: Optional[List[float]] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        w = csv.writer(f)
        if reward_vals is not None:
            w.writerow(["step", "kl_kde", "mmd2_rbf", "reward_mean"])
            for s, k, m, r in zip(steps, kl_vals, mmd_vals, reward_vals):
                w.writerow([s, k, m, r])
        else:
            w.writerow(["step", "kl_kde", "mmd2_rbf"])
            for s, k, m in zip(steps, kl_vals, mmd_vals):
                w.writerow([s, k, m])
