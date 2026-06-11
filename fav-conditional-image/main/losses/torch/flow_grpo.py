"""SDE solver + log-prob + GRPO advantage utilities for Flow-GRPO.

A stochastic Euler-Maruyama solver over a single-step velocity predictor that
exposes per-step Gaussian transition log-probs, so PPO's importance ratio can be
computed during the inner loop.
"""
from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor


def sde_step_with_logprob(
    model_output: torch.Tensor,
    t: Union[float, torch.FloatTensor],
    dt_b: Union[float, torch.FloatTensor],
    sample: torch.Tensor,
    *,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    t_max: float = 0.99,
):
    """One Euler-Maruyama step + log-prob under the resulting Gaussian kernel.
    Returns ``(prev_sample, log_prob, prev_sample_mean, std_dev_t)``."""
    # bf16 can overflow in the mean computation; promote to fp32 internally.
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    ndim = sample.ndim
    expand = lambda x: x.view(-1, *([1] * (ndim - 1))) if x.ndim == 1 else x

    t_e = expand(t)
    dt_b_signed = -1 * dt_b           # r - t < 0
    dt_e = expand(dt_b_signed)

    t_clamped = torch.where(torch.isclose(t_e, torch.ones_like(t_e)), t_max, t_e)
    std_dev_t_e = torch.sqrt(t_e / (1 - t_clamped)) * noise_level
    std_dev_t = std_dev_t_e.squeeze()

    prev_sample_mean = (
        sample * (1 + std_dev_t_e ** 2 / (2 * t_e) * dt_e)
        + model_output * (1 + std_dev_t_e ** 2 * (1 - t_e) / (2 * t_e)) * dt_e
    )

    sqrt_neg_dt = torch.sqrt(-1 * dt_e)

    if prev_sample is None:
        noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t_e * sqrt_neg_dt * noise

    std_total = std_dev_t_e * sqrt_neg_dt
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_total ** 2)
        - torch.log(std_total)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, prev_sample_mean, std_dev_t


@torch.no_grad()
def pipeline_with_logprob(
    *,
    backbone,                    # TorchBackbone in VELOCITY mode, unwrapped
    n_sample: int,
    num_steps: int,
    omega,
    t_min,
    t_max,
    labels: torch.Tensor,
    rng,                         # BatchGenerator
    img_channels: int,
    img_size: int,
    vae,                         # VAEWrapper (for final decode)
    noise_level: float = 0.7,
    dtype: torch.dtype = torch.float32,
):
    """SDE rollout returning final images, all latents, and per-step log-probs."""
    device = next(backbone.parameters()).device
    x_shape = (n_sample, img_channels, img_size, img_size)
    latents = rng.randn(x_shape).to(dtype)

    y = labels.to(latents.device)

    t_steps = torch.linspace(1.0, 0.0, num_steps + 1).to(dtype).to(latents.device)
    omega = omega if torch.is_tensor(omega) else torch.tensor(omega, dtype=dtype, device=latents.device)
    t_min = t_min if torch.is_tensor(t_min) else torch.tensor(t_min, dtype=dtype, device=latents.device)
    t_max = t_max if torch.is_tensor(t_max) else torch.tensor(t_max, dtype=dtype, device=latents.device)

    all_latents = [latents]
    all_log_probs: list[torch.Tensor] = []

    for i in range(num_steps):
        t = t_steps[i]
        r = t_steps[i + 1]
        bsz = latents.shape[0]
        t_b = t.expand(bsz)
        dt_b = (t_b - r.expand(bsz))
        omega_b = omega.expand(bsz)
        t_min_b = t_min.expand(bsz)
        t_max_b = t_max.expand(bsz)

        v_t = backbone(
            x=latents, t=t_b, h=dt_b,
            omega=omega_b, t_min=t_min_b, t_max=t_max_b, y=y,
        )[0]

        latents, log_prob, _mean, _std = sde_step_with_logprob(
            v_t, t_b, dt_b, latents, noise_level=noise_level,
        )
        all_latents.append(latents)
        all_log_probs.append(log_prob)

    images = vae.decode(latents, enable_grad=True)
    return images, all_latents, all_log_probs


def compute_log_prob_at_step(
    *,
    backbone,                     # DDP-wrapped, VELOCITY mode
    sample_batch: dict,
    j: int,
    t_steps: torch.Tensor,
    cfg_omega: float,
    interval_min: float,
    interval_max: float,
    noise_level: float,
    dtype: torch.dtype = torch.float32,
):
    """Recompute log-prob of the stored ``next_latents[:, j]`` under ``backbone`` (PPO inner-loop)."""
    device = sample_batch["latents"].device
    bsz = sample_batch["latents"].shape[0]

    t = t_steps[j].expand(bsz)
    h = (t_steps[j] - t_steps[j + 1]).expand(bsz)
    omega_b = torch.tensor(cfg_omega, dtype=dtype, device=device).expand(bsz)
    t_min_b = torch.tensor(interval_min, dtype=dtype, device=device).expand(bsz)
    t_max_b = torch.tensor(interval_max, dtype=dtype, device=device).expand(bsz)

    v_t = backbone(
        x=sample_batch["latents"][:, j],
        t=t,
        h=h,
        omega=omega_b,
        t_min=t_min_b,
        t_max=t_max_b,
        y=sample_batch["labels"],
    )[0]

    return sde_step_with_logprob(
        v_t, t, h, sample_batch["latents"][:, j],
        noise_level=noise_level,
        prev_sample=sample_batch["next_latents"][:, j],
    )


class PerPromptStatTracker:
    """Per-class reward stats for GRPO advantage normalization ``(reward - mean) / std``.
    ``global_std=True`` uses the across-batch std; ``clear()`` resets between rollouts."""

    def __init__(self, *, global_std: bool = False) -> None:
        self.global_std = global_std
        self.stats: dict[int, list[float]] = {}
        self.history_prompts: set[int] = set()

    def update(self, prompts: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        prompts = np.asarray(prompts)
        rewards = np.asarray(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.zeros_like(rewards)

        for p in unique:
            r_p = rewards[prompts == p]
            self.stats.setdefault(int(p), []).extend(r_p.tolist())
            self.history_prompts.add(int(p))

        for p in unique:
            stats_p = np.asarray(self.stats[int(p)])
            r_p = rewards[prompts == p]
            mean = stats_p.mean(keepdims=True)
            if self.global_std:
                std = rewards.std(keepdims=True) + 1e-4
            else:
                std = stats_p.std(keepdims=True) + 1e-4
            advantages[prompts == p] = (r_p - mean) / std

        return advantages

    def clear(self) -> None:
        self.stats = {}
