"""IMM few-step generators (pushforward / Euler)."""
from __future__ import annotations

from typing import Optional

import torch


def pushforward_generator_fn(
    net,
    latents: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    *,
    discretization: Optional[str] = None,
    mid_nt: Optional[list] = None,
    num_steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
) -> torch.Tensor:
    """Push-forward solver for the IMM precond (a.k.a. CFG-warped Euler chain)."""
    if discretization == "uniform":
        t_steps = torch.linspace(net.T, net.eps, num_steps + 1, dtype=torch.float64, device=latents.device)
    elif discretization == "edm":
        nt_min = net.get_log_nt(torch.as_tensor(net.eps, dtype=torch.float64)).exp().item()
        nt_max = net.get_log_nt(torch.as_tensor(net.T, dtype=torch.float64)).exp().item()
        rho = 7
        idx = torch.arange(num_steps + 1, dtype=torch.float64, device=latents.device)
        nt_steps = (nt_max ** (1 / rho) + idx / num_steps * (nt_min ** (1 / rho) - nt_max ** (1 / rho))) ** rho
        t_steps = net.nt_to_t(nt_steps)
    else:
        if mid_nt is None:
            mid_nt = []
        mid_t = [net.nt_to_t(torch.as_tensor(nt)).item() for nt in mid_nt]
        t_steps = torch.tensor([net.T] + list(mid_t), dtype=torch.float64, device=latents.device)
        t_steps = torch.cat([t_steps, torch.ones_like(t_steps[:1]) * net.eps])

    x = latents.to(torch.float64)
    use_grad_ckpt = (len(t_steps) - 1) > 1

    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        if use_grad_ckpt:
            x = torch.utils.checkpoint.checkpoint(
                net.cfg_forward, x, t_cur, t_next,
                use_reentrant=False,
                class_labels=class_labels, cfg_scale=cfg_scale,
            ).to(torch.float64)
        else:
            x = net.cfg_forward(
                x, t_cur, t_next, class_labels=class_labels, cfg_scale=cfg_scale,
            ).to(torch.float64)
    return x


def generator_fn(*args, name: str = "pushforward_generator_fn", **kwargs):
    """Dispatch to a named generator function. Mirrors the original IMM API."""
    if name == "pushforward_generator_fn":
        return pushforward_generator_fn(*args, **kwargs)
    raise NotImplementedError(f"generator_fn: {name!r} not yet supported")
