"""``iMeanFlow`` — improved MeanFlow generator over the MiT backbone (inference-only)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt_fn

from . import model as mit_models


class iMeanFlow(nn.Module):
    """improved MeanFlow generator (inference-only by construction)."""

    def __init__(
        self,
        model_str: str,
        *,
        dtype: torch.dtype = torch.float32,
        img_size: int = 32,
        img_channels: int = 4,
        num_classes: int = 1000,
        eval_mode: bool = True,
    ) -> None:
        super().__init__()
        if not eval_mode:
            raise AssertionError("iMeanFlow only supports inference mode (no pretraining).")

        self.model_str = model_str
        self.dtype = dtype
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        net_fn = getattr(mit_models, model_str)
        self.net = net_fn(
            input_size=img_size,
            in_channels=img_channels,
            num_classes=num_classes,
            eval_mode=eval_mode,
        )

    def u_fn(self, x, t, h, omega, t_min, t_max, y):
        """Single-step velocity prediction. Returns ``(u, v)``."""
        bz = x.shape[0]
        return self.net(
            x,
            t.reshape(bz),
            h.reshape(bz),
            omega.reshape(bz),
            t_min.reshape(bz),
            t_max.reshape(bz),
            y,
        )

    def sample_one_step(self, z_t, labels, i, t_steps, omega, t_min, t_max):
        """Take one Euler step from time ``t_steps[i]`` to ``t_steps[i+1]``."""
        t = t_steps[i]
        r = t_steps[i + 1]
        bsz = z_t.shape[0]
        t = t.expand(bsz)
        r = r.expand(bsz)
        omega = omega.expand(bsz)
        t_min = t_min.expand(bsz)
        t_max = t_max.expand(bsz)

        u = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)[0]
        return z_t - (t - r)[:, None, None, None] * u

    def generate(
        self,
        n_sample: int,
        rng,
        num_steps: int,
        omega,
        t_min,
        t_max,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Few-step iMeanFlow sampling. Differentiable when ``torch.is_grad_enabled``."""
        x_shape = (n_sample, self.img_channels, self.img_size, self.img_size)
        z_t = rng.randn(x_shape).to(self.dtype)

        if labels is not None:
            y = labels.to(z_t.device)
        else:
            y = rng.randint(
                0, self.num_classes, size=(n_sample,), dtype=torch.int32
            ).to(z_t.device)

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1).to(self.dtype).to(z_t.device)

        def _to_tensor(v):
            return torch.tensor(v, dtype=self.dtype, device=z_t.device) \
                if not torch.is_tensor(v) else v

        omega = _to_tensor(omega)
        t_min = _to_tensor(t_min)
        t_max = _to_tensor(t_max)

        for i in range(num_steps):
            t = t_steps[i]
            r = t_steps[i + 1]
            bsz = z_t.shape[0]
            t_b = t.expand(bsz)
            r_b = r.expand(bsz)
            dt_b = t_b - r_b
            omega_b = omega.expand(bsz)
            t_min_b = t_min.expand(bsz)
            t_max_b = t_max.expand(bsz)

            def _step(z_t, t_b, dt_b, omega_b, t_min_b, t_max_b, y):
                u = self.u_fn(z_t, t_b, dt_b, omega_b, t_min_b, t_max_b, y=y)[0]
                return z_t - dt_b[:, None, None, None] * u

            if torch.is_grad_enabled() and num_steps > 1:
                z_t = ckpt_fn(
                    _step, z_t, t_b, dt_b, omega_b, t_min_b, t_max_b, y,
                    use_reentrant=False,
                )
            else:
                z_t = _step(z_t, t_b, dt_b, omega_b, t_min_b, t_max_b, y)

        return z_t
