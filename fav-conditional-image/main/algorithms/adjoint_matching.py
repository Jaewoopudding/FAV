"""Adjoint Matching trainer: adjoint-ODE-based reward gradient. iMF-only; differentiable reward required."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from ..backbones.base import ForwardMode, TorchBackbone
from ..rewards import Reward
from ..utils.distributed import BatchGenerator
from ..utils.vae import VAEWrapper
from . import _common
from .base import Trainer


def _compute_sigma(t: torch.Tensor, h: float) -> torch.Tensor:
    """Memoryless noise schedule σ(t) = √(2(1 - t + h) / (t + h)); +h stabilises it."""
    return torch.sqrt(2 * (1 - t + h) / (t + h))


def _compute_kappa(t: torch.Tensor, h: float) -> torch.Tensor:
    """κ(t) = 1 / (t + h)."""
    return 1.0 / (t + h)


class AdjointMatchingTrainer(Trainer):
    """Adjoint-ODE-based alignment. iMF-only."""

    name = "adjoint_matching"
    required_capabilities = ("supports_velocity_mode",)

    def __init__(
        self,
        backbone: TorchBackbone,
        ref_backbone: TorchBackbone,
        reward: Reward,
        cfg: Any,
        accelerator: Any,
        *,
        vae: Optional[VAEWrapper] = None,
        clip_encoder: Optional[Any] = None,
    ) -> None:
        super().__init__(backbone, reward, cfg, accelerator)
        if vae is None:
            raise ValueError("Adjoint Matching requires a VAE wrapper (iMF latent-space backbone).")
        if cfg.reward.name in ("jpeg_compress", "jpeg_incompress"):
            raise ValueError(
                f"Adjoint Matching requires a differentiable reward; got {cfg.reward.name!r}. "
                "Black-box rewards are supported only by FAV (with NES) and Flow-GRPO."
            )
        self.ref_backbone = ref_backbone
        self.vae = vae
        self.clip_encoder = clip_encoder
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.global_step = 0
        self._last_metrics: dict = {}

        algo = cfg.algorithm
        self.num_sampling_steps = int(algo.num_adjoint_steps)
        self.h = 1.0 / self.num_sampling_steps
        self.lambda_am = float(algo.lambda_am)
        self.lct_constant = float(algo.get("lct_constant", 1.6))
        self.lct = self.lct_constant * (self.lambda_am ** 2)

        self.grad_accum_steps = int(cfg.runtime.grad_accum_steps)
        self.micro_batch_size = int(cfg.runtime.micro_batch_size)
        self.max_grad_norm = float(algo.max_grad_norm)
        self.labels = torch.tensor(list(algo.labels), dtype=torch.int32)

        bb_sample = cfg.backbone.sample
        self.cfg_omega = float(bb_sample.cfg_omega)
        self.interval_min = float(bb_sample.interval_min)
        self.interval_max = float(bb_sample.interval_max)
        self.img_size = int(cfg.backbone.img_size)
        self.img_channels = int(cfg.backbone.img_channels)
        self.dtype = getattr(torch, cfg.backbone.get("dtype", "float32"))

    @staticmethod
    def compute_accumulation_steps(cfg: Any) -> int:
        """One outer step accumulates across ``grad_accum_steps`` micro-batches."""
        return int(cfg.runtime.grad_accum_steps)

    def setup(self) -> None:
        algo = self.cfg.algorithm
        bb_lora = self.cfg.backbone.lora

        n_train, n_total = self.backbone.inject_lora(
            rank=int(bb_lora.rank),
            alpha=bb_lora.get("alpha", None),
            dropout=float(bb_lora.get("dropout", 0.0)),
            target_modules=bb_lora.get("target_modules", None),
        )
        self.accelerator.print(
            f"[Adjoint] LoRA: trainable={n_train:,} / total={n_total:,} "
            f"({100 * n_train / n_total:.2f}%)"
        )
        self.backbone.forward_mode = ForwardMode.VELOCITY
        self.ref_backbone.forward_mode = ForwardMode.VELOCITY

        trainable = [p for p in self.backbone.parameters() if p.requires_grad]
        self.optimizer = _common.build_optimizer(
            trainable,
            type=algo.optimizer.type,
            lr=float(algo.optimizer.lr),
            betas=tuple(algo.optimizer.get("betas", (0.9, 0.95))),
            weight_decay=float(algo.optimizer.get("weight_decay", 0.01)),
        )
        self.scheduler = _common.build_scheduler(
            self.optimizer,
            type=algo.scheduler.type,
            warmup_steps=int(algo.scheduler.get("warmup_steps", 0)),
            total_steps=int(algo.num_steps),
        )
        self.backbone, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.backbone, self.optimizer, self.scheduler,
        )

        # linspace with an ε at the end so the last sample-step lands on x_eps (~0)
        # and the adjoint solve has an extra slot for the t=0 endpoint.
        device = self.accelerator.device
        eps = 1e-5
        ts = torch.linspace(1.0, 0.0, self.num_sampling_steps + 1).to(self.dtype).to(device)
        ts[-1] = eps
        self.t_steps = torch.cat([ts, torch.tensor([0.0], device=device, dtype=self.dtype)])
        self.t_steps_reversed = torch.flip(self.t_steps[:-1], dims=[0])

        self.class_rng = torch.Generator()
        self.class_rng.manual_seed(int(self.cfg.get("seed", 0)) + self.accelerator.process_index)
        self.np_rng = np.random.default_rng(int(self.cfg.get("seed", 0)) + self.accelerator.process_index)

    def step(self) -> dict:
        device = self.accelerator.device
        rank = self.accelerator.process_index
        n_proc = self.accelerator.num_processes
        micro_bsz = self.micro_batch_size
        n_classes = len(self.labels)
        N = self.num_sampling_steps

        seed_counter = self.global_step * self.grad_accum_steps * n_proc * micro_bsz

        omega_b = torch.tensor(self.cfg_omega, dtype=self.dtype, device=device).expand(micro_bsz)
        t_min_b = torch.tensor(self.interval_min, dtype=self.dtype, device=device).expand(micro_bsz)
        t_max_b = torch.tensor(self.interval_max, dtype=self.dtype, device=device).expand(micro_bsz)

        accum_loss = 0.0
        accum_raw = 0.0
        accum_clip_ratio = 0.0
        accum_ctrl_norm = 0.0
        accum_lean_norm = 0.0
        last_grad_norm = 0.0

        n_25 = max(1, int(N * 0.25))
        n_75 = max(1, int(N * 0.75))

        for _ in range(self.grad_accum_steps):
            with self.accelerator.accumulate(self.backbone):
                label_idx = torch.randint(0, n_classes, (micro_bsz,), generator=self.class_rng)
                labels = self.labels[label_idx].to(device)

                seed_off = seed_counter + rank * micro_bsz
                seeds = torch.arange(seed_off, seed_off + micro_bsz)
                seed_counter += n_proc * micro_bsz

                rng = BatchGenerator(device=device, seeds=seeds)
                x_shape = (micro_bsz, self.img_channels, self.img_size, self.img_size)
                x_1 = rng.randn(x_shape).to(self.dtype)
                x_t = x_1.clone()
                x_traj = [x_1.clone()]

                # Phase 1: SDE rollout under current policy.
                with torch.no_grad():
                    for i in range(N):
                        t = self.t_steps[i]
                        r = self.t_steps[i + 1]
                        t_rev = self.t_steps_reversed[i]
                        t_b = t.expand(micro_bsz)
                        dt_b = t_b - r.expand(micro_bsz)

                        sigma = _compute_sigma(t_rev, self.h)
                        kappa = _compute_kappa(t_rev, self.h)

                        v_t = self.backbone(
                            x=x_t, t=t_b, h=dt_b,
                            omega=omega_b, t_min=t_min_b, t_max=t_max_b, y=labels,
                        )[0]
                        b_forward = -(2.0 * v_t + kappa * x_t)
                        if i < N - 1:
                            x_t = x_t + self.h * b_forward + (self.h ** 0.5) * sigma * torch.randn_like(x_t)
                        else:
                            x_t = x_t + self.h * b_forward
                        x_traj.append(x_t)

                # Phase 2: backward lean-adjoint solve using the frozen ref backbone.
                lean_adjoint: list[Optional[torch.Tensor]] = [None] * (N + 1)

                x_final = x_traj[-1].detach().clone().requires_grad_(True)
                images_final = self.vae.decode(x_final, enable_grad=True)
                reward_final = self.reward(images_final) * self.lambda_am
                g_final = -reward_final.sum()
                grad_final = torch.autograd.grad(g_final, x_final, create_graph=False)[0]
                lean_adjoint[-1] = grad_final.detach()

                for i in reversed(range(N)):
                    t = self.t_steps[i + 1]
                    r = self.t_steps[i + 2]
                    t_rev = self.t_steps_reversed[i + 1]
                    t_b = t.expand(micro_bsz)
                    dt_b = t_b - r.expand(micro_bsz)

                    kappa = _compute_kappa(t_rev, self.h)

                    x_t_solve = x_traj[i + 1].detach().clone().requires_grad_(True)
                    v_t_base = self.ref_backbone(
                        x=x_t_solve, t=t_b, h=dt_b,
                        omega=omega_b, t_min=t_min_b, t_max=t_max_b, y=labels,
                    )[0]
                    b_base = -(2.0 * v_t_base + kappa * x_t_solve)

                    Jv = (b_base * lean_adjoint[i + 1]).sum()
                    dJdx = torch.autograd.grad(Jv, x_t_solve, create_graph=False)[0]
                    lean_adjoint[i] = (lean_adjoint[i + 1] + self.h * dJdx).detach()

                # Phase 3: loss on selected timesteps (last 25% always + random 25% from first 75%).
                first_block = list(self.np_rng.choice(n_75, n_25, replace=False))
                selected_timesteps = first_block + list(range(n_75, N))

                micro_batch_loss = 0.0
                denom = N * self.grad_accum_steps

                for i in selected_timesteps:
                    t = self.t_steps[i]
                    r = self.t_steps[i + 1]
                    t_rev = self.t_steps_reversed[i]
                    t_b = t.expand(micro_bsz)
                    dt_b = t_b - r.expand(micro_bsz)
                    sigma = _compute_sigma(t_rev, self.h)

                    x_t_loss = x_traj[i].detach()
                    v_t = self.backbone(
                        x=x_t_loss, t=t_b, h=dt_b,
                        omega=omega_b, t_min=t_min_b, t_max=t_max_b, y=labels,
                    )[0]
                    with torch.no_grad():
                        v_t_base = self.ref_backbone(
                            x=x_t_loss, t=t_b, h=dt_b,
                            omega=omega_b, t_min=t_min_b, t_max=t_max_b, y=labels,
                        )[0]
                    lean_adj = lean_adjoint[i].detach()

                    diff = 2.0 / sigma * (v_t - v_t_base) - sigma * lean_adj

                    ctrl_term = 2.0 / sigma * (v_t - v_t_base)
                    lean_term = sigma * lean_adj
                    accum_ctrl_norm += ctrl_term.detach().pow(2).sum(dim=[1, 2, 3]).mean().item() / denom
                    accum_lean_norm += lean_term.detach().pow(2).sum(dim=[1, 2, 3]).mean().item() / denom

                    sample_loss_raw = (diff ** 2).sum(dim=[1, 2, 3])
                    sample_loss_clipped = torch.clamp(sample_loss_raw, max=self.lct)
                    step_loss = sample_loss_clipped.mean() / N

                    accum_raw += sample_loss_raw.detach().mean().item() / denom
                    accum_clip_ratio += (sample_loss_raw >= self.lct).float().mean().item() / denom
                    micro_batch_loss += step_loss.item()

                    self.accelerator.backward(step_loss)
                    del v_t, v_t_base, diff, step_loss

                accum_loss += micro_batch_loss / self.grad_accum_steps

                if self.accelerator.sync_gradients:
                    gn = self.accelerator.clip_grad_norm_(
                        self.backbone.parameters(), max_norm=self.max_grad_norm,
                    )
                    last_grad_norm = gn.item() if torch.is_tensor(gn) else float(gn)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        self.global_step += 1
        self._last_metrics = {
            "loss": accum_loss,
            "raw_loss": accum_raw,
            "clip_ratio": accum_clip_ratio,
            "ctrl_norm": accum_ctrl_norm,
            "lean_norm": accum_lean_norm,
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": last_grad_norm,
        }
        return self._last_metrics

    def eval(self) -> dict:
        # Eval uses deterministic ODE sampling; toggle SAMPLE mode on the unwrapped backbone.
        raw_backbone = self.accelerator.unwrap_model(self.backbone)
        prev_mode = raw_backbone.forward_mode
        raw_backbone.forward_mode = ForwardMode.SAMPLE
        try:
            metrics = _common.evaluate_reward_per_label(
                backbone=self.backbone,
                accelerator=self.accelerator,
                labels=self.labels,
                reward_fn=lambda images: self.reward(images),
                vae=self.vae,
                sample_kwargs=dict(
                    num_steps=self.num_sampling_steps,
                    cfg_omega=self.cfg_omega,
                    interval_min=self.interval_min,
                    interval_max=self.interval_max,
                ),
                eval_batch_size=int(self.cfg.runtime.eval_batch_size),
                global_seed=int(self.cfg.get("seed", 0)),
            )
        finally:
            raw_backbone.forward_mode = prev_mode
        return metrics

    def save(self, path: str) -> None:
        _common.save_trainer_ckpt(
            path,
            accelerator=self.accelerator,
            backbone=self.backbone,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            cfg=self.cfg,
        )

    def load(self, path: str) -> None:
        self.global_step = _common.load_trainer_ckpt(
            path,
            accelerator=self.accelerator,
            backbone=self.backbone,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
