"""DRaFT trainer: direct policy-gradient alignment. Requires a differentiable reward."""
from __future__ import annotations

from typing import Any, Optional

import torch

from ..backbones.base import ForwardMode, TorchBackbone
from ..losses.torch.draft import draft_loss
from ..rewards import Reward
from ..utils.distributed import BatchGenerator
from ..utils.vae import VAEWrapper
from . import _common
from .base import Trainer


class DRaFTTrainer(Trainer):
    """Direct policy-gradient alignment with reference-policy L2 anchor."""

    name = "draft"
    required_capabilities = ()

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
        self.ref_backbone = ref_backbone
        self.vae = vae
        self.clip_encoder = clip_encoder
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.global_step = 0
        self._last_metrics: dict = {}

        algo = cfg.algorithm
        self.reward_multiplier = float(algo.get("reward_multiplier", 1.0))
        self.grad_accum_steps = int(cfg.runtime.grad_accum_steps)
        self.micro_batch_size = int(cfg.runtime.micro_batch_size)
        self.max_grad_norm = float(algo.max_grad_norm)
        self.labels = torch.tensor(list(algo.labels), dtype=torch.int32)

        self.sample_cfg = dict(cfg.backbone.sample)

        if cfg.reward.name in ("jpeg_compress", "jpeg_incompress"):
            raise ValueError(
                f"DRaFT requires a differentiable reward; got {cfg.reward.name!r}. "
                "Use algorithm=fav with gradient_estimator.enabled=true for non-differentiable rewards."
            )

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
            f"[DRaFT] LoRA: trainable={n_train:,} / total={n_total:,} "
            f"({100 * n_train / n_total:.2f}%)"
        )
        self.backbone.forward_mode = ForwardMode.SAMPLE

        trainable = [p for p in self.backbone.parameters() if p.requires_grad]
        self.optimizer = _common.build_optimizer(
            trainable,
            type=algo.optimizer.type,
            lr=float(algo.optimizer.lr),
            betas=tuple(algo.optimizer.betas),
            weight_decay=float(algo.optimizer.weight_decay),
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

        # Per-rank class RNG (different seeds across ranks for a diverse class mix).
        self.class_rng = torch.Generator()
        self.class_rng.manual_seed(int(self.cfg.get("seed", 0)) + self.accelerator.process_index)

    def step(self) -> dict:
        assert self.optimizer is not None and self.scheduler is not None
        device = self.accelerator.device
        rank = self.accelerator.process_index
        n_proc = self.accelerator.num_processes
        micro_bsz = self.micro_batch_size
        n_classes = len(self.labels)

        seed_counter = _common.compute_seed_offsets(
            global_step=self.global_step,
            classes_per_step=1,
            grad_accum_steps=self.grad_accum_steps,
            n_proc=n_proc,
            micro_bsz=micro_bsz,
            pairs_per_micro=1,
        )

        last_loss = None
        last_grad_norm = None

        for _ in range(self.grad_accum_steps):
            with self.accelerator.accumulate(self.backbone):
                label_idx = torch.randint(0, n_classes, (micro_bsz,), generator=self.class_rng)
                labels = self.labels[label_idx]

                seed_off = seed_counter + rank * micro_bsz
                indices = torch.arange(seed_off, seed_off + micro_bsz)
                seed_counter += n_proc * micro_bsz

                sample_kwargs = {**self.sample_cfg, "labels": labels}

                x_gen = self.backbone(
                    n_sample=micro_bsz,
                    rng=BatchGenerator(device=device, seeds=indices),
                    **sample_kwargs,
                )

                # Latent backbones VAE-decode; pixel-space backbones use output as-is.
                images_gen = self.vae.decode(x_gen, enable_grad=True) if self.vae is not None else x_gen

                loss = draft_loss(
                    images_gen, self.reward,
                    reward_multiplier=self.reward_multiplier,
                )

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    gn = self.accelerator.clip_grad_norm_(
                        self.backbone.parameters(), max_norm=self.max_grad_norm,
                    )
                    last_grad_norm = gn.item() if torch.is_tensor(gn) else float(gn)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                last_loss = loss.detach().item()

        self.global_step += 1
        self._last_metrics = {
            "loss": last_loss,
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": last_grad_norm if last_grad_norm is not None else 0.0,
        }
        return self._last_metrics

    def eval(self) -> dict:
        return _common.evaluate_reward_per_label(
            backbone=self.backbone,
            accelerator=self.accelerator,
            labels=self.labels,
            reward_fn=lambda images: self.reward(images),
            vae=self.vae,
            sample_kwargs=dict(self.sample_cfg),
            eval_batch_size=int(self.cfg.runtime.eval_batch_size),
            global_seed=int(self.cfg.get("seed", 0)),
        )

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
