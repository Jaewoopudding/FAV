"""FAV trainer: SVGD amortized-MLE alignment, sharded across GPUs + chunked per GPU."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import torch

from ..backbones.base import ForwardMode, TorchBackbone
from ..losses.torch.fav_accum import fav_loss_chunked, fav_total_score
from ..rewards import Reward
from ..utils.distributed import BatchGenerator
from ..utils.vae import VAEWrapper
from . import _common
from .base import Trainer


class FAVTrainer(Trainer):
    """SVGD amortized-MLE alignment, sharded across GPUs + chunked per GPU."""

    name = "fav"
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
        self.grad_estimator = None
        self.class_sampler: Optional[_common.RoundRobinClassSampler] = None
        self.global_step = 0
        self._last_metrics: dict = {}

        algo = cfg.algorithm
        self.beta = float(algo.beta)
        self.temp_kde = list(algo.temp_kde) if hasattr(algo.temp_kde, "__iter__") else [float(algo.temp_kde)]
        self.temp_stein = list(algo.temp_stein) if hasattr(algo.temp_stein, "__iter__") else [float(algo.temp_stein)]
        self.classes_per_step = int(algo.classes_per_step)
        self.micro_batch_size = int(cfg.runtime.micro_batch_size)
        self.max_grad_norm = float(algo.max_grad_norm)

        self.sample_cfg = dict(cfg.backbone.sample)

        self.use_jpeg_reward = cfg.reward.name in ("jpeg_compress", "jpeg_incompress")

        # effective_batch_size = 2 * classes_per_step * gen_per_class (gen + ref).
        self.effective_batch_size = int(algo.effective_batch_size)
        denom = 2 * self.classes_per_step
        if self.effective_batch_size % denom != 0:
            raise ValueError(
                f"effective_batch_size={self.effective_batch_size} must be divisible by "
                f"2 * classes_per_step = {denom} (got remainder "
                f"{self.effective_batch_size % denom})"
            )
        self.gen_per_class = self.effective_batch_size // denom
        if self.gen_per_class < 1:
            raise ValueError(
                f"effective_batch_size={self.effective_batch_size} too small for "
                f"classes_per_step={self.classes_per_step}: gen_per_class < 1"
            )
        if self.micro_batch_size < 1:
            raise ValueError(f"micro_batch_size must be >= 1, got {self.micro_batch_size}")

        self._reward_fn = self._jpeg_latent_reward if self.use_jpeg_reward else self.reward

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
            f"[FAV] LoRA: trainable={n_train:,} / total={n_total:,} "
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

        # Wraps backbone in DDP; ref is not prepared (no params trained).
        self.backbone, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.backbone, self.optimizer, self.scheduler,
        )

        self.grad_estimator = _common.build_gradient_estimator(
            algo.get("gradient_estimator", None)
        )
        if self.use_jpeg_reward and self.grad_estimator is None:
            raise ValueError(
                f"reward {self.cfg.reward.name!r} is non-differentiable; "
                "set algorithm.gradient_estimator.enabled=true"
            )

        labels = torch.tensor(list(algo.labels), dtype=torch.int32)
        self.class_sampler = _common.RoundRobinClassSampler(
            labels=labels,
            classes_per_step=self.classes_per_step,
            seed=int(self.cfg.get("seed", 0)),
        )

    def _jpeg_latent_reward(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Decode flattened latents and score via the JPEG reward (no grad)."""
        z = z_flat.reshape(-1, 4, 32, 32)
        with torch.no_grad():
            imgs = self.vae.decode(z, enable_grad=False)
        return self.reward(imgs).to(z_flat.device, dtype=z_flat.dtype)

    def _sample_latents(self, label: int, seeds: torch.Tensor, device, *, ref: bool):
        """Sample latents for the given per-particle seeds (caller sets grad context)."""
        n = int(seeds.numel())
        labels = torch.tensor([label] * n, dtype=torch.int32)
        rng = BatchGenerator(device=device, seeds=seeds)
        backbone = self.ref_backbone if ref else self.backbone
        return backbone(n_sample=n, rng=rng, **{**self.sample_cfg, "labels": labels})

    def _features(self, latents: torch.Tensor, *, grad: bool) -> torch.Tensor:
        """Map latents to SVGD particle features. Aesthetic: VAE-decode + CLIP-embed;
        JPEG: latents are the particles (reward decodes internally)."""
        if self.use_jpeg_reward:
            return latents.flatten(1)
        assert self.clip_encoder is not None
        if self.vae is not None:
            images = self.vae.decode(latents, enable_grad=grad)
        else:
            images = latents  # pixel-space backbone (StyleGAN-XL)
        return self.clip_encoder(images)

    def _build_set(self, label: int, gen_seeds: torch.Tensor, ref_seeds: torch.Tensor, device):
        """Generate + embed a detached particle set in micro chunks (a rank's shard,
        or the full K when G == 1)."""
        micro = self.micro_batch_size
        n = int(gen_seeds.numel())
        gen_feats, ref_feats = [], []
        with torch.no_grad():
            for s in range(0, n, micro):
                sl = slice(s, min(s + micro, n))
                gen_lat = self._sample_latents(label, gen_seeds[sl], device, ref=False)
                ref_lat = self._sample_latents(label, ref_seeds[sl], device, ref=True)
                gen_feats.append(self._features(gen_lat, grad=False))
                ref_feats.append(self._features(ref_lat, grad=False))
        return torch.cat(gen_feats).detach(), torch.cat(ref_feats).detach()

    def _maybe_no_sync(self, is_last: bool):
        """Defer DDP gradient sync until the final backward (multi-process only)."""
        if self.accelerator.num_processes > 1 and not is_last:
            return self.accelerator.no_sync(self.backbone)
        return nullcontext()

    def step(self) -> dict:
        """One optimizer update: sharded across GPUs + chunked per GPU."""
        assert self.optimizer is not None and self.scheduler is not None and self.class_sampler is not None
        acc = self.accelerator
        device = acc.device
        K = self.gen_per_class
        micro = self.micro_batch_size
        C = self.classes_per_step
        G = acc.num_processes
        rank = acc.process_index

        if K % G != 0:  # even sharding requires K % G == 0
            raise ValueError(
                f"gen_per_class={K} must be divisible by num_processes={G} for even "
                f"sharding (effective_batch_size / (2*classes_per_step) % n_gpu == 0)."
            )
        Kr = K // G                      # particles owned by this rank
        r0 = rank * Kr                   # this rank's offset into the full K

        step_labels = self.class_sampler.next_step_labels()
        base_seed = self.global_step * C * K * 2   # identical layout on every rank

        chunk_starts = list(range(0, Kr, micro))
        total_chunks = C * len(chunk_starts)

        self.optimizer.zero_grad(set_to_none=True)
        last_loss = None
        work_idx = 0

        for ci, label in enumerate(step_labels):
            label_i = int(label.item())
            class_base = base_seed + ci * K * 2
            ref_seeds_full = torch.arange(class_base, class_base + K)
            gen_seeds_full = torch.arange(class_base + K, class_base + 2 * K)
            gen_seeds_shard = gen_seeds_full[r0:r0 + Kr]
            ref_seeds_shard = ref_seeds_full[r0:r0 + Kr]

            # Phase 1: sharded build, gather assembles the full-K detached set on every rank.
            gen_local, ref_local = self._build_set(label_i, gen_seeds_shard, ref_seeds_shard, device)
            gen_global = acc.gather(gen_local).detach()    # (K, D), seed order
            ref_global = acc.gather(ref_local).detach()
            total_score_global = fav_total_score(
                gen_global, ref_global, self._reward_fn,
                beta=self.beta, temp_kde=self.temp_kde,
                gradient_estimator=self.grad_estimator,
            )

            # Phase 2: chunked grad pass over this rank's shard.
            for start in chunk_starts:
                work_idx += 1
                is_last = work_idx == total_chunks
                size = min(micro, Kr - start)
                with self._maybe_no_sync(is_last):
                    seeds = gen_seeds_shard[start:start + size]
                    gen_lat = self._sample_latents(label_i, seeds, device, ref=False)
                    gen_feat_chunk = self._features(gen_lat, grad=True)
                    loss = fav_loss_chunked(
                        gen_feat_chunk, gen_global, total_score_global,
                        temp_stein=self.temp_stein, n_global=K,
                    )
                    # scale = G*(size/K)/C: the G cancels DDP's ÷G averaging.
                    scale = G * (size / K) / C
                    acc.backward(loss * scale)
                    last_loss = loss.detach().item()

        gn = acc.clip_grad_norm_(self.backbone.parameters(), max_norm=self.max_grad_norm)
        last_grad_norm = gn.item() if torch.is_tensor(gn) else float(gn)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.global_step += 1
        self._last_metrics = {
            "loss": last_loss,
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": last_grad_norm,
        }
        return self._last_metrics

    def eval(self) -> dict:
        algo = self.cfg.algorithm
        labels = torch.tensor(list(algo.labels), dtype=torch.int32)

        def reward_fn(images: torch.Tensor) -> torch.Tensor:
            return self.reward(images)

        return _common.evaluate_reward_per_label(
            backbone=self.backbone,
            accelerator=self.accelerator,
            labels=labels,
            reward_fn=reward_fn,
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
