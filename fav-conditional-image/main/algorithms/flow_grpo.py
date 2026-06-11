"""Flow-GRPO trainer: group-relative PPO on stochastic flow-matching trajectories. iMF-only."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from ..backbones.base import ForwardMode, TorchBackbone
from ..losses.torch.flow_grpo import (
    PerPromptStatTracker,
    compute_log_prob_at_step,
    pipeline_with_logprob,
)
from ..rewards import Reward
from ..utils.distributed import BatchGenerator
from ..utils.vae import VAEWrapper
from . import _common
from .base import Trainer


class _LabelDataset(Dataset):
    def __init__(self, labels: torch.Tensor) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.labels[idx]


class _DistributedKRepeatSampler(Sampler):
    """Synchronized k-repeat sampler: each prompt gets exactly ``k`` samples scattered
    across the full distributed batch. All replicas share an RNG seed for a consistent split."""

    def __init__(
        self,
        dataset: _LabelDataset,
        *,
        samples_per_gpu: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        total = num_replicas * samples_per_gpu
        if total % k != 0:
            raise ValueError(
                f"num_replicas * samples_per_gpu = {total} not divisible by k={k}"
            )
        self.m = total // k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            n = len(self.dataset)
            if self.m <= n:
                indices = torch.randperm(n, generator=g)[: self.m].tolist()
            else:
                indices = torch.randint(0, n, (self.m,), generator=g).tolist()
            repeated = [i for i in indices for _ in range(self.k)]
            shuffle = torch.randperm(len(repeated), generator=g).tolist()
            shuffled = [repeated[i] for i in shuffle]
            chunk = shuffled[self.rank * self.samples_per_gpu : (self.rank + 1) * self.samples_per_gpu]
            yield chunk

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class FlowGRPOTrainer(Trainer):
    """SDE-rollout + PPO inner-loop alignment. iMF-only."""

    name = "flow_grpo"
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
            raise ValueError("Flow-GRPO requires a VAE wrapper (iMF latent-space backbone).")
        self.ref_backbone = ref_backbone
        self.vae = vae
        self.clip_encoder = clip_encoder
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.global_step = 0
        self.inner_step = 0
        self._last_metrics: dict = {}

        algo = cfg.algorithm
        self.num_sde_steps = int(algo.num_sde_steps)
        self.noise_level = float(algo.get("noise_level", 0.7))
        self.timestep_fraction = float(algo.get("timestep_fraction", 1.0))
        self.num_train_timesteps = max(1, int(self.num_sde_steps * self.timestep_fraction))

        self.clip_range = float(algo.clip_range)
        self.kl_coeff = float(algo.kl_coeff)
        self.adv_clip_max = float(algo.get("adv_clip_max", 5.0))
        self.adv_normalize_global_std = bool(algo.get("global_std", False))
        self.inner_epochs = int(algo.inner_epochs)
        self.k_repeat = int(algo.get("k_repeat", 4))

        self.micro_batch_size = int(cfg.runtime.micro_batch_size)
        self.total_batch_size = int(algo.get("total_batch_size", 256))   # rollout total across all GPUs
        self.train_micro_batch_size = int(algo.get("train_micro_batch_size", self.micro_batch_size))
        self.max_grad_norm = float(algo.max_grad_norm)
        self.labels = torch.tensor(list(algo.labels), dtype=torch.int32)

        bb_sample = cfg.backbone.sample
        self.cfg_omega = float(bb_sample.cfg_omega)
        self.interval_min = float(bb_sample.interval_min)
        self.interval_max = float(bb_sample.interval_max)
        self.img_size = int(cfg.backbone.img_size)
        self.img_channels = int(cfg.backbone.img_channels)
        self.dtype = getattr(torch, cfg.backbone.get("dtype", "float32"))

        self.use_jpeg_reward = cfg.reward.name in ("jpeg_compress", "jpeg_incompress")

        self.stat_tracker = PerPromptStatTracker(global_std=self.adv_normalize_global_std)

    @staticmethod
    def compute_accumulation_steps(cfg: Any) -> int:
        """Accelerator's ``gradient_accumulation_steps`` for Flow-GRPO."""
        algo = cfg.algorithm
        num_sde = int(algo.num_sde_steps)
        timestep_fraction = float(algo.get("timestep_fraction", 1.0))
        return max(1, int(num_sde * timestep_fraction))

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
            f"[Flow-GRPO] LoRA: trainable={n_train:,} / total={n_total:,} "
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

        n_proc = self.accelerator.num_processes
        if self.total_batch_size % n_proc != 0:
            raise ValueError(
                f"total_batch_size={self.total_batch_size} not divisible by world={n_proc}"
            )
        self.samples_per_gpu = self.total_batch_size // n_proc
        if self.samples_per_gpu % self.micro_batch_size != 0:
            raise ValueError(
                f"samples_per_gpu={self.samples_per_gpu} not divisible by "
                f"micro_batch_size={self.micro_batch_size}"
            )
        self.n_sample_iters = self.samples_per_gpu // self.micro_batch_size

        dataset = _LabelDataset(self.labels)
        sampler = _DistributedKRepeatSampler(
            dataset,
            samples_per_gpu=self.samples_per_gpu,
            k=self.k_repeat,
            num_replicas=n_proc,
            rank=self.accelerator.process_index,
            seed=int(self.cfg.get("seed", 0)),
        )
        self._train_sampler = sampler
        self._train_iter = iter(DataLoader(dataset, batch_sampler=sampler, num_workers=0))

        self.t_steps = (
            torch.linspace(1.0, 0.0, self.num_sde_steps + 1).to(self.dtype).to(self.accelerator.device)
        )

    def step(self) -> dict:
        device = self.accelerator.device
        rank = self.accelerator.process_index
        n_proc = self.accelerator.num_processes
        micro_bsz = self.micro_batch_size

        # Phase 1: Rollout
        raw = self.accelerator.unwrap_model(self.backbone)
        raw.eval()
        all_samples: list[dict] = []

        self._train_sampler.set_epoch(self.global_step)
        my_labels = next(self._train_iter).to(device)

        seed_counter = self.global_step * self.n_sample_iters * n_proc * micro_bsz

        with torch.no_grad():
            for i in range(self.n_sample_iters):
                mb_labels = my_labels[i * micro_bsz : (i + 1) * micro_bsz]

                seed_off = seed_counter + rank * micro_bsz
                seeds = torch.arange(seed_off, seed_off + micro_bsz)
                seed_counter += n_proc * micro_bsz

                images, latents_list, logp_list = pipeline_with_logprob(
                    backbone=raw,
                    n_sample=micro_bsz,
                    num_steps=self.num_sde_steps,
                    omega=self.cfg_omega,
                    t_min=self.interval_min,
                    t_max=self.interval_max,
                    labels=mb_labels,
                    rng=BatchGenerator(device=device, seeds=seeds),
                    img_channels=self.img_channels,
                    img_size=self.img_size,
                    vae=self.vae,
                    noise_level=self.noise_level,
                    dtype=self.dtype,
                )

                latents_stacked = torch.stack(latents_list, dim=1)
                logp_stacked = torch.stack(logp_list, dim=1)
                rewards = self.reward(images)
                if not torch.is_tensor(rewards):
                    rewards = torch.as_tensor(rewards, device=device, dtype=torch.float32)

                all_samples.append(dict(
                    latents=latents_stacked[:, :-1].detach(),
                    next_latents=latents_stacked[:, 1:].detach(),
                    log_probs=logp_stacked.detach(),
                    rewards=rewards.detach(),
                    labels=mb_labels,
                ))

        raw.train()

        # Phase 2: Advantage normalization
        local_rewards = torch.cat([s["rewards"] for s in all_samples])
        local_labels = torch.cat([s["labels"] for s in all_samples])
        gathered_rewards = self.accelerator.gather(local_rewards)
        gathered_labels = self.accelerator.gather(local_labels)

        all_advantages = self.stat_tracker.update(
            gathered_labels.cpu().numpy(),
            gathered_rewards.cpu().numpy(),
        )
        N_local = local_rewards.shape[0]
        my_adv = all_advantages[rank * N_local : (rank + 1) * N_local]

        offset = 0
        for s in all_samples:
            B = s["rewards"].shape[0]
            adv = torch.tensor(my_adv[offset : offset + B], device=device, dtype=torch.float32)
            s["advantages"] = adv.unsqueeze(1).repeat(1, self.num_train_timesteps)
            offset += B

        rollout_reward_mean = local_rewards.mean().item()
        rollout_reward_std = local_rewards.std().item()
        self.stat_tracker.clear()

        for s in all_samples:
            del s["rewards"]

        samples_dict = {
            k: torch.cat([s[k] for s in all_samples]) for k in all_samples[0].keys()
        }

        # Phase 3: PPO inner-loop
        info: dict[str, list[float]] = defaultdict(list)
        last_grad_norm = 0.0

        for inner_epoch in range(self.inner_epochs):
            perm = torch.randperm(N_local, device=device)
            shuffled = {k: v[perm] for k, v in samples_dict.items()}
            num_micro = N_local // self.train_micro_batch_size
            batched = {
                k: v[: num_micro * self.train_micro_batch_size].reshape(
                    num_micro, self.train_micro_batch_size, *v.shape[1:]
                )
                for k, v in shuffled.items()
            }
            micro_batches = [
                dict(zip(batched, x)) for x in zip(*batched.values())
            ]

            for batch in micro_batches:
                for j in range(self.num_train_timesteps):
                    with self.accelerator.accumulate(self.backbone):
                        prev, log_prob, prev_mean, std_dev_t = compute_log_prob_at_step(
                            backbone=self.backbone,
                            sample_batch=batch,
                            j=j,
                            t_steps=self.t_steps,
                            cfg_omega=self.cfg_omega,
                            interval_min=self.interval_min,
                            interval_max=self.interval_max,
                            noise_level=self.noise_level,
                            dtype=self.dtype,
                        )

                        if self.kl_coeff > 0:
                            with torch.no_grad():
                                _, _, prev_mean_ref, _ = compute_log_prob_at_step(
                                    backbone=self.ref_backbone,
                                    sample_batch=batch,
                                    j=j,
                                    t_steps=self.t_steps,
                                    cfg_omega=self.cfg_omega,
                                    interval_min=self.interval_min,
                                    interval_max=self.interval_max,
                                    noise_level=self.noise_level,
                                    dtype=self.dtype,
                                )

                        adv = torch.clamp(
                            batch["advantages"][:, j],
                            -self.adv_clip_max, self.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - batch["log_probs"][:, j])
                        unclip = -adv * ratio
                        clip = -adv * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        policy_loss = torch.maximum(unclip, clip).mean()

                        if self.kl_coeff > 0:
                            std_e = std_dev_t.view(-1, *([1] * (prev_mean.ndim - 1)))
                            kl = ((prev_mean - prev_mean_ref) ** 2).mean(
                                dim=tuple(range(1, prev_mean.ndim)), keepdim=True,
                            ) / (2 * std_e ** 2)
                            kl_loss = kl.mean()
                            loss = policy_loss + self.kl_coeff * kl_loss
                            info["kl_loss"].append(kl_loss.item())
                        else:
                            loss = policy_loss

                        info["policy_loss"].append(policy_loss.item())
                        info["loss"].append(loss.item())
                        info["approx_kl"].append(
                            0.5 * ((log_prob - batch["log_probs"][:, j]) ** 2).mean().item()
                        )
                        info["clipfrac"].append(
                            ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                        )

                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            gn = self.accelerator.clip_grad_norm_(
                                self.backbone.parameters(), max_norm=self.max_grad_norm,
                            )
                            last_grad_norm = gn.item() if torch.is_tensor(gn) else float(gn)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.scheduler.step()
                    self.inner_step += 1

        self.global_step += 1

        self._last_metrics = {
            "loss": float(np.mean(info["loss"])),
            "policy_loss": float(np.mean(info["policy_loss"])),
            "approx_kl": float(np.mean(info["approx_kl"])),
            "clipfrac": float(np.mean(info["clipfrac"])),
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": last_grad_norm,
            "rollout_reward_mean": rollout_reward_mean,
            "rollout_reward_std": rollout_reward_std,
            "inner_step": self.inner_step,
        }
        if info.get("kl_loss"):
            self._last_metrics["kl_loss"] = float(np.mean(info["kl_loss"]))
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
                    num_steps=self.num_sde_steps,
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
