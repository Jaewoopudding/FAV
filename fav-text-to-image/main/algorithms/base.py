"""Abstract trainer: shared setup / eval / checkpoint; subclasses implement step()."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from . import _common


class Trainer(ABC):
    name: str = "abstract"

    def __init__(self, *, backbone, train_reward, raw_reward, cfg, accelerator,
                 train_prompts: list[str], eval_prompts: list[str]) -> None:
        self.backbone = backbone
        self.train_reward = train_reward      # possibly NegatedReward
        self.raw_reward = raw_reward          # underlying scorer
        self.cfg = cfg
        self.accelerator = accelerator
        self.train_prompts = train_prompts
        self.eval_prompts = eval_prompts
        self.n_train = len(train_prompts)

        self.device = accelerator.device
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self._last_metrics: dict = {}

        algo = cfg.algorithm
        self.num_steps = int(algo.num_steps)
        self.lr = float(algo.lr)
        self.warmup_steps = int(algo.get("warmup_steps", 0))
        self.max_grad_norm = float(algo.max_grad_norm)
        self.micro_batch_size = int(cfg.runtime.micro_batch_size)
        self.eval_batch_size = int(cfg.runtime.eval_batch_size)
        self.cfg_scale = float(cfg.backbone.sample.cfg_scale)
        self.grad_ckpt_sampling = bool(cfg.backbone.sample.get("grad_checkpoint_sampling", False))
        self.global_seed = int(cfg.get("seed", 0))

    def setup(self) -> None:
        bb = self.backbone
        bb.load_models(self.device)

        self.train_cache = bb.encode_prompts(self.train_prompts, self.device)
        self.eval_cache = bb.encode_prompts(self.eval_prompts, self.device)
        bb.free_text_encoder()

        lora = self.cfg.backbone.lora
        n_train, n_total = bb.inject_lora(
            rank=int(lora.rank), alpha=float(lora.alpha),
            dropout=float(lora.get("dropout", 0.0)),
            target_modules=list(lora.target_modules),
        )
        self.accelerator.print(
            f"LoRA injected: trainable={n_train:,} / total={n_total:,} "
            f"({100 * n_train / n_total:.2f}%)"
        )

        self.optimizer = _common.build_optimizer(bb.trainable_parameters(), lr=self.lr)
        self.scheduler = _common.build_scheduler(self.optimizer, warmup_steps=self.warmup_steps)

        bb.prepare(self.accelerator)
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)

    @abstractmethod
    def step(self) -> dict: ...

    @torch.no_grad()
    def eval(self) -> dict:
        acc = self.accelerator
        device = self.device
        bb = self.backbone
        n_proc = acc.num_processes
        rank = acc.process_index
        micro = self.micro_batch_size
        eval_per_gpu = max(self.eval_batch_size // n_proc, 1)

        bb.model.eval()
        all_rewards: list[torch.Tensor] = []
        sample_images: dict[int, tuple] = {}

        for eidx, prompt in enumerate(self.eval_prompts):
            embs, masks = self.eval_cache[eidx]
            seed_base = self.global_seed + eidx * 10000 + rank * eval_per_gpu
            prompt_rewards = []
            for mb_start in range(0, eval_per_gpu, micro):
                mb_n = min(micro, eval_per_gpu - mb_start)
                self.raw_reward.set_labels(
                    torch.full((mb_n,), self.n_train + eidx, dtype=torch.long, device=device)
                )
                seeds = [seed_base + mb_start + i for i in range(mb_n)]
                noise = bb.make_noise(seeds, device)
                z = bb.sample(noise, embs, masks, cfg_scale=self.cfg_scale)
                img = bb.decode_latents(z, enable_grad=False)
                feat = self.raw_reward.encode_images(img.float())
                prompt_rewards.append(self.raw_reward(feat))
                if acc.is_main_process and mb_start == 0:
                    sample_images[eidx] = (img[0].detach().float().cpu(), prompt)
            all_rewards.append(acc.gather(torch.cat(prompt_rewards)))

        bb.model.train()
        rewards_cat = torch.cat(all_rewards)
        return {
            "reward_mean": rewards_cat.mean().item(),
            "reward_std": rewards_cat.std().item(),
            "reward_max": rewards_cat.max().item(),
            "reward_min": rewards_cat.min().item(),
            "sample_images": sample_images,
        }

    def save(self, path: str) -> None:
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "lora_model": self.backbone.get_lora_state(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "cfg": self.cfg,
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(str(path), map_location="cpu")
        self.backbone.load_lora_state(state["lora_model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state.get("global_step", 0))
