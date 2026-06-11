"""DRaFTTrainer — direct reward backprop (pixel-space reward maximization).

Each optimizer step runs ``grad_accum`` micro-batches; each picks a random prompt,
samples latents, decodes, and applies ``draft_loss``.
"""
from __future__ import annotations

from typing import Any

import torch

from ..losses.torch.draft import draft_loss
from ..rewards import composed_reward_fn
from .base import Trainer


class DRaFTTrainer(Trainer):
    name = "draft"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        algo = self.cfg.algorithm
        self.reward_multiplier = float(algo.get("reward_multiplier", 1.0))
        self._composed_reward = composed_reward_fn(self.train_reward)
        self.prompt_rng = torch.Generator()
        self.prompt_rng.manual_seed(self.global_seed)

    def step(self) -> dict:
        acc = self.accelerator
        device = self.device
        bb = self.backbone
        micro = self.micro_batch_size
        n_proc = acc.num_processes
        rank = acc.process_index
        grad_accum = int(self.cfg.runtime.grad_accum_steps)

        seed_counter = self.global_step * grad_accum * n_proc * micro

        last_loss = None
        last_grad_norm = 0.0

        for _ in range(grad_accum):
            pidx = int(torch.randint(self.n_train, (1,), generator=self.prompt_rng).item())
            embs, masks = self.train_cache[pidx]
            self.train_reward.set_labels(
                torch.full((micro,), pidx, dtype=torch.long, device=device)
            )
            with acc.accumulate(bb.model):
                off = seed_counter + rank * micro
                seed_counter += n_proc * micro
                noise = bb.make_noise(range(off, off + micro), device)

                z_gen = bb.sample(noise, embs, masks, cfg_scale=self.cfg_scale,
                                  grad_checkpoint_sampling=self.grad_ckpt_sampling)
                img_gen = bb.decode_latents(z_gen, enable_grad=True)

                loss = draft_loss(img_gen, self._composed_reward, self.reward_multiplier)
                acc.backward(loss)

                if acc.sync_gradients:
                    gn = acc.clip_grad_norm_(bb.trainable_parameters(), self.max_grad_norm)
                    last_grad_norm = gn.item() if torch.is_tensor(gn) else float(gn)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                last_loss = loss.detach().item()

        self.global_step += 1
        self._last_metrics = {
            "loss": last_loss,
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": last_grad_norm,
        }
        return self._last_metrics
