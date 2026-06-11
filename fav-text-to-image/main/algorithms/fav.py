"""FAVTrainer — feature-space amortized variational alignment (SVGD).

GPU-count-agnostic: shards the ``K`` per-prompt generator particles across ranks,
chunks each shard into ``micro_batch_size`` forwards, and gathers the full ``K``
detached set so SVGD coupling spans all ``K`` particles regardless of GPU count.
Per prompt: (1) sharded no-grad build of gen/ref features, gathered to full ``K``,
then ``total_score`` over ``K``; (2) chunked grad pass over this rank's shard.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from ..losses.torch.fav_accum import fav_loss_chunked, fav_total_score
from ..utils.prompts import RoundRobinPromptSampler
from .base import Trainer


class FAVTrainer(Trainer):
    name = "fav"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        algo = self.cfg.algorithm
        self.beta = float(algo.beta)
        self.temp_kde = list(algo.temp_kde) if hasattr(algo.temp_kde, "__iter__") else [float(algo.temp_kde)]
        self.temp_stein = list(algo.temp_stein) if hasattr(algo.temp_stein, "__iter__") else [float(algo.temp_stein)]
        self.prompts_per_step = min(int(algo.prompts_per_step), self.n_train)
        self.sampler = RoundRobinPromptSampler(
            self.n_train, self.prompts_per_step, seed=self.global_seed,
        )

        # effective_batch_size = 2 * prompts_per_step * gen_per_prompt
        # (counts both gen and ref particles over all prompts in a step).
        self.effective_batch_size = int(algo.effective_batch_size)
        denom = 2 * self.prompts_per_step
        if self.effective_batch_size % denom != 0:
            raise ValueError(
                f"effective_batch_size={self.effective_batch_size} must be divisible by "
                f"2 * prompts_per_step = {denom}"
            )
        self.gen_per_prompt = self.effective_batch_size // denom
        if self.gen_per_prompt < 1:
            raise ValueError("effective_batch_size too small for prompts_per_step")

    def _reward_features(self, latents, *, enable_grad: bool) -> torch.Tensor:
        img = self.backbone.decode_latents(latents, enable_grad=enable_grad)
        return self.raw_reward.encode_images(img.float())

    def _maybe_no_sync(self, is_last: bool):
        if self.accelerator.num_processes > 1 and not is_last:
            return self.accelerator.no_sync(self.backbone.model)
        return nullcontext()

    @torch.no_grad()
    def _build_set(self, embs, masks, gen_seeds, ref_seeds, device):
        """Detached gen/ref features for a shard, built in micro chunks."""
        bb = self.backbone
        micro = self.micro_batch_size
        n = int(gen_seeds.numel())
        gen_feats, ref_feats = [], []
        for s in range(0, n, micro):
            sl = slice(s, min(s + micro, n))
            gen_lat = bb.sample(bb.make_noise(gen_seeds[sl], device), embs, masks,
                                cfg_scale=self.cfg_scale)
            gen_feats.append(self._reward_features(gen_lat, enable_grad=False))
            with bb.reference_context():
                ref_lat = bb.sample(bb.make_noise(ref_seeds[sl], device), embs, masks,
                                    cfg_scale=self.cfg_scale)
            ref_feats.append(self._reward_features(ref_lat, enable_grad=False))
        return torch.cat(gen_feats).detach(), torch.cat(ref_feats).detach()

    def step(self) -> dict:
        acc = self.accelerator
        device = self.device
        bb = self.backbone
        K = self.gen_per_prompt
        micro = self.micro_batch_size
        C = self.prompts_per_step
        G = acc.num_processes
        rank = acc.process_index

        if K % G != 0:
            raise ValueError(
                f"gen_per_prompt={K} must be divisible by num_processes={G} for even "
                f"sharding (effective_batch_size / (2*prompts_per_step) % n_gpu == 0)."
            )
        Kr = K // G                      # particles owned by this rank
        r0 = rank * Kr                   # this rank's offset into the full K

        step_prompts = self.sampler.next_step()
        base_seed = self.global_step * C * K * 2

        chunk_starts = list(range(0, Kr, micro))
        total_chunks = len(step_prompts) * len(chunk_starts)

        self.optimizer.zero_grad(set_to_none=True)
        last_loss = None
        last_grad_norm = 0.0
        work = 0

        for ci, pidx in enumerate(step_prompts):
            embs, masks = self.train_cache[pidx]
            pbase = base_seed + ci * K * 2
            ref_seeds_full = torch.arange(pbase, pbase + K)
            gen_seeds_full = torch.arange(pbase + K, pbase + 2 * K)
            gen_seeds_shard = gen_seeds_full[r0:r0 + Kr]
            ref_seeds_shard = ref_seeds_full[r0:r0 + Kr]

            # Sharded build + gather → full K detached set.
            gen_local, ref_local = self._build_set(embs, masks, gen_seeds_shard, ref_seeds_shard, device)
            gen_global = acc.gather(gen_local).detach()
            ref_global = acc.gather(ref_local).detach()

            self.train_reward.set_labels(
                torch.full((K,), pidx, dtype=torch.long, device=device)
            )
            total_score = fav_total_score(
                gen_global, ref_global, self.train_reward,
                beta=self.beta, temp_kde=self.temp_kde,
            )

            # Chunked grad pass over this rank's shard.
            for start in chunk_starts:
                work += 1
                is_last = work == total_chunks
                size = min(micro, Kr - start)
                with self._maybe_no_sync(is_last):
                    seeds = gen_seeds_shard[start:start + size]
                    gen_lat = bb.sample(
                        bb.make_noise(seeds, device), embs, masks,
                        cfg_scale=self.cfg_scale,
                        grad_checkpoint_sampling=self.grad_ckpt_sampling,
                    )
                    feat_chunk = self._reward_features(gen_lat, enable_grad=True)
                    loss = fav_loss_chunked(
                        feat_chunk, gen_global, total_score,
                        temp_stein=self.temp_stein, n_global=K,
                    )
                    # G cancels DDP's ÷G averaging; size/K and /C normalize per
                    # chunk and per prompt → GPU-count-agnostic gradient.
                    scale = G * (size / K) / C
                    acc.backward(loss * scale)
                    last_loss = loss.detach().item()

        gn = acc.clip_grad_norm_(bb.trainable_parameters(), self.max_grad_norm)
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
