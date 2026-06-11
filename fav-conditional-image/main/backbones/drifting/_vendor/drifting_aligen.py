"""
ALIGEN training for the Drifting backbone (JAX).
Supports LoRA fine-tuning (default) or full fine-tuning.
"""
from __future__ import annotations

import argparse
import os
import time
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as mu
import numpy as np
import optax
from flax import serialization
from flax.training import train_state
from tqdm import tqdm
from PIL import Image

from aligen_loss import (
    jax_fav_total_score,
    jax_fav_loss_chunked,
    nes_gradient_estimator,
)
from aligen_lora import (
    create_lora_params,
    merge_lora,
    lora_summary,
    save_lora_checkpoint,
    attn_qkv_out_target,
)
from aligen_reward import (
    clip_vision_forward,
    mlp_diff_forward,
    load_clip_params_jax,
    load_mlp_params_jax,
    replicate_params,
)
from dataset.vae import vae_enc_decode
from utils.env import HF_ROOT
from utils.hsdp_util import (
    set_global_mesh,
    get_global_mesh,
    init_state_from_dummy_input,
    data_shard,
)
from utils.init_util import load_generator_model_and_params, maybe_init_state_params
from utils.logging import WandbLogger, log_for_0, is_rank_zero
from utils.misc import prepare_rng

class _DummyTrainState(train_state.TrainState):
    ema_params: Optional[Any] = None
    ema_decay: float = 0.999


def get_args_parser():
    p = argparse.ArgumentParser(description="ALIGEN training on Drifting backbone")

    # Model / checkpoint
    p.add_argument("--init-from", type=str, default="hf://latent_L_sota",
                   help="hf://<name> or local artifact path for the generator")
    p.add_argument("--workdir", type=str, default="runs/aligen")

    # Reward model
    p.add_argument("--clip-model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--mlp-weights", type=str, required=True,
                   help="Path to aesthetic MLP weights (.pth)")

    # Sampling
    p.add_argument("--cfg-scale", type=float, default=1.5,
                   help="CFG scale for sample generation")

    # ALIGEN training
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--labels", type=int, nargs="+",
                   default=[207, 22, 291, 387, 985, 483, 562, 649,
                            0, 9, 39, 55, 69, 80, 105, 108,
                            115, 130, 398, 403, 404, 409, 414, 497,
                            540, 547, 550, 561, 620, 650, 671, 732])
    p.add_argument("--batch-size-per-class", type=int, default=64,
                   help="Total samples per class across all GPUs")
    p.add_argument("--micro-batch-size", type=int, default=0,
                   help="Samples per GPU per micro-step. "
                        "0 = no micro-batching (whole per-GPU batch at once)")
    p.add_argument("--classes-per-step", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-grad-norm", type=float, default=2.0)
    p.add_argument("--weight-decay", type=float, default=0.01)

    # LoRA
    p.add_argument("--no-lora", action="store_true",
                   help="Disable LoRA and fine-tune all parameters")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=16.0)

    # ALIGEN loss
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--temp-kde", type=float, nargs="+", default=[0.5])
    p.add_argument("--temp-stein", type=float, nargs="+", default=[0.5])

    # NES (zeroth-order gradient estimation)
    p.add_argument("--use-gradient-estimate", action="store_true")
    p.add_argument("--nes-sigma", type=float, default=0.01)
    p.add_argument("--nes-n-samples", type=int, default=16)

    # Evaluation
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--global-seed", type=int, default=0)

    # Infrastructure
    p.add_argument("--hsdp-dim", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="drifting-aligen")
    p.add_argument("--wandb-entity", type=str, default=None)
    return p


def _embed(vae_decode_fn, clip_params, x):
    """Latents -> CLIP features. VAE returns NCHW; CLIP expects NHWC."""
    images = vae_decode_fn(x).transpose(0, 2, 3, 1)
    return clip_vision_forward(images, clip_params)


def _sample_latents(model_apply_fn, params, labels_batch, rng, cfg_scale):
    out = model_apply_fn(
        {"params": params}, c=labels_batch, cfg_scale=cfg_scale,
        train=False, rngs=prepare_rng(rng, ["noise"]),
    )
    return out["samples"]


# Two-phase FAV with full-K coupling: Phase 1 (build_step) embeds gen+ref
# particles per chunk (no grad) and the trainer computes total_score once over
# all K; Phase 2 (grad_step) regenerates the same chunk with grad and
# differentiates the chunked SVGD loss against the detached full-K set.


def _make_lora_build_step(model_apply_fn, vae_decode_fn, clip_params,
                          base_params, lora_scaling, cfg_scale,
                          out_sharding=None):
    """Phase-1: (gen_feat, ref_feat) for a chunk, detached. LoRA params."""

    def _step(lora_params, labels_batch, rng):
        rng_ref, rng_gen = jax.random.split(rng, 2)
        effective = merge_lora(base_params, lora_params, lora_scaling)
        x_gen = _sample_latents(model_apply_fn, effective, labels_batch, rng_gen, cfg_scale)
        x_ref = _sample_latents(model_apply_fn, base_params, labels_batch, rng_ref, cfg_scale)
        feat_gen = _embed(vae_decode_fn, clip_params, x_gen)
        feat_ref = _embed(vae_decode_fn, clip_params, x_ref)
        return jax.lax.stop_gradient(feat_gen), jax.lax.stop_gradient(feat_ref)

    jit_kwargs = {}
    if out_sharding is not None:
        jit_kwargs["out_shardings"] = out_sharding
    return jax.jit(_step, **jit_kwargs)


def _make_full_build_step(model_apply_fn, vae_decode_fn, clip_params,
                          cfg_scale, out_sharding=None):
    """Phase-1: (gen_feat, ref_feat) for a chunk, detached. Full params."""

    def _step(params, ref_params, labels_batch, rng):
        rng_ref, rng_gen = jax.random.split(rng, 2)
        x_gen = _sample_latents(model_apply_fn, params, labels_batch, rng_gen, cfg_scale)
        x_ref = _sample_latents(model_apply_fn, ref_params, labels_batch, rng_ref, cfg_scale)
        feat_gen = _embed(vae_decode_fn, clip_params, x_gen)
        feat_ref = _embed(vae_decode_fn, clip_params, x_ref)
        return jax.lax.stop_gradient(feat_gen), jax.lax.stop_gradient(feat_ref)

    jit_kwargs = {}
    if out_sharding is not None:
        jit_kwargs["out_shardings"] = out_sharding
    return jax.jit(_step, **jit_kwargs)


def _make_total_score_step(mlp_params, beta, temp_kde,
                           use_nes, nes_sigma, nes_n_samples,
                           out_sharding=None):
    """Phase 1: total SVGD score over the full detached K-particle set."""

    def _reward_fn(features, params):
        return mlp_diff_forward(features, params)

    def _step(gen_global, ref_global, rng_nes):
        grad_est = None
        if use_nes:
            grad_est = partial(nes_gradient_estimator,
                               sigma=nes_sigma, n_samples=nes_n_samples,
                               rng=rng_nes)
        return jax_fav_total_score(
            gen_global, ref_global, _reward_fn, mlp_params,
            beta, temp_kde, gradient_estimator=grad_est,
        )

    jit_kwargs = {}
    if out_sharding is not None:
        jit_kwargs["out_shardings"] = out_sharding
    return jax.jit(_step, **jit_kwargs)


def _make_lora_grad_step(model_apply_fn, vae_decode_fn, clip_params,
                         base_params, lora_scaling, cfg_scale,
                         temp_stein, n_global, out_sharding=None):
    """Phase-2: chunked grad w.r.t. *lora_params* against the detached full-K set."""

    def _step(lora_params, labels_batch, rng, gen_global, total_score_global):
        # Same split as build step -> identical rng_gen -> same gen particles.
        _rng_ref, rng_gen = jax.random.split(rng, 2)

        def loss_fn(lora_p):
            effective = merge_lora(base_params, lora_p, lora_scaling)
            x_gen = _sample_latents(model_apply_fn, effective, labels_batch, rng_gen, cfg_scale)
            feat_gen = _embed(vae_decode_fn, clip_params, x_gen)
            return jax_fav_loss_chunked(
                feat_gen, gen_global, total_score_global, temp_stein, n_global,
            )

        return jax.value_and_grad(loss_fn)(lora_params)

    jit_kwargs = {}
    if out_sharding is not None:
        jit_kwargs["out_shardings"] = out_sharding
    return jax.jit(_step, **jit_kwargs)


def _make_full_grad_step(model_apply_fn, vae_decode_fn, clip_params,
                         cfg_scale, temp_stein, n_global, out_sharding=None):
    """Phase-2: chunked grad w.r.t. *all* params against the detached full-K set."""

    def _step(params, labels_batch, rng, gen_global, total_score_global):
        _rng_ref, rng_gen = jax.random.split(rng, 2)

        def loss_fn(p):
            x_gen = _sample_latents(model_apply_fn, p, labels_batch, rng_gen, cfg_scale)
            feat_gen = _embed(vae_decode_fn, clip_params, x_gen)
            return jax_fav_loss_chunked(
                feat_gen, gen_global, total_score_global, temp_stein, n_global,
            )

        return jax.value_and_grad(loss_fn)(params)

    jit_kwargs = {}
    if out_sharding is not None:
        jit_kwargs["out_shardings"] = out_sharding
    return jax.jit(_step, **jit_kwargs)


def make_eval_fn(model_apply_fn, vae_decode_fn, clip_params, mlp_params, cfg_scale):
    @jax.jit
    def _eval_step(params, labels_batch, rng):
        out = model_apply_fn(
            {"params": params}, c=labels_batch, cfg_scale=cfg_scale,
            train=False, rngs=prepare_rng(rng, ["noise"]),
        )
        images = vae_decode_fn(out["samples"]).transpose(0, 2, 3, 1)
        features = clip_vision_forward(images, clip_params)
        scores = mlp_diff_forward(features, mlp_params)
        return images, scores
    return _eval_step


def main():
    args = get_args_parser().parse_args()
    use_lora = not args.no_lora

    labels = jnp.array(args.labels, dtype=jnp.int32)
    n_classes = len(args.labels)
    classes_per_step = args.classes_per_step
    assert n_classes % classes_per_step == 0

    hsdp = args.hsdp_dim or min(8, jax.local_device_count() * jax.process_count())
    set_global_mesh(hsdp)
    mesh = get_global_mesh()
    n_devices = jax.device_count()

    from jax.sharding import NamedSharding, PartitionSpec as P
    replicate_sharding = NamedSharding(mesh, P())

    assert args.batch_size_per_class % n_devices == 0, (
        f"batch_size_per_class ({args.batch_size_per_class}) must be "
        f"divisible by device_count ({n_devices})"
    )
    batch_size_per_gpu = args.batch_size_per_class // n_devices

    if args.micro_batch_size > 0:
        micro_bsz_per_gpu = args.micro_batch_size
        assert batch_size_per_gpu % micro_bsz_per_gpu == 0, (
            f"batch_size_per_gpu ({batch_size_per_gpu}) must be "
            f"divisible by micro_batch_size ({micro_bsz_per_gpu})"
        )
        grad_accum_steps = batch_size_per_gpu // micro_bsz_per_gpu
    else:
        micro_bsz_per_gpu = batch_size_per_gpu
        grad_accum_steps = 1

    micro_bsz_global = micro_bsz_per_gpu * n_devices

    assert args.eval_batch_size % n_devices == 0, (
        f"eval_batch_size ({args.eval_batch_size}) must be "
        f"divisible by device_count ({n_devices})"
    )

    timestamp = time.strftime("%m%d_%H%M%S")
    kde_str = "+".join(str(t) for t in args.temp_kde)
    stein_str = "+".join(str(t) for t in args.temp_stein)
    nes_tag = f"_nes_sig{args.nes_sigma}_n{args.nes_n_samples}" if args.use_gradient_estimate else ""
    lora_tag = f"_lora{args.lora_rank}" if use_lora else "_full"
    mbs_tag = f"_mbs{micro_bsz_per_gpu * n_devices}" if grad_accum_steps > 1 else ""
    run_name = (f"{timestamp}_ALIGEN_Drift_ms{args.num_steps}"
                f"_bs{args.batch_size_per_class * classes_per_step}{mbs_tag}"
                f"_lr{args.lr}_beta{args.beta}_kde{kde_str}_stein{stein_str}"
                f"_cfg{args.cfg_scale}{lora_tag}_seed{args.global_seed}{nes_tag}")
    workdir = os.path.join(args.workdir, run_name)
    log_for_0("Run name: %s", run_name)
    log_for_0("Workdir:  %s", workdir)

    log_for_0("Loading generator from %s ...", args.init_from)
    model, init_params, metadata = load_generator_model_and_params(
        args.init_from, hf_cache_dir=HF_ROOT,
    )

    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, args.lr, args.warmup_steps),
            optax.constant_schedule(args.lr),
        ],
        boundaries=[args.warmup_steps],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(lr_schedule, b1=0.9, b2=0.95, weight_decay=args.weight_decay),
    )

    rng = jax.random.PRNGKey(args.seed)
    dummy_state = init_state_from_dummy_input(
        model, optimizer, _DummyTrainState, rng,
        model.dummy_input(), model.rng_keys(), ema_decay=0.999,
    )
    dummy_state = maybe_init_state_params(
        dummy_state, model_type="generator", init_from=args.init_from,
        hf_cache_dir=HF_ROOT,
    )
    base_params = dummy_state.params          # frozen, properly sharded
    del dummy_state

    if use_lora:
        rng, lora_rng = jax.random.split(rng)
        lora_params = create_lora_params(
            base_params, rank=args.lora_rank, rng=lora_rng,
            target_fn=attn_qkv_out_target,
        )
        lora_params = replicate_params(lora_params, mesh)
        log_for_0(lora_summary(base_params, lora_params))

        opt_state = optimizer.init(lora_params)
        trainable_params = lora_params
    else:
        ref_params = jax.tree.map(lambda x: x, base_params)
        opt_state = optimizer.init(base_params)
        trainable_params = base_params

    state = train_state.TrainState(
        step=jnp.array(0, dtype=jnp.int32),
        apply_fn=model.apply,
        params=trainable_params,
        tx=optimizer,
        opt_state=opt_state,
    )

    log_for_0("Generator loaded. Devices=%d, LoRA=%s, "
              "micro_bsz_per_gpu=%d, grad_accum_steps=%d",
              n_devices, use_lora, micro_bsz_per_gpu, grad_accum_steps)

    clip_params = load_clip_params_jax(args.clip_model)
    clip_params = replicate_params(clip_params, mesh)
    mlp_params = load_mlp_params_jax(args.mlp_weights)
    mlp_params = replicate_params(mlp_params, mesh)

    _, vae_decode_fn = vae_enc_decode()
    log_for_0("VAE decoder loaded.")

    lora_scaling = args.lora_alpha / args.lora_rank

    # K generator particles per class: the SVGD kernel normaliser and size of
    # the full coupled set built in Phase 1.
    K = args.batch_size_per_class
    if use_lora:
        build_step_fn = _make_lora_build_step(
            model.apply, vae_decode_fn, clip_params,
            base_params, lora_scaling, cfg_scale=args.cfg_scale,
            out_sharding=replicate_sharding,
        )
        grad_step_fn = _make_lora_grad_step(
            model.apply, vae_decode_fn, clip_params,
            base_params, lora_scaling, cfg_scale=args.cfg_scale,
            temp_stein=args.temp_stein, n_global=K,
            out_sharding=replicate_sharding,
        )
    else:
        build_step_fn = _make_full_build_step(
            model.apply, vae_decode_fn, clip_params,
            cfg_scale=args.cfg_scale, out_sharding=replicate_sharding,
        )
        grad_step_fn = _make_full_grad_step(
            model.apply, vae_decode_fn, clip_params,
            cfg_scale=args.cfg_scale, temp_stein=args.temp_stein, n_global=K,
            out_sharding=replicate_sharding,
        )
    total_score_step_fn = _make_total_score_step(
        mlp_params, beta=args.beta, temp_kde=args.temp_kde,
        use_nes=args.use_gradient_estimate,
        nes_sigma=args.nes_sigma, nes_n_samples=args.nes_n_samples,
        out_sharding=replicate_sharding,
    )

    def _get_eval_params():
        """Return the effective full param tree for evaluation."""
        if use_lora:
            return merge_lora(base_params, state.params, lora_scaling)
        return state.params

    eval_step_fn = make_eval_fn(
        model.apply, vae_decode_fn, clip_params, mlp_params, args.cfg_scale,
    )

    @jax.jit
    def apply_grads(state, grads):
        return state.apply_gradients(grads=grads)

    logger = WandbLogger()
    if args.use_wandb:
        logger.set_logging(
            project=args.wandb_project, entity=args.wandb_entity,
            name=run_name, use_wandb=True, workdir=workdir, log_every_k=1,
        )

    class_rng = np.random.RandomState(args.global_seed)
    n_chunks = n_classes // classes_per_step

    # Training loop
    global_step = 0
    rng_train = jax.random.PRNGKey(args.seed + 1)
    pbar = tqdm(range(args.num_steps), desc="Training", disable=not is_rank_zero())

    for step in pbar:
        step_start = time.time()

        chunk_idx = step % n_chunks
        if chunk_idx == 0:
            perm = class_rng.permutation(n_classes)
            shuffled_labels = labels[perm]
        step_labels = shuffled_labels[chunk_idx * classes_per_step:
                                      (chunk_idx + 1) * classes_per_step]

        # Per class: build all K particles across grad_accum_steps chunks
        # (Phase 1), couple with one total_score over K, then differentiate
        # chunk-by-chunk (Phase 2). Summed per-chunk grads / n_accum recover
        # the monolithic per-class SVGD gradient on any device count.
        total_grads = jax.tree.map(jnp.zeros_like, state.params)
        total_loss = 0.0
        n_accum = classes_per_step * grad_accum_steps

        for cls_idx in range(classes_per_step):
            label_val = int(step_labels[cls_idx])
            micro_labels = jnp.full((micro_bsz_global,), label_val, dtype=jnp.int32)
            micro_labels = jax.device_put(micro_labels, data_shard())

            # Per-chunk RNGs reused across both phases so Phase 2 regenerates
            # the exact Phase 1 generator particles.
            chunk_rngs = []
            for _ in range(grad_accum_steps):
                rng_train, rng_chunk = jax.random.split(rng_train)
                chunk_rngs.append(jax.device_put(rng_chunk, replicate_sharding))
            rng_train, rng_nes = jax.random.split(rng_train)
            rng_nes = jax.device_put(rng_nes, replicate_sharding)

            # Phase 1: build the full K gen/ref feature set (no grad).
            gen_feats, ref_feats = [], []
            for rng_chunk in chunk_rngs:
                if use_lora:
                    gf, rf = build_step_fn(state.params, micro_labels, rng_chunk)
                else:
                    gf, rf = build_step_fn(state.params, base_params, micro_labels, rng_chunk)
                gen_feats.append(gf)
                ref_feats.append(rf)
            gen_global = jnp.concatenate(gen_feats, axis=0)     # (K, D)
            ref_global = jnp.concatenate(ref_feats, axis=0)     # (K, D)
            total_score = total_score_step_fn(gen_global, ref_global, rng_nes)

            # Phase 2: chunked grad pass against the full-K detached set.
            for rng_chunk in chunk_rngs:
                loss_i, grads_i = grad_step_fn(
                    state.params, micro_labels, rng_chunk, gen_global, total_score)
                total_grads = jax.tree.map(jnp.add, total_grads, grads_i)
                total_loss += float(loss_i)

        total_grads = jax.tree.map(lambda g: g / n_accum, total_grads)
        state = apply_grads(state, total_grads)
        global_step += 1
        avg_loss = total_loss / n_accum

        step_time = time.time() - step_start
        lr_val = float(lr_schedule(state.step - 1))
        grad_norm = float(optax.global_norm(total_grads))

        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_val:.2e}",
                         grad=f"{grad_norm:.2e}", t=f"{step_time:.1f}s")

        if is_rank_zero():
            logger.set_step(global_step)
            logger.log_dict({
                "train/loss": avg_loss, "train/lr": lr_val,
                "train/grad_norm": grad_norm, "train/step_time": step_time,
            })

        if global_step % args.eval_interval == 0 or global_step == args.num_steps:
            log_for_0("Evaluating at step %d ...", global_step)
            eval_dir = os.path.join(workdir, "imgs", str(global_step))
            os.makedirs(eval_dir, exist_ok=True)

            eval_params = _get_eval_params()
            all_rewards = []

            eval_micro_steps = max(1, args.eval_batch_size // micro_bsz_global)

            for label_val in args.labels:
                label_imgs = []
                label_scores = []

                for em_idx in range(eval_micro_steps):
                    micro_labels_eval = jnp.full(
                        (micro_bsz_global,), label_val, dtype=jnp.int32)
                    micro_labels_eval = jax.device_put(micro_labels_eval, data_shard())
                    eval_rng = jax.random.PRNGKey(
                        args.global_seed + label_val * 10000 + em_idx)
                    eval_rng = jax.device_put(eval_rng, replicate_sharding)

                    imgs_mb, scores_mb = eval_step_fn(
                        eval_params, micro_labels_eval, eval_rng)

                    label_scores.append(np.array(jax.device_get(scores_mb)))
                    if is_rank_zero():
                        label_imgs.append(np.array(jax.device_get(imgs_mb)))

                scores_np = np.concatenate(label_scores)
                all_rewards.append(scores_np)

                if is_rank_zero():
                    imgs_np = np.concatenate(label_imgs)
                    imgs_np = np.clip((imgs_np + 1) / 2, 0, 1)
                    imgs_np = np.round(imgs_np * 255).astype(np.uint8)
                    for i, img in enumerate(imgs_np):
                        dev_rank = (i % micro_bsz_global) // micro_bsz_per_gpu
                        r = round(float(scores_np[i]), 5)
                        fp = os.path.join(eval_dir, f"{dev_rank}_{label_val}_{i}_{r}.png")
                        Image.fromarray(img).save(fp)
                    logger.log_image(f"eval/class_{label_val}", imgs_np[:1])

            all_rewards = np.concatenate(all_rewards)
            mean_reward = float(all_rewards.mean())
            log_for_0("  Eval reward mean: %.4f", mean_reward)

            if is_rank_zero():
                logger.log_dict({
                    "eval/reward_mean": mean_reward,
                    "eval/reward_std": float(all_rewards.std()),
                })
            mu.sync_global_devices("eval done")

        if global_step % args.save_interval == 0 or global_step == args.num_steps:
            ckpt_dir = os.path.join(workdir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            if is_rank_zero():
                log_for_0("Saving checkpoint at step %d ...", global_step)
                if use_lora:
                    save_lora_checkpoint(
                        state.params,
                        os.path.join(ckpt_dir, f"lora_{global_step}"),
                        step=global_step,
                        lora_config={
                            "rank": args.lora_rank,
                            "alpha": args.lora_alpha,
                        },
                    )
                else:
                    cpu_params = jax.device_get(
                        mu.process_allgather(state.params))
                    p = os.path.join(ckpt_dir, f"params_{global_step}.msgpack")
                    with open(p, "wb") as f:
                        f.write(serialization.msgpack_serialize(cpu_params))
                log_for_0("  Saved to %s", ckpt_dir)
            mu.sync_global_devices("save done")

    logger.finish()
    log_for_0("Training complete.")


if __name__ == "__main__":
    main()
