"""Unified Hydra-driven training entrypoint.

Example:
    python -m scripts.train backbone=imf algorithm=fav reward=aesthetic
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from main.algorithms import (
    AdjointMatchingTrainer,
    DRaFTTrainer,
    FAVTrainer,
    FlowGRPOTrainer,
)
from main.algorithms import _common
from main.backbones.imf import IMFBackbone
from main.utils.distributed import seed_everything
from main.utils.logging import init_wandb, setup_logging


def build_backbone(cfg_backbone: DictConfig):
    """Construct a backbone (no LoRA, no checkpoint) from cfg.backbone."""
    name = cfg_backbone.name
    if name == "imf":
        return IMFBackbone(
            model_str=cfg_backbone.model_str,
            dtype=getattr(torch, cfg_backbone.get("dtype", "float32")),
            img_size=int(cfg_backbone.img_size),
            img_channels=int(cfg_backbone.img_channels),
            num_classes=int(cfg_backbone.num_classes),
        )
    if name == "imm":
        from main.backbones.imm import IMMBackbone
        return IMMBackbone(
            config_name=cfg_backbone.get("config_name", "im256_generate_images.yaml"),
            temb_type=cfg_backbone.get("temb_type", "identity"),
            img_resolution=int(cfg_backbone.get("img_resolution", 32)),
            img_channels=int(cfg_backbone.img_channels),
            label_dim=int(cfg_backbone.get("label_dim", 1000)),
            dtype=getattr(torch, cfg_backbone.get("dtype", "float32")),
        )
    if name == "stylegan_xl":
        from main.backbones.stylegan_xl import StyleGANXLBackbone
        return StyleGANXLBackbone(
            img_resolution=int(cfg_backbone.get("img_resolution", 256)),
            label_dim=int(cfg_backbone.get("label_dim", 1000)),
            dtype=getattr(torch, cfg_backbone.get("dtype", "float32")),
        )
    if name == "drifting":
        raise RuntimeError(
            "Drifting backbone is JAX-native — run via:\n"
            "    python -m scripts.train_drifting backbone=drifting algorithm=<fav|draft> ...\n"
            "(Requires the [jax] extra: `pip install -e '.[jax]'`.)"
        )
    raise NotImplementedError(f"Backbone {name!r} is not yet ported.")


def build_trainer(cfg: DictConfig, backbone, ref_backbone, reward, vae, clip_encoder, accelerator):
    """Construct the algorithm-specific trainer."""
    name = cfg.algorithm.name
    common_kwargs = dict(
        backbone=backbone,
        ref_backbone=ref_backbone,
        reward=reward,
        cfg=cfg,
        accelerator=accelerator,
        vae=vae,
        clip_encoder=clip_encoder,
    )
    if name == "fav":
        return FAVTrainer(**common_kwargs)
    if name == "draft":
        return DRaFTTrainer(**common_kwargs)
    if name == "flow_grpo":
        return FlowGRPOTrainer(**common_kwargs)
    if name == "adjoint_matching":
        return AdjointMatchingTrainer(**common_kwargs)
    raise NotImplementedError(f"Unknown trainer {name!r}")


def compose_run_name(cfg: DictConfig) -> str:
    if cfg.run_name is not None:
        return str(cfg.run_name)
    ts = time.strftime("%m%d_%H%M%S")
    return (
        f"{ts}_{cfg.backbone.name}_{cfg.algorithm.name}_{cfg.reward.name}"
        f"_lora{cfg.backbone.lora.rank}_seed{cfg.seed}"
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    micro_bsz = int(cfg.runtime.micro_batch_size)
    algo_name = cfg.algorithm.name
    # classes_per_step is FAV-specific; default to 1 for the other algorithms.
    classes_per_step = int(cfg.algorithm.get("classes_per_step", 1))

    # Auto-compute grad accumulation from the global batch so the effective
    # batch is GPU-count independent. FAV self-normalises (total_accum=1) and
    # Flow-GRPO accumulates over SDE timesteps, so both keep their own scheme.
    n_gpu = int(os.environ.get("WORLD_SIZE") or torch.cuda.device_count() or 1)
    if algo_name in ("draft", "adjoint_matching"):
        target_batch = int(cfg.algorithm.total_batch_size)
        grad_accum = max(1, target_batch // n_gpu // micro_bsz)
    else:
        grad_accum = int(cfg.runtime.get("grad_accum_steps", 1))
    cfg.runtime.grad_accum_steps = grad_accum

    if algo_name == "fav":
        total_accum = 1
    elif algo_name == "flow_grpo":
        total_accum = FlowGRPOTrainer.compute_accumulation_steps(cfg)
    else:  # draft, adjoint_matching
        total_accum = grad_accum

    accelerator = _common.build_accelerator(
        gradient_accumulation_steps=total_accum,
        mixed_precision=cfg.runtime.get("mixed_precision", "fp32"),
    )

    run_name = compose_run_name(cfg)
    workdir = Path(cfg.workdir) / run_name
    workdir.mkdir(parents=True, exist_ok=True) if accelerator.is_main_process else None
    accelerator.wait_for_everyone()

    logger = setup_logging(workdir if accelerator.is_main_process else None)
    accelerator.print(f"[main] run={run_name}")
    accelerator.print(f"[main] workdir={workdir}")
    accelerator.print(f"[main] world_size={accelerator.num_processes}")
    if algo_name == "fav":
        eff = int(cfg.algorithm.effective_batch_size)
        gen_per_class = eff // (2 * classes_per_step)
        accelerator.print(
            f"[main] effective_batch_size={eff}, classes_per_step={classes_per_step}, "
            f"gen_per_class={gen_per_class}, micro_bsz={micro_bsz}, "
            f"chunks_per_rank={(gen_per_class // accelerator.num_processes + micro_bsz - 1) // micro_bsz}"
        )
    else:
        accelerator.print(
            f"[main] micro_bsz={micro_bsz}, grad_accum={grad_accum}, "
            f"classes_per_step={classes_per_step}, total_accum={total_accum}"
        )

    if accelerator.is_main_process:
        wandb_run = init_wandb(
            project=str(cfg.wandb.project),
            run_name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            workdir=workdir,
            enabled=bool(cfg.wandb.enabled),
        )
    else:
        wandb_run = None

    seed_everything(int(cfg.seed))
    device = accelerator.device

    # Clone the frozen reference backbone before LoRA injection.
    backbone = build_backbone(cfg.backbone)
    backbone.load_pretrained(cfg.backbone.ckpt_path)
    backbone.to(device)

    ref_backbone = _common.clone_for_reference(backbone, device=device)

    # VAE is skipped for pixel-space backbones.
    reward, clip_encoder = _common.build_reward(
        cfg.reward,
        device=device,
        dtype=getattr(torch, cfg.backbone.get("dtype", "float32")),
    )
    needs_vae = cfg.backbone.name in ("imf", "imm")
    vae = (
        _common.build_vae(
            cfg.backbone.name,
            decode_batch_size=int(cfg.runtime.gen_bsz),
            dtype=getattr(torch, cfg.backbone.get("dtype", "float32")),
        )
        if needs_vae else None
    )

    trainer = build_trainer(
        cfg=cfg,
        backbone=backbone,
        ref_backbone=ref_backbone,
        reward=reward,
        vae=vae,
        clip_encoder=clip_encoder,
        accelerator=accelerator,
    )
    trainer.setup()

    if cfg.resume_from is not None:
        accelerator.print(f"[main] resuming from {cfg.resume_from}")
        trainer.load(cfg.resume_from)

    num_steps = int(cfg.algorithm.num_steps)
    eval_interval = int(cfg.eval_interval)
    save_interval = int(cfg.save_interval)
    log_interval = int(cfg.log_interval)
    ckpt_dir = workdir / "checkpoints"

    accelerator.print(f"[main] training for {num_steps - trainer.global_step} steps")
    while trainer.global_step < num_steps:
        t0 = time.time()
        metrics = trainer.step()
        step_time = time.time() - t0

        if accelerator.is_main_process and trainer.global_step % log_interval == 0:
            log = {f"train/{k}": v for k, v in metrics.items()}
            log["train/step_time"] = step_time
            if wandb_run is not None:
                import wandb
                wandb.log(log, step=trainer.global_step)

        if trainer.global_step % eval_interval == 0 or trainer.global_step == num_steps:
            accelerator.print(f"[eval] step={trainer.global_step}")
            eval_metrics = trainer.eval()
            if accelerator.is_main_process and wandb_run is not None:
                import wandb
                wandb_payload = {
                    f"eval/{k}": v for k, v in eval_metrics.items()
                    if k != "sample_images"
                }
                for label, img_t in eval_metrics.get("sample_images", {}).items():
                    img_np = ((img_t.permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype("uint8")
                    wandb_payload[f"eval/class_{label}"] = wandb.Image(img_np)
                wandb.log(wandb_payload, step=trainer.global_step)

        if trainer.global_step % save_interval == 0 or trainer.global_step == num_steps:
            accelerator.print(f"[save] step={trainer.global_step}")
            trainer.save(str(ckpt_dir / f"checkpoint_{trainer.global_step}.pth"))

    accelerator.print(f"[main] done. final step={trainer.global_step}")


if __name__ == "__main__":
    main()
