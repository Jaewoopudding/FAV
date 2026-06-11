"""Hydra training entrypoint for FAV / DRaFT.

    python -m scripts.train +experiment=sana_fav_hpsv2
    python -m scripts.train backbone=sana_sprint_hf algorithm=draft reward=hpsv2
"""
from __future__ import annotations

import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from main.algorithms import _common, build_trainer
from main.backbones import build_backbone
from main.rewards import build_reward
from main.utils.distributed import seed_everything
from main.utils.logging import init_wandb
from main.utils.prompts import load_prompts

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)


def compose_run_name(cfg: DictConfig) -> str:
    if cfg.run_name is not None:
        return str(cfg.run_name)
    ts = time.strftime("%m%d_%H%M%S")
    return f"{ts}_{cfg.backbone.name}_{cfg.algorithm.name}_{cfg.reward.name}_seed{cfg.seed}"


def _grad_accum(cfg, n_gpu, micro, n_train):
    """Per-algorithm ``(grad_accum_steps, total_accum)``."""
    algo = cfg.algorithm
    name = algo.name
    if name == "fav":
        # FAV applies its own per-chunk normalization, so accelerate must not
        # also divide the loss.
        return 1, 1
    if name == "draft":
        grad_accum = max(int(algo.total_batch_size) // n_gpu // micro, 1)
        return grad_accum, grad_accum
    raise NotImplementedError(name)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train_prompts = load_prompts(_resolve(cfg.train_prompt_file))
    eval_prompts = load_prompts(_resolve(cfg.eval_prompt_file))
    n_train = len(train_prompts)

    # Light __init__: weights load in setup().
    backbone = build_backbone(cfg.backbone)

    micro = int(cfg.runtime.micro_batch_size)
    n_gpu = _common.compute_n_gpu()
    grad_accum, total_accum = _grad_accum(cfg, n_gpu, micro, n_train)
    cfg.runtime.grad_accum_steps = grad_accum

    mp = cfg.runtime.get("mixed_precision", None)
    if mp in (None, "null", ""):
        mp = _common.mixed_precision_mode(backbone.weight_dtype)
    accelerator = _common.build_accelerator(
        gradient_accumulation_steps=total_accum, mixed_precision=mp,
    )
    device = accelerator.device

    run_name = compose_run_name(cfg)
    workdir = REPO_ROOT / cfg.workdir / run_name
    if accelerator.is_main_process:
        workdir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.print(f"[main] run={run_name}")
    accelerator.print(f"[main] world_size={accelerator.num_processes}, n_gpu={n_gpu}")
    accelerator.print(
        f"[main] algorithm={cfg.algorithm.name}, micro={micro}, "
        f"grad_accum={grad_accum}, total_accum={total_accum}, mixed_precision={mp}"
    )

    wandb_run = None
    if accelerator.is_main_process:
        wandb_run = init_wandb(
            project=str(cfg.wandb.project), run_name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            enabled=bool(cfg.wandb.enabled),
        )

    seed_everything(int(cfg.seed))

    # Reward built over train + eval prompts; eval labels use n_train + eidx.
    train_reward, raw_reward = build_reward(
        cfg.reward, prompts=train_prompts + eval_prompts, device=device, dtype=torch.float32,
    )
    accelerator.print(f"[main] reward={cfg.reward.name} over {n_train + len(eval_prompts)} prompts")

    trainer = build_trainer(
        cfg, backbone=backbone, train_reward=train_reward, raw_reward=raw_reward,
        accelerator=accelerator, train_prompts=train_prompts, eval_prompts=eval_prompts,
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
            if accelerator.is_main_process:
                accelerator.print(
                    f"  reward mean {eval_metrics['reward_mean']:.4f} "
                    f"± {eval_metrics['reward_std']:.4f}"
                )
                if wandb_run is not None:
                    import numpy as np
                    import wandb
                    payload = {f"eval/{k}": v for k, v in eval_metrics.items() if k != "sample_images"}
                    for eidx, (img_t, prompt) in eval_metrics.get("sample_images", {}).items():
                        arr = ((img_t.permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                        payload[f"eval/prompt_{eidx}"] = wandb.Image(arr, caption=prompt[:60])
                    wandb.log(payload, step=trainer.global_step)

        if trainer.global_step % save_interval == 0 or trainer.global_step == num_steps:
            accelerator.print(f"[save] step={trainer.global_step}")
            trainer.save(str(ckpt_dir / f"checkpoint_{trainer.global_step}.pth"))

    accelerator.print(f"[main] done. final step={trainer.global_step}")
    if accelerator.is_main_process and wandb_run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
