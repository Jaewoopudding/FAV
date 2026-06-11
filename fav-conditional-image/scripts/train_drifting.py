"""Hydra-to-argv shim that runs the vendored Drifting (JAX) trainer.

    python -m scripts.train_drifting backbone=drifting algorithm=fav reward=aesthetic

Requires JAX (``pip install -e '.[jax]'``).
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


_VENDOR_DIR = Path(__file__).resolve().parents[1] / "main" / "backbones" / "drifting" / "_vendor"


def _build_argv(cfg: DictConfig, script_stem: str) -> list[str]:
    """Translate Hydra cfg into the argv expected by Drifting's argparse."""
    algo = cfg.algorithm
    bb = cfg.backbone
    rt = cfg.runtime
    argv: list[str] = [
        f"{script_stem}.py",
        "--init-from", str(bb.get("init_from", "hf://latent_L_sota")),
        "--workdir", str(Path(cfg.workdir) / (cfg.get("run_name") or f"{bb.name}_{algo.name}")),
        "--mlp-weights", str(cfg.reward.mlp_ckpt),
        "--cfg-scale", str(float(bb.sample.get("cfg_omega", 1.0))),
        "--num-steps", str(int(algo.num_steps)),
        "--lr", str(float(algo.optimizer.lr)),
        "--warmup-steps", str(int(algo.scheduler.get("warmup_steps", 0))),
        "--max-grad-norm", str(float(algo.max_grad_norm)),
        "--micro-batch-size", str(int(rt.micro_batch_size)),
        "--eval-batch-size", str(int(rt.eval_batch_size)),
        "--eval-interval", str(int(cfg.eval_interval)),
        "--save-interval", str(int(cfg.save_interval)),
        "--global-seed", str(int(cfg.get("seed", 0))),
        "--lora-rank", str(int(bb.lora.rank)),
        "--lora-alpha", str(float(bb.lora.get("alpha", bb.lora.rank))),
    ]
    if bool(cfg.get("wandb", {}).get("enabled", False)):
        argv += ["--use-wandb", "--wandb-project", str(cfg.wandb.project)]
        if cfg.wandb.get("entity", None):
            argv += ["--wandb-entity", str(cfg.wandb.entity)]
    if "labels" in algo:
        argv += ["--labels", *[str(int(l)) for l in algo.labels]]
    if algo.name == "fav":
        argv += [
            "--beta", str(float(algo.beta)),
            "--temp-kde", *[str(t) for t in (algo.temp_kde if hasattr(algo.temp_kde, "__iter__") else [algo.temp_kde])],
            "--temp-stein", *[str(t) for t in (algo.temp_stein if hasattr(algo.temp_stein, "__iter__") else [algo.temp_stein])],
        ]
        if "classes_per_step" in algo:
            argv += ["--classes-per-step", str(int(algo.classes_per_step))]
        # batch_size_per_class == gen_per_class given our effective_batch_size convention.
        bspc = int(algo.effective_batch_size) // (2 * int(algo.classes_per_step))
        argv += ["--batch-size-per-class", str(bspc)]
    elif algo.name == "draft":
        if "reward_multiplier" in algo:
            argv += ["--reward-multiplier", str(float(algo.reward_multiplier))]
        argv += ["--total-batch-size", str(int(algo.total_batch_size))]
    return argv


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.backbone.name != "drifting":
        raise ValueError(
            f"train_drifting expects backbone=drifting, got {cfg.backbone.name!r}. "
            f"Use scripts.train for PyTorch backbones."
        )
    algo_name = cfg.algorithm.name
    if algo_name == "fav":
        script_name = "drifting_aligen.py"
    elif algo_name == "draft":
        script_name = "drifting_draft.py"
    else:
        raise ValueError(
            f"Drifting only supports FAV and DRaFT — got algorithm={algo_name!r}."
        )

    script_path = _VENDOR_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    # Let vendored Drifting modules resolve under their original absolute names.
    sys.path.insert(0, str(_VENDOR_DIR))
    sys.argv = _build_argv(cfg, script_path.stem)

    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
