"""Fine-tune 2D generators on the 8gaussians toy with FAV / RTB / Adjoint Matching.

  python finetune.py --model {drift,meanflow,vae} --algo {fav,adjoint_matching,reg_reinforce}

adjoint_matching and reg_reinforce are MeanFlow only (require ODE structure).
"""
import argparse
import json
import os
import shutil
import sys
from typing import Optional

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from utils_2d import (
    get_energy_fn,
    load_dataset,
    sample_from_target,
    mmd_squared_rbf,
    kl_divergence_kde_2d,
)
from pretrain.drift    import DriftModel
from pretrain.meanflow import MeanFlowModel
from pretrain.vae      import VAEModel
from finetune          import fav, adjoint_matching, reg_reinforce

CKPT_ROOT    = os.path.join(_HERE, "checkpoints")
RESULTS_ROOT = os.path.join(_HERE, "results")
DATASET      = "8gaussians"

_VALID_ALGOS = {
    "drift":    ["fav"],
    "meanflow": ["fav", "adjoint_matching", "reg_reinforce"],
    "vae":      ["fav"],
}

# Best per-n_steps noise levels found via sweep; user can override with --noise_level.
_RTB_NOISE_LEVELS = {2: 0.01, 4: 0.05, 8: 0.10, 16: 0.20}

def _copy_best_ckpt(save_dir: str, ckpt_dir: str) -> None:
    src = os.path.join(save_dir, "model_best.pt")
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(ckpt_dir, "model_best.pt"))

def _dump_config(save_dir: str, args, cfg: dict, extras: Optional[dict] = None) -> None:
    os.makedirs(save_dir, exist_ok=True)
    out = {"args": vars(args).copy(), "cfg": dict(cfg)}
    if extras:
        out["extras"] = dict(extras)
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

def _load_pretrained(model_name: str, device: torch.device) -> torch.nn.Module:
    ckpt = os.path.join(CKPT_ROOT, model_name, DATASET, "model_pretrained.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"No pretrained checkpoint at {ckpt}.\n"
            f"Run: python pretrain.py --model {model_name} first."
        )
    if model_name == "drift":
        model = DriftModel(in_dim=32, hidden=256, data_dim=2)
    elif model_name == "meanflow":
        model = MeanFlowModel(data_dim=2, hidden=256)
    elif model_name == "vae":
        model = VAEModel(latent_dim=8, hidden=256, data_dim=2)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    print(f"  Loaded {model_name} checkpoint from {ckpt}")
    return model

@torch.no_grad()
def _sample_pretrained_pool(
    model_name: str,
    device: torch.device,
    n_total: int = 200_000,
    batch: int = 10_000,
    sample_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    sample_kwargs = sample_kwargs or {}
    model = _load_pretrained(model_name, device)
    model.eval()
    pool = []
    remaining = n_total
    while remaining > 0:
        n = min(batch, remaining)
        pool.append(model.sample(n, device, **sample_kwargs).detach())
        remaining -= n
    pool = torch.cat(pool, dim=0)
    tag = f"  (n_steps={sample_kwargs['n_steps']})" if "n_steps" in sample_kwargs else ""
    print(f"  Pretrained pos pool: {pool.shape[0]:,} samples from {model_name}{tag}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pool

_BETA = 1.0

_FINETUNE_CFG = {
    "fav": dict(
        num_steps=20_000, batch_size=2048, lr=1e-4,
        beta=_BETA, metric_every=100,
    ),
    "adjoint_matching": dict(
        num_steps=20_000, batch_size=256, lr=1e-4,
        beta=_BETA, num_timesteps=10, max_grad_norm=1.0,
        metric_every=100,
    ),
    "reg_reinforce": dict(
        num_steps=20_000, batch_size=256, lr=1e-4,
        beta=_BETA, max_grad_norm=1.0,
        metric_every=100,
    ),
}

def run_finetune(args) -> None:
    if args.seed is not None:
        import random, numpy as np
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"  Seed: {args.seed}")

    device = torch.device(args.device)
    model_name = args.model
    algo = args.algo

    if algo not in _VALID_ALGOS[model_name]:
        raise ValueError(
            f"Algorithm '{algo}' is not compatible with model '{model_name}'. "
            f"{model_name} supports: {_VALID_ALGOS[model_name]}."
        )

    print(f"\n{'='*60}")
    print(f" Model: {model_name}   Algo: {algo}")
    print(f"{'='*60}")

    seed_tag    = f"_seed{args.seed}" if args.seed is not None else ""
    run_suffix  = seed_tag + args.run_tag
    save_dir    = os.path.join(RESULTS_ROOT, "finetune", algo, model_name, DATASET + run_suffix)
    ckpt_dir    = os.path.join(CKPT_ROOT,    "finetune", algo, model_name, DATASET + run_suffix)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    energy_fn = get_energy_fn()

    model = _load_pretrained(model_name, device)
    data, _ = load_dataset(n=200_000, device=device)
    ref_data = data[:10_000]

    cfg = dict(_FINETUNE_CFG[algo])
    if args.num_steps is not None:
        cfg["num_steps"] = args.num_steps
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    if algo == "fav" and model_name == "meanflow":
        _MEANFLOW_STEPS = args.steps if args.steps else [1, 4, 16, 64]
        for n_steps in _MEANFLOW_STEPS:
            print(f"\n  [MeanFlow FAV] n_steps={n_steps}")
            step_save_dir = os.path.join(save_dir, f"steps_{n_steps}")
            step_ckpt_dir = os.path.join(ckpt_dir, f"steps_{n_steps}")
            os.makedirs(step_save_dir, exist_ok=True)
            os.makedirs(step_ckpt_dir, exist_ok=True)

            step_model = _load_pretrained(model_name, device)
            # pos_pool must use the same n_steps as the generator.
            pos_pool = _sample_pretrained_pool(
                model_name, device, sample_kwargs={"n_steps": n_steps}
            )
            _dump_config(step_save_dir, args, cfg,
                         extras={"n_steps": n_steps, "temp": args.temp})
            fav.finetune(
                step_model, data, device, energy_fn,
                save_dir=step_save_dir, dataset_name=DATASET,
                ref_data=ref_data, sample_kwargs={"n_steps": n_steps},
                temp=args.temp, pos_pool=pos_pool, **cfg,
            )
            torch.save(step_model.state_dict(),
                       os.path.join(step_ckpt_dir, "model_finetuned.pt"))
            _copy_best_ckpt(step_save_dir, step_ckpt_dir)

    elif algo == "fav":
        pos_pool = _sample_pretrained_pool(model_name, device)
        _dump_config(save_dir, args, cfg, extras={"temp": args.temp})
        fav.finetune(
            model, data, device, energy_fn,
            save_dir=save_dir, dataset_name=DATASET,
            ref_data=ref_data, temp=args.temp, pos_pool=pos_pool, **cfg,
        )

    elif algo == "adjoint_matching":
        _AM_STEPS = args.steps if args.steps else [2, 4, 8, 16]
        for n_steps in _AM_STEPS:
            print(f"\n  [MeanFlow AM] n_steps={n_steps}")
            step_save_dir = os.path.join(save_dir, f"steps_{n_steps}")
            step_ckpt_dir = os.path.join(ckpt_dir, f"steps_{n_steps}")
            os.makedirs(step_save_dir, exist_ok=True)
            os.makedirs(step_ckpt_dir, exist_ok=True)

            step_model = _load_pretrained(model_name, device)
            am_cfg = dict(cfg)
            am_cfg["num_timesteps"] = n_steps
            _dump_config(step_save_dir, args, am_cfg, extras={"n_steps": n_steps})
            adjoint_matching.finetune(
                step_model, device, energy_fn,
                save_dir=step_save_dir, dataset_name=DATASET,
                ref_data=ref_data, **am_cfg,
            )
            torch.save(step_model.state_dict(),
                       os.path.join(step_ckpt_dir, "model_finetuned.pt"))
            _copy_best_ckpt(step_save_dir, step_ckpt_dir)

    elif algo == "reg_reinforce":
        _RR_STEPS = args.steps if args.steps else [4]
        for n_steps in _RR_STEPS:
            nl = (args.noise_level if args.noise_level is not None
                  else _RTB_NOISE_LEVELS.get(n_steps, 0.1))
            print(f"\n  [MeanFlow RTB] n_steps={n_steps}  noise_level={nl}")
            step_save_dir = os.path.join(save_dir, f"steps_{n_steps}")
            step_ckpt_dir = os.path.join(ckpt_dir, f"steps_{n_steps}")
            os.makedirs(step_save_dir, exist_ok=True)
            os.makedirs(step_ckpt_dir, exist_ok=True)

            step_model = _load_pretrained(model_name, device)
            rr_cfg = dict(cfg)
            rr_cfg["n_steps"] = n_steps
            rr_cfg["noise_level"] = nl
            _dump_config(step_save_dir, args, rr_cfg,
                         extras={"n_steps": n_steps, "noise_level": nl})
            reg_reinforce.finetune(
                step_model, device, energy_fn,
                save_dir=step_save_dir, dataset_name=DATASET,
                ref_data=ref_data, **rr_cfg,
            )
            torch.save(step_model.state_dict(),
                       os.path.join(step_ckpt_dir, "model_finetuned.pt"))
            _copy_best_ckpt(step_save_dir, step_ckpt_dir)
        return

    if model_name == "meanflow" and algo in ("fav", "adjoint_matching"):
        return

    beta = cfg.get("beta", 1.0)
    n_eval = len(ref_data)
    gt = sample_from_target(ref_data, energy_fn, n_eval, beta=beta)
    model.eval()
    with torch.no_grad():
        gen = (model.sample(n_eval, device, n_steps=64) if model_name == "meanflow"
               else model.sample(n_eval, device))
    kl   = kl_divergence_kde_2d(gt.cpu().numpy(), gen.cpu().numpy())
    mmd2 = mmd_squared_rbf(gt.to(device), gen)
    print(f"\n[{model_name}/{algo}] Final vs p(x)*r(x):  KL={kl:.4f}  MMD2={mmd2:.6f}")

    ckpt_path = os.path.join(ckpt_dir, "model_finetuned.pt")
    torch.save(model.state_dict(), ckpt_path)
    _copy_best_ckpt(save_dir, ckpt_dir)
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump({"kl": kl, "mmd2": mmd2, "dataset": DATASET,
                   "model": model_name, "algo": algo}, f, indent=2)
    print(f"  Checkpoint -> {ckpt_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune 2D generators (FAV / Adjoint Matching / RTB) on 8gaussians."
    )
    parser.add_argument("--model", required=True,
                        choices=["drift", "meanflow", "vae"])
    parser.add_argument("--algo",  required=True,
                        choices=["fav", "adjoint_matching", "reg_reinforce"])
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Override the per-run step list for MeanFlow.")
    parser.add_argument("--temp", type=float, default=0.05,
                        help="SVGD RBF bandwidth for FAV.")
    parser.add_argument("--num_steps",  type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--noise_level", type=float, default=None,
                        help="reg_reinforce only: override SDE noise level.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default="")
    args = parser.parse_args()
    run_finetune(args)

if __name__ == "__main__":
    main()
