"""Pretrain a 2D generator on the 8gaussians toy.

  python pretrain.py --model {drift,meanflow,vae}
"""
import argparse
import json
import os
import sys
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from utils_2d import (
    kl_divergence_kde_2d,
    load_dataset,
    mmd_squared_rbf,
)
from pretrain.drift    import DriftModel
from pretrain.meanflow import MeanFlowModel
from pretrain.vae      import VAEModel

CKPT_ROOT     = os.path.join(_HERE, "checkpoints")
RESULTS_ROOT  = os.path.join(_HERE, "results")
PRETRAIN_ROOT = os.path.join(RESULTS_ROOT, "pretrain")
REF_ROOT      = os.path.join(RESULTS_ROOT, "ref")
DATASET       = "8gaussians"

def _ref_path() -> str:
    return os.path.join(REF_ROOT, f"{DATASET}_ref.pt")

def load_or_create_ref(data: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    path = _ref_path()
    os.makedirs(REF_ROOT, exist_ok=True)
    if os.path.exists(path):
        ref = torch.load(path, map_location=device, weights_only=True)
        print(f"  Loaded reference samples  ({path})")
    else:
        perm = torch.randperm(data.shape[0])[:n]
        ref  = data[perm].cpu()
        torch.save(ref, path)
        print(f"  Saved reference samples   ({path})")
    return ref.to(device)

def build_model(model_name: str, device: torch.device) -> torch.nn.Module:
    if model_name == "drift":
        return DriftModel(in_dim=32, hidden=256, data_dim=2).to(device)
    if model_name == "meanflow":
        return MeanFlowModel(data_dim=2, hidden=256, depth=3,
                             activation="silu").to(device)
    if model_name == "vae":
        return VAEModel(latent_dim=8, hidden=256, data_dim=2).to(device)
    raise ValueError(f"Unknown model: {model_name}")

@torch.no_grad()
def sample_final(model_name: str, model, n: int, device: torch.device) -> torch.Tensor:
    if model_name == "meanflow":
        return model.sample(n, device, n_steps=1)
    return model.sample(n, device)

_PRETRAIN_CFG = {
    "drift": dict(
        num_steps=1_000_000, batch_size=8192, lr=1e-3, temp=0.15,
        metric_every=10_000,
    ),
    "meanflow": dict(
        num_steps=1_000_000, batch_size=8192, lr=3e-4,
        fraction_equal=0.5, adp_p=1.0, monitor_n_steps=1,
        metric_every=10_000,
    ),
    "vae": dict(
        num_steps=1_000_000, batch_size=8192, lr=1e-3, beta=0.02,
        metric_every=10_000,
    ),
}

def run_pretrain(
    model_name: str,
    device: torch.device,
    data_n: int = 500_000,
    ref_n: int  = 10_000,
    overrides: Optional[dict] = None,
) -> None:
    print(f"\n{'='*60}")
    print(f" Model: {model_name}   Device: {device}")
    print(f"{'='*60}")

    save_dir = os.path.join(PRETRAIN_ROOT, model_name, DATASET)
    ckpt_dir = os.path.join(CKPT_ROOT,     model_name, DATASET)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    data, _  = load_dataset(n=data_n, device=device)
    ref_data = load_or_create_ref(data, ref_n, device)

    model = build_model(model_name, device)
    cfg   = dict(_PRETRAIN_CFG[model_name])
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    print(f"  cfg: {cfg}")

    model.pretrain(
        data, device,
        save_dir=save_dir, dataset_name=DATASET,
        ref_data=ref_data, **cfg,
    )

    model.eval()
    gen  = sample_final(model_name, model, ref_n, device)
    kl   = kl_divergence_kde_2d(ref_data.cpu().numpy(), gen.cpu().numpy())
    mmd2 = mmd_squared_rbf(ref_data, gen)
    print(f"\n[{model_name}] Final  KL={kl:.4f}  MMD2={mmd2:.6f}")

    ckpt_path = os.path.join(ckpt_dir, "model_pretrained.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump({"kl": kl, "mmd2": mmd2, "dataset": DATASET,
                   "model": model_name}, f, indent=2)
    print(f"  Checkpoint -> {ckpt_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain 2D generators on the 8gaussians toy."
    )
    parser.add_argument("--model", required=True,
                        choices=["drift", "meanflow", "vae"])
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_n", type=int, default=500_000)
    parser.add_argument("--ref_n",  type=int, default=10_000)
    parser.add_argument("--num_steps",  type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--beta",       type=float, default=None, help="VAE only.")
    parser.add_argument("--temp",       type=float, default=None, help="Drift only.")
    parser.add_argument("--fraction_equal", type=float, default=None,
                        help="MeanFlow only.")
    parser.add_argument("--adp_p", type=float, default=None, help="MeanFlow only.")
    parser.add_argument("--metric_every", type=int, default=None)
    args = parser.parse_args()

    overrides = {
        "num_steps":    args.num_steps,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "metric_every": args.metric_every,
    }
    if args.model == "vae":
        overrides["beta"] = args.beta
    if args.model == "drift":
        overrides["temp"] = args.temp
    if args.model == "meanflow":
        overrides["fraction_equal"] = args.fraction_equal
        overrides["adp_p"] = args.adp_p

    run_pretrain(
        model_name=args.model,
        device=torch.device(args.device),
        data_n=args.data_n,
        ref_n=args.ref_n,
        overrides=overrides,
    )

if __name__ == "__main__":
    main()
