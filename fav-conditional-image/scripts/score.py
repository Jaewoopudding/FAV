"""Score a directory of generated images with a chosen reward model.

    python -m scripts.score samples_dir=runs/imf_fav/samples reward=aesthetic
"""
from __future__ import annotations

import csv
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True

from main.algorithms import _common


def _load_images_to_tensor(paths: list[Path], device, dtype) -> torch.Tensor:
    arrs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    arr = np.stack(arrs).astype(np.float32) / 255.0
    arr = arr * 2 - 1  # to [-1, 1]
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
    return t


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    samples_dir = Path(cfg.get("samples_dir", "samples"))
    if not samples_dir.exists():
        raise FileNotFoundError(samples_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    reward, _ = _common.build_reward(cfg.reward, device=device, dtype=dtype)

    paths = sorted(samples_dir.glob("*.png"))
    if not paths:
        print(f"[score] no PNGs found under {samples_dir}")
        return

    micro = int(cfg.runtime.micro_batch_size)
    rows: list[tuple[str, float]] = []

    with torch.no_grad():
        for i in range(0, len(paths), micro):
            chunk = paths[i : i + micro]
            imgs = _load_images_to_tensor(chunk, device, dtype)
            scores = reward(imgs)
            if not torch.is_tensor(scores):
                scores = torch.as_tensor(scores)
            scores = scores.detach().float().cpu().numpy()
            for p, s in zip(chunk, scores):
                rows.append((p.name, float(s)))

    csv_path = samples_dir / f"scores_{cfg.reward.name}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", f"reward_{cfg.reward.name}"])
        w.writerows(rows)

    arr = np.array([r[1] for r in rows], dtype=np.float64)
    print(f"[score] {cfg.reward.name}: n={len(arr)}, mean={arr.mean():.4f}, "
          f"std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")
    print(f"[score] wrote {csv_path}")


if __name__ == "__main__":
    main()
