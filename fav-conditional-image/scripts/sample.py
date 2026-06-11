"""Sample images from a backbone (optionally with a LoRA checkpoint).

Example:
    python -m scripts.sample backbone=imf num_samples=64 \\
        ckpt=runs/imf_fav/checkpoints/checkpoint_1000.pth \\
        labels=[207,209,279] out_dir=runs/imf_fav/samples_step1000
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from main.algorithms import _common
from main.backbones.base import ForwardMode
from main.backbones.imf import IMFBackbone
from main.utils.distributed import BatchGenerator, seed_everything


def build_backbone(cfg_backbone: DictConfig):
    name = cfg_backbone.name
    if name == "imf":
        return IMFBackbone(
            model_str=cfg_backbone.model_str,
            dtype=getattr(torch, cfg_backbone.get("dtype", "float32")),
            img_size=int(cfg_backbone.img_size),
            img_channels=int(cfg_backbone.img_channels),
            num_classes=int(cfg_backbone.num_classes),
        )
    raise NotImplementedError(f"Backbone {name!r} is not yet ported.")


def _save_grid_png(out_dir: Path, images: torch.Tensor, labels: torch.Tensor) -> None:
    """Save each (B, 3, H, W) image in [-1, 1] as a PNG under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = ((images.detach().float().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    arr = arr.transpose(0, 2, 3, 1)
    for i, (img, lab) in enumerate(zip(arr, labels.tolist())):
        Image.fromarray(img).save(out_dir / f"{i:06d}_class{int(lab):04d}.png")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    num_samples: int = int(cfg.get("num_samples", 16))
    out_dir = Path(cfg.get("out_dir", "samples"))
    ckpt: Optional[str] = cfg.get("ckpt", None)
    label_list = cfg.get("labels", None) or list(cfg.algorithm.labels)
    labels_t = torch.tensor(list(label_list), dtype=torch.int32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.backbone.get("dtype", "float32"))

    seed_everything(int(cfg.get("seed", 0)))

    print(f"[sample] backbone={cfg.backbone.name}, num_samples={num_samples}, "
          f"labels={label_list}, ckpt={ckpt}, out={out_dir}")

    backbone = build_backbone(cfg.backbone)
    backbone.load_pretrained(cfg.backbone.ckpt_path)
    backbone.to(device)

    if ckpt is not None:
        bb_lora = cfg.backbone.lora
        backbone.inject_lora(
            rank=int(bb_lora.rank),
            alpha=bb_lora.get("alpha", None),
            dropout=float(bb_lora.get("dropout", 0.0)),
            target_modules=bb_lora.get("target_modules", None),
        )
        ckpt_state = torch.load(ckpt, map_location="cpu")
        backbone.load_lora_state(ckpt_state["lora_state"])
        print(f"[sample] loaded LoRA from {ckpt} (step={ckpt_state.get('global_step')})")

    backbone.forward_mode = ForwardMode.SAMPLE
    backbone.eval()

    vae = _common.build_vae(
        decode_batch_size=int(cfg.runtime.gen_bsz), dtype=dtype,
    )

    sample_kwargs = dict(cfg.backbone.sample)

    # Round-robin labels to fill num_samples.
    rep = (num_samples + len(labels_t) - 1) // len(labels_t)
    full_labels = labels_t.repeat(rep)[:num_samples]

    micro = int(cfg.runtime.micro_batch_size)
    seed_base = int(cfg.get("seed", 0))
    n_chunks = (num_samples + micro - 1) // micro

    with torch.no_grad():
        for ch in range(n_chunks):
            mb_labels = full_labels[ch * micro : (ch + 1) * micro]
            B = mb_labels.shape[0]
            seeds = torch.arange(seed_base + ch * micro, seed_base + ch * micro + B)
            rng = BatchGenerator(device=device, seeds=seeds)
            x = backbone._sample(n_sample=B, rng=rng, labels=mb_labels, **sample_kwargs)
            images = vae.decode(x, enable_grad=False)
            _save_grid_png(out_dir, images, mb_labels)
            print(f"[sample] chunk {ch+1}/{n_chunks}: saved {B} images")

    print(f"[sample] done. {num_samples} images in {out_dir}")


if __name__ == "__main__":
    main()
