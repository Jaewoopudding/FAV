"""JPEG compressibility / incompressibility rewards (non-differentiable)."""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

from .base import Reward


def _jpeg_size_kb(images: torch.Tensor | np.ndarray, *, quality: int = 95) -> np.ndarray:
    """Encode each image as JPEG and return the resulting file size in KB."""
    if isinstance(images, torch.Tensor):
        images_np = (
            ((images / 2 + 0.5).clamp(0, 1) * 255)
            .byte().cpu().permute(0, 2, 3, 1).numpy()
        )
    else:
        images_np = images

    sizes = []
    for img in images_np:
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        sizes.append(buf.tell() / 1000)
    return np.array(sizes, dtype=np.float32)


class JPEGIncompressibilityReward(Reward):
    """Reward is the JPEG size: higher means more high-frequency detail."""

    name = "jpeg_incompress"

    def __init__(self, quality: int = 95) -> None:
        self.quality = quality

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        sizes = _jpeg_size_kb(images, quality=self.quality)
        return torch.as_tensor(sizes, device=images.device, dtype=torch.float32)


class JPEGCompressibilityReward(Reward):
    """Reward is the negative JPEG size: higher means a smoother image."""

    name = "jpeg_compress"

    def __init__(self, quality: int = 95) -> None:
        self.quality = quality

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        sizes = _jpeg_size_kb(images, quality=self.quality)
        return torch.as_tensor(-sizes, device=images.device, dtype=torch.float32)
