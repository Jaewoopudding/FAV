"""LAION aesthetic predictor: CLIP ViT-L/14 features to an MLP score."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .base import Reward
from ..utils.clip import CLIPEncoder


class _AestheticMLP(nn.Module):
    """LAION sac+logos+ava1-l14-linearMSE score predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return self.layers(embed)


class AestheticReward(Reward, nn.Module):
    """Differentiable aesthetic-score reward.

    Accepts pixel images ``(B, 3, H, W)`` in ``[-1, 1]`` or pre-computed CLIP
    embeddings ``(B, 768)``.
    """

    name = "aesthetic"

    def __init__(
        self,
        mlp_ckpt: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
        clip_encoder: Optional[CLIPEncoder] = None,
        clip_model_name: str = "openai/clip-vit-large-patch14",
    ) -> None:
        nn.Module.__init__(self)
        self.dtype = dtype

        self.clip_encoder = clip_encoder or CLIPEncoder(
            model_name=clip_model_name, dtype=dtype
        )
        self.mlp = _AestheticMLP()
        state_dict = torch.load(str(mlp_ckpt), map_location="cpu")
        self.mlp.load_state_dict(state_dict)
        self.eval()

    def __call__(self, images_or_features: torch.Tensor) -> torch.Tensor:
        return nn.Module.__call__(self, images_or_features)

    def forward(self, images_or_features: torch.Tensor) -> torch.Tensor:
        if images_or_features.dim() == 2 and images_or_features.shape[-1] == 768:
            embed = images_or_features
        else:
            embed = self.clip_encoder(images_or_features)
        return self.mlp(embed).squeeze(1)
