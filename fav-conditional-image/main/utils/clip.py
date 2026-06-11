"""Frozen CLIP ViT-L/14 image encoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import CLIPModel


class CLIPEncoder(nn.Module):
    """Maps (B, 3, H, W) pixels in [-1, 1] to (B, 768) L2-normalized embeddings."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.clip.gradient_checkpointing_enable()
        self.dtype = dtype

        self.resize = T.Resize(224, antialias=True)
        self.normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = (images / 2 + 0.5).clamp(0, 1)
        x = self.resize(x)
        x = self.normalize(x).to(self.dtype)
        embed = self.clip.get_image_features(pixel_values=x)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return embed
