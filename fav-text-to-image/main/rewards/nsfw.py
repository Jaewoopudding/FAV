"""Falconsai NSFW reward (ViT classifier): ``forward`` → (B,) NSFW probability in
[0, 1] (higher = more NSFW). To minimize NSFW, wrap with ``NegatedReward``."""
from __future__ import annotations

import torch
import torchvision

from .base import Reward


class NSFWReward(Reward):
    name = "nsfw"
    is_text_conditioned = False
    feature_dim = 768

    def __init__(self, *, dtype: torch.dtype, device) -> None:
        super().__init__()
        self.dtype = dtype

        from transformers import AutoModelForImageClassification, ViTImageProcessor

        full_model = AutoModelForImageClassification.from_pretrained(
            "Falconsai/nsfw_image_detection",
        )
        processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")

        self.vit = full_model.vit
        self.classifier = full_model.classifier
        self.nsfw_index = int(full_model.config.label2id.get("nsfw", 1))

        for p in self.parameters():
            p.requires_grad = False
        self.to(device, dtype=dtype)

        self.resize = torchvision.transforms.Resize(224, antialias=True)
        self.normalize = torchvision.transforms.Normalize(
            mean=processor.image_mean, std=processor.image_std,
        )
        self.eval()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        x = (images / 2 + 0.5).clamp(0, 1)
        x = self.resize(x)
        x = self.normalize(x).to(self.dtype)
        outputs = self.vit(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, images_or_features: torch.Tensor) -> torch.Tensor:
        if images_or_features.dim() == 2:
            embed = images_or_features
        else:
            embed = self.encode_images(images_or_features)
        logits = self.classifier(embed)
        return torch.softmax(logits, dim=-1)[:, self.nsfw_index]
