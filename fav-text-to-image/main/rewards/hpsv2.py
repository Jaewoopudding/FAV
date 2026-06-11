"""HPSv2 reward (fine-tuned ViT-H/14): cosine similarity between normalized image
features and the current prompt's text features.

``use_raw_prompt=True`` scores against the raw prompt; the default wraps it as
``"a photo of a {prompt}"``.
"""
from __future__ import annotations

import os

import torch
import torchvision

from .base import Reward


class HPSv2Reward(Reward):
    name = "hpsv2"
    is_text_conditioned = True
    feature_dim = 1024

    def __init__(self, *, dtype: torch.dtype, device, use_raw_prompt: bool = False) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.use_raw_prompt = use_raw_prompt

        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

        model, _, _ = create_model_and_transforms(
            "ViT-H-14", "laion2B-s32B-b79K",
            precision=dtype, device="cpu", jit=False,
            force_quick_gelu=False, force_custom_text=False,
            force_patch_dropout=False, force_image_size=None,
            pretrained_image=False, image_mean=None, image_std=None,
            light_augmentation=True, aug_cfg={}, output_dict=True,
            with_score_predictor=False, with_region_predictor=False,
        )

        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download("xswu/HPSv2", "HPS_v2_compressed.pt")
        except Exception:
            checkpoint_path = os.path.expanduser(
                "~/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/"
                "697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        self.model = model.to(device, dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self._tokenizer = get_tokenizer("ViT-H-14")
        self.resize = torchvision.transforms.Resize(224, antialias=True)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        self.text_features_dict: dict[int, torch.Tensor] = {}
        self._current_text_features = None
        self.eval()

    def build_labels(self, prompts: list[str]) -> None:
        """Build the prompt-index → text-feature table."""
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                if self.use_raw_prompt:
                    text = prompt
                else:
                    text = f"a photo of a {prompt.split(',')[0].strip()}"
                tokens = self._tokenizer([text]).to(self.device)
                feat = self.model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                self.text_features_dict[i] = feat.squeeze(0).to(self.dtype)

    def set_labels(self, labels: torch.Tensor) -> None:
        self._current_text_features = torch.stack(
            [self.text_features_dict[int(l.item())] for l in labels]
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        x = (images / 2 + 0.5).clamp(0, 1)
        x = self.resize(x)
        x = self.normalize(x).to(self.dtype)
        features = self.model.encode_image(x)
        return features / features.norm(dim=-1, keepdim=True)

    def forward(self, images_or_features: torch.Tensor) -> torch.Tensor:
        if images_or_features.dim() == 2:
            embed = images_or_features
        else:
            embed = self.encode_images(images_or_features)
        return (embed * self._current_text_features).sum(dim=-1)
