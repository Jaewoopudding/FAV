"""Stable-Diffusion VAE decode wrapper for latent-diffusion backbones.

Decodes (B, 4, 32, 32) latents to (B, 3, 256, 256) pixels in [-1, 1] using
per-backbone normalization stats:

    decode:  raw_latent = final * std + mean
    encode:  final      = (raw_latent - mean) / std
"""
from __future__ import annotations

from typing import Sequence

import torch
from diffusers.models import AutoencoderKL


# Per-backbone latent statistics (iMF uses sd-vae-ft-MSE, IMM uses ft-EMA).

IMF_STATS = {
    "vae_type": "mse",
    "mean": [0.86488, -0.27787343, 0.21616915, 0.3738409],
    "std":  [4.85503674, 5.31922414, 3.93725398, 3.9870003],
}

IMM_STATS = {
    "vae_type": "ema",
    "mean": [1.56, -0.695, 0.483, 0.729],
    # std = raw_std / final_std, with final_std = 0.5
    "std":  [5.27 * 2, 5.91 * 2, 4.21 * 2, 4.31 * 2],
}


class VAEWrapper:
    """Decode-only wrapper around ``stabilityai/sd-vae-ft-{vae_type}``."""

    LATENT_SIZE = 32

    def __init__(
        self,
        *,
        mean: Sequence[float],
        std: Sequence[float],
        device: torch.device | str = "cuda",
        decode_batch_size: int = 8,
        vae_type: str = "mse",
        dtype: torch.dtype = torch.float32,
        mix_context=None,
        compile_decode: bool = False,
    ) -> None:
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{vae_type}",
            torch_dtype=dtype,
            local_files_only=False,
        )
        del vae.encoder
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()
        vae = vae.to(device)
        vae.to(memory_format=torch.channels_last)

        self.vae = vae
        self.dtype = dtype
        self.batch_size = decode_batch_size
        self.mix_context = mix_context
        self.vae_type = vae_type

        if compile_decode:
            self.compiled_decode = torch.compile(
                self.vae.decode, mode="reduce-overhead", fullgraph=True
            )
        else:
            self.compiled_decode = self.vae.decode

        self.mean = torch.as_tensor(list(mean), dtype=torch.float32)
        self.std = torch.as_tensor(list(std), dtype=torch.float32)

    def decode(self, latents: torch.Tensor, enable_grad: bool = False) -> torch.Tensor:
        """Decode (B, 4, 32, 32) latents to (B, 3, 256, 256) pixels in [-1, 1].

        ``enable_grad=True`` uses the uncompiled path so gradients flow through.
        """
        assert latents.shape[1:] == (4, self.LATENT_SIZE, self.LATENT_SIZE), (
            f"expected (B, 4, 32, 32), got {tuple(latents.shape)}"
        )
        # Cast to VAE dtype (the conv layers do not auto-cast).
        if latents.dtype != self.dtype:
            latents = latents.to(self.dtype)
        latents = latents * self.std.view(-1, 1, 1).to(latents.device, dtype=latents.dtype) + \
                  self.mean.view(-1, 1, 1).to(latents.device, dtype=latents.dtype)
        latents = latents.contiguous(memory_format=torch.channels_last)

        decode_fn = self.vae.decode if enable_grad else self.compiled_decode

        if self.mix_context is not None:
            with self.mix_context:
                out = decode_fn(latents)["sample"]
        else:
            out = decode_fn(latents)["sample"]

        return out.contiguous(memory_format=torch.channels_last)
