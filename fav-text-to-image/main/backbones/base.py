"""Backbone abstraction for text-to-image reward alignment.

A new HuggingFace diffusion backbone implements this interface and the trainers,
losses, and rewards work unchanged. Only ``sample`` (and ``decode_latents`` with
``enable_grad``) carry gradient to the LoRA parameters; everything else is frozen.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterable


class Backbone(ABC):
    """Backend-agnostic text-to-image backbone interface."""

    name: str = "abstract"

    @abstractmethod
    def load_models(self, device) -> None:
        """Build the transformer, VAE, text encoder, and sampler on ``device``."""

    @abstractmethod
    def encode_prompts(self, prompts: list[str], device) -> list[tuple]:
        """Pre-encode prompts → list of ``(caption_embs, emb_masks)`` per prompt."""

    def free_text_encoder(self) -> None:
        """Release the text encoder after prompt caching (optional)."""

    @abstractmethod
    def inject_lora(self, *, rank: int, alpha: float, dropout: float,
                    target_modules: Iterable[str]) -> tuple[int, int]:
        """Inject LoRA; return ``(n_trainable, n_total)``."""

    @abstractmethod
    def get_lora_state(self) -> dict: ...

    @abstractmethod
    def load_lora_state(self, state: dict) -> None: ...

    @abstractmethod
    def trainable_parameters(self) -> Iterable: ...

    @abstractmethod
    def prepare(self, accelerator) -> None:
        """DDP-wrap the trainable transformer via ``accelerator.prepare``."""

    @abstractmethod
    def make_noise(self, seeds, device):
        """Deterministic initial noise for the given per-sample integer seeds."""

    @abstractmethod
    def sample(self, noise, caption_embs, emb_masks, *, cfg_scale: float,
               grad_checkpoint_sampling: bool = False):
        """Differentiable few-step generation → latents (model scale)."""

    @abstractmethod
    def decode_latents(self, latents, *, enable_grad: bool):
        """Decode latents → pixel images in ``[-1, 1]``."""

    @contextmanager
    def reference_context(self):
        """Context in which ``sample`` produces the reference distribution
        (default no-op; Sana toggles LoRA adapters off)."""
        yield
