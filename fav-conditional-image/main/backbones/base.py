"""Backbone abstract interface. PyTorch backbones route training-time calls
through ``forward()`` so DDP gradient sync fires correctly."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class ForwardMode(str, Enum):
    """SAMPLE: full generation chain. VELOCITY: single-step velocity prediction."""

    SAMPLE = "sample"
    VELOCITY = "velocity"


class Backbone(ABC):
    """Backend-agnostic interface (PyTorch or JAX)."""

    is_jax: bool = False
    supports_velocity_mode: bool = False
    supports_sample_mode: bool = True

    @abstractmethod
    def load_pretrained(self, ckpt_path: str) -> None:
        """Load backbone weights from disk."""

    @abstractmethod
    def inject_lora(self, rank: int, **kwargs: Any) -> None:
        """Replace target modules with LoRA-augmented variants in-place."""

    @abstractmethod
    def get_lora_state(self) -> dict:
        """Return a state dict containing only the trainable LoRA parameters."""

    @abstractmethod
    def load_lora_state(self, state: dict) -> None:
        """Load a state dict produced by ``get_lora_state``."""


try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class TorchBackbone(Backbone, nn.Module):
        """PyTorch backbone with a forward dispatcher for DDP safety.

        Subclasses implement ``_sample`` (required) and optionally ``_velocity``;
        the active path is chosen by ``forward_mode``.
        """

        forward_mode: ForwardMode = ForwardMode.SAMPLE

        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.forward_mode = ForwardMode.SAMPLE

        def forward(self, **kwargs: Any):
            if self.forward_mode == ForwardMode.SAMPLE:
                return self._sample(**kwargs)
            if self.forward_mode == ForwardMode.VELOCITY:
                if not self.supports_velocity_mode:
                    raise NotImplementedError(
                        f"{type(self).__name__} does not support VELOCITY mode"
                    )
                return self._velocity(**kwargs)
            raise ValueError(f"unknown forward_mode: {self.forward_mode!r}")

        @abstractmethod
        def _sample(self, **kwargs: Any):
            """Run the full (potentially differentiable) generation chain."""

        def _velocity(self, **kwargs: Any):
            """Single-step velocity / score prediction."""
            raise NotImplementedError

else:  # pragma: no cover

    class TorchBackbone(Backbone):  # type: ignore[no-redef]
        """Stub when PyTorch is not installed."""

        def __init__(self) -> None:
            raise ImportError("torch is required for TorchBackbone")


class JAXBackbone(Backbone):
    """JAX/Flax backbone (currently only Drifting)."""

    is_jax: bool = True

    @abstractmethod
    def sample(self, *, n_sample: int, labels: Any, rng: Any, **kwargs: Any):
        """Generate samples (JAX arrays). Used by ``scripts/sample.py``."""
