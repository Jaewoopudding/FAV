"""Abstract trainer interface shared by all alignment algorithms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..backbones import Backbone
    from ..rewards import Reward


class Trainer(ABC):
    """Algorithm-side training loop."""

    name: str = "abstract"
    required_capabilities: tuple[str, ...] = ()

    def __init__(
        self,
        backbone: "Backbone",
        reward: "Reward",
        cfg: Any,
        accelerator: Any = None,
    ) -> None:
        self.backbone = backbone
        self.reward = reward
        self.cfg = cfg
        self.accelerator = accelerator
        self._validate_capabilities()

    def _validate_capabilities(self) -> None:
        for cap in self.required_capabilities:
            if not getattr(self.backbone, cap, False):
                raise RuntimeError(
                    f"Algorithm {self.name!r} requires backbone capability "
                    f"{cap!r}, which {type(self.backbone).__name__} does not provide."
                )

    @abstractmethod
    def setup(self) -> None:
        """Inject LoRA, configure backbone forward_mode, build optimizer/scheduler."""

    @abstractmethod
    def step(self) -> dict:
        """One optimizer step; returns scalar metrics for logging."""

    @abstractmethod
    def eval(self) -> dict:
        """Periodic evaluation (samples + reward score)."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist LoRA + optimizer state for resume."""

    def load(self, path: str) -> None:
        """Load a checkpoint produced by ``save``."""
        raise NotImplementedError
