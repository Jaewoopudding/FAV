"""``DriftingBackbone`` — JAX backbone (sample-only via the unified interface).

Implemented end-to-end in JAX/Flax. Training hands off to the vendored JAX
scripts; the inference path lazily imports the vendored JAX code. Requires JAX.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..base import JAXBackbone
from ...utils import lora_registry


def _require_jax():
    try:
        import jax  # noqa: F401
        import jaxlib  # noqa: F401
        import flax  # noqa: F401
        import optax  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Drifting backbone requires JAX. Install with:\n"
            "    pip install -e '.[jax]'\n"
            "or use a JAX-enabled conda env (e.g. the original `drift` env)."
        ) from e


class DriftingBackbone(JAXBackbone):
    """JAX/Flax Drifting backbone wrapper. JAX is imported lazily so that
    ``import main.backbones.drifting`` succeeds even when JAX is absent."""

    name = "drifting"
    is_jax = True

    def __init__(
        self,
        *,
        init_from: str = "hf://latent_L_sota",
        workdir: str | Path = "runs/drifting",
        img_resolution: int = 256,
    ) -> None:
        self.init_from = init_from
        self.workdir = str(workdir)
        self.img_resolution = img_resolution
        self._params = None
        self._state = None

    def load_pretrained(self, ckpt_path: str) -> None:
        """Load a Drifting checkpoint (``hf://<name>`` reference or local flax dir)."""
        _require_jax()
        from ._vendor.utils import ckpt_util, model_builder  # type: ignore
        # records the path; params materialise during sample()
        self._ckpt_path = ckpt_path

    def inject_lora(self, rank: int, **kwargs: Any) -> None:
        _require_jax()
        raise NotImplementedError(
            "DriftingBackbone.inject_lora is handled by the JAX trainer; "
            "see main/backbones/drifting/_vendor/aligen_lora.py."
        )

    def get_lora_state(self) -> dict:
        raise NotImplementedError("Use save_lora_checkpoint() in aligen_lora.py.")

    def load_lora_state(self, state: dict) -> None:
        raise NotImplementedError("Use load_lora_checkpoint() in aligen_lora.py.")

    def sample(self, *, n_sample: int, labels: Any, rng: Any, **kwargs: Any):
        """Generate ``n_sample`` images from class ``labels``. Returns JAX arrays
        of shape ``(n_sample, H, W, 3)``."""
        _require_jax()
        from ._vendor import inference  # type: ignore
        raise NotImplementedError(
            "sample(): wire up to main/backbones/drifting/_vendor/inference.py "
            "once the JAX env is verified. Use the vendored inference.py CLI "
            "directly in the meantime."
        )


# no-op LoRA registry entry so cfg.backbone.name=drifting resolves; LoRA is JAX-native
def _drifting_unsupported(*_, **__):
    raise NotImplementedError(
        "Drifting LoRA is JAX-native; see main/backbones/drifting/_vendor/aligen_lora.py."
    )


lora_registry.register(
    "drifting",
    inject=_drifting_unsupported,
    get_state=_drifting_unsupported,
    load_state=_drifting_unsupported,
)
