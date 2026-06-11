"""Compatibility shim for StyleGAN-XL's NVIDIA-style persistence pickling.

Registers vendored ``dnnlib`` / ``torch_utils`` / ``training`` as top-level
imports so the pickled source can re-import them. Conflicts with IMM's shim:
only one backbone can install aliases per process.
"""
from __future__ import annotations

import importlib
import os
import sys


def _ensure_ninja_on_path() -> None:
    """Add the env's bin/ to PATH so the ``ninja`` binary (used for CUDA JIT) is found."""
    env_bin = os.path.dirname(sys.executable)
    current = os.environ.get("PATH", "")
    if env_bin and env_bin not in current.split(":"):
        os.environ["PATH"] = f"{env_bin}:{current}"


def _install_timm_legacy_shims() -> None:
    """Alias modern ``timm.layers`` under the legacy ``timm.models.layers`` path
    that StyleGAN-XL's pickled source expects. No-op if timm isn't present."""
    try:
        import timm                              # noqa: F401
        import timm.layers as new_layers
    except ImportError:
        return
    sys.modules.setdefault("timm.models.layers", new_layers)
    for sub in (
        "conv2d_same", "mlp", "helpers", "weight_init", "drop", "patch_embed",
        "create_act", "create_norm_act", "trace_utils", "pos_embed", "format",
        "create_conv2d",
    ):
        try:
            mod = importlib.import_module(f"timm.layers.{sub}")
        except ImportError:
            continue
        sys.modules.setdefault(f"timm.models.layers.{sub}", mod)
    # public -> private renames between timm 0.6 and 1.x
    for old, new in (
        ("timm.models.efficientnet_blocks", "timm.models._efficientnet_blocks"),
        ("timm.models.efficientnet_builder", "timm.models._efficientnet_builder"),
    ):
        try:
            sys.modules.setdefault(old, importlib.import_module(new))
        except ImportError:
            continue


_VENDOR_ROOT = "main.backbones.stylegan_xl._vendor"
_TOPLEVEL_PACKAGES = ("dnnlib", "torch_utils", "training", "pg_modules", "feature_networks")
_TOPLEVEL_SUBMODULES = (
    "dnnlib.util",
    "torch_utils.persistence",
    "torch_utils.misc",
    "torch_utils.training_stats",
    "torch_utils.custom_ops",
    "torch_utils.gen_utils",
    "torch_utils.utils_spectrum",
    "training.networks_stylegan2",
    "training.networks_stylegan3",
    "training.networks_stylegan3_resetting",
    "training.networks_fastgan",
    "pg_modules.blocks",
    "pg_modules.discriminator",
    "pg_modules.projector",
)

_INSTALLED = False


def install_aliases() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _ensure_ninja_on_path()
    _install_timm_legacy_shims()
    for name in _TOPLEVEL_PACKAGES + _TOPLEVEL_SUBMODULES:
        if name in sys.modules:
            continue
        try:
            sys.modules[name] = importlib.import_module(f"{_VENDOR_ROOT}.{name}")
        except Exception:
            # optional submodules (custom_ops, pg_modules.*) drag in deps like
            # ninja; skip here and let them load lazily if actually exercised
            pass
    if "lora_utils" not in sys.modules:
        try:
            sys.modules["lora_utils"] = importlib.import_module(f"{_VENDOR_ROOT}.lora_utils")
        except Exception:
            pass
    if "legacy" not in sys.modules:
        try:
            sys.modules["legacy"] = importlib.import_module(f"{_VENDOR_ROOT}.legacy")
        except Exception:
            pass
    _INSTALLED = True
