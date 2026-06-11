"""Compatibility shim for IMM's NVIDIA-style persistence pickling.

The IMM checkpoint embeds class source code that re-imports top-level modules
like ``training``/``torch_utils`` on load, so we alias the vendored packages
into ``sys.modules``. Cannot coexist with StyleGAN-XL's shim in one process.
"""
from __future__ import annotations

import importlib
import sys

_VENDOR_ROOT = "main.backbones.imm._vendor"
_TOPLEVEL_PACKAGES = ("dnnlib", "torch_utils", "training")
_TOPLEVEL_SUBMODULES = (
    "dnnlib.util",
    "torch_utils.persistence",
    "torch_utils.misc",
    "torch_utils.training_stats",
    "torch_utils.distributed",
    "training.dit",
    "training.preconds",
    "training.unets",
    # training.encoders omitted: pulls in unvendored utils.torch_util
)

_INSTALLED = False


def install_aliases() -> None:
    """Register vendored IMM packages as top-level imports. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    for name in _TOPLEVEL_PACKAGES + _TOPLEVEL_SUBMODULES:
        if name in sys.modules:
            continue
        sys.modules[name] = importlib.import_module(f"{_VENDOR_ROOT}.{name}")
    _INSTALLED = True
