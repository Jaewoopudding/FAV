"""Drifting (JAX) backbone — sample-only via the unified interface.

Training is handled by the vendored JAX scripts, run via ``scripts/train_drifting.py``.
"""
from .backbone import DriftingBackbone

__all__ = ["DriftingBackbone"]
