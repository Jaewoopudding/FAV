"""Generative backbones with a unified training-time interface."""
from .base import Backbone, TorchBackbone, JAXBackbone, ForwardMode

__all__ = ["Backbone", "TorchBackbone", "JAXBackbone", "ForwardMode"]
