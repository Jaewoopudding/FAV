"""Backbone-agnostic alignment algorithms (FAV, DRaFT, Flow-GRPO, Adjoint Matching)."""
from .base import Trainer
from .fav import FAVTrainer
from .draft import DRaFTTrainer
from .flow_grpo import FlowGRPOTrainer
from .adjoint_matching import AdjointMatchingTrainer

__all__ = [
    "Trainer",
    "FAVTrainer",
    "DRaFTTrainer",
    "FlowGRPOTrainer",
    "AdjointMatchingTrainer",
]
