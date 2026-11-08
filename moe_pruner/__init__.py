"""Prune MoE experts using the routing stats `moe_profiler` records.

`scoring` scores experts over runs, `budget` picks how many each layer keeps, `plan`
records the decision, `masking` previews it on a loaded model, `surgery` writes the
pruned checkpoint, `resize` loads it, and `verify` checks it.
"""
from .budget import Allocation, allocate, layer_weights, uniform_curve
from .masking import attach, pruned
from .plan import PruningPlan
from .resize import load_pruned_model, resize_experts
from .scoring import PRESETS, Criterion, ExpertScores
from .surgery import prune_checkpoint

__all__ = ["ExpertScores", "Criterion", "PRESETS", "Allocation", "allocate", "layer_weights",
           "uniform_curve", "PruningPlan", "attach", "pruned",
           "prune_checkpoint", "resize_experts", "load_pruned_model"]
