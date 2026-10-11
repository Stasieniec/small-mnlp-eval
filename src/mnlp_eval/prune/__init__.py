"""Structured pruning of attention heads and FFN channels: SlimGPT, FLAP and LLM-Pruner.

torch and transformers are imported inside functions so the package imports in
the COMET and MetricX environments.
"""

from __future__ import annotations

__all__ = ["PruneError"]


class PruneError(ValueError):
    """Raised when a model cannot be pruned as asked."""
