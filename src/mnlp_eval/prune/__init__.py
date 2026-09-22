"""Structured pruning: SlimGPT, FLAP and LLM-Pruner.

Selects which attention heads and FFN channels to keep, compacts the model
around that, and writes the descriptor the overlap analysis needs. See
docs/pruning.md.

Every torch and transformers import here is function-local, so the core package
stays installable in the COMET and MetricX environments.
"""

from __future__ import annotations

__all__ = ["PruneError"]


class PruneError(ValueError):
    """Raised when a model cannot be pruned as asked.

    Covers an unsupported architecture, a selection that the attention
    implementation would silently misroute, and a budget that cannot be met.
    """
