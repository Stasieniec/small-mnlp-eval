"""The pruning criteria.

Each module supplies a score per group and, where the method has one, a
compensation step. The calibration pass, budget, surgery and descriptor are
shared. See docs/pruning.md.
"""

from __future__ import annotations

from typing import Any

from mnlp_eval.prune import PruneError
from mnlp_eval.prune.groups import LayerGroups

__all__ = ["pool_heads"]


def pool_heads(values: Any, group: LayerGroups) -> list[float]:
    """Sum each head's ``head_dim`` consecutive ``o_proj`` input channels.

    Both activation criteria hang their statistic off ``o_proj``'s input, whose
    channels are the heads' output dimensions in order.
    """
    if values.numel() != group.num_heads * group.head_dim:
        msg = (
            f"layer {group.index}: o_proj has {values.numel()} input channels, "
            f"expected {group.num_heads * group.head_dim}"
        )
        raise PruneError(msg)
    return [float(value) for value in values.view(group.num_heads, group.head_dim).sum(1)]
