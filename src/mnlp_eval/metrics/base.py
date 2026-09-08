"""Metric result container and group availability probing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["METRIC_GROUPS", "MetricScore", "group_availability"]

#: Metric groups, in the order they should be reported. Each maps onto one of
#: the three environments described in docs/environments.md.
METRIC_GROUPS = ("surface", "neural", "metricx")


@dataclass
class MetricScore:
    """One metric's verdict on one direction.

    ``segment_scores`` is kept whenever the metric produces them, because
    bootstrap significance testing later would otherwise require re-running a
    multi-billion-parameter metric model over the whole test set.
    """

    name: str
    score: float | None
    signature: str | None = None
    segment_scores: list[float] | None = None
    higher_is_better: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_segments: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "score": self.score,
            "higher_is_better": self.higher_is_better,
        }
        if self.signature:
            payload["signature"] = self.signature
        if include_segments and self.segment_scores is not None:
            payload["segment_scores"] = self.segment_scores
        if self.extra:
            payload["extra"] = self.extra
        return payload


def group_availability() -> dict[str, tuple[bool, str]]:
    """Report which metric groups this interpreter can actually compute.

    Returned rather than raised, so a scoring job reports the groups it cannot
    serve as pending instead of failing the whole run. The dependency pins that
    make this necessary are documented in docs/environments.md.
    """
    status: dict[str, tuple[bool, str]] = {}

    try:
        import sacrebleu  # noqa: F401

        status["surface"] = (True, "")
    except ImportError:
        status["surface"] = (False, "sacrebleu not installed; pip install -e '.[surface]'")

    try:
        import comet  # noqa: F401

        status["neural"] = (True, "")
    except ImportError:
        status["neural"] = (
            False,
            "unbabel-comet not installed. It pins numpy<2 and torchmetrics<0.11, so it "
            "needs its own environment: see envs/comet-requirements.txt",
        )

    try:
        import metricx24  # noqa: F401

        status["metricx"] = (True, "")
    except ImportError:
        status["metricx"] = (
            False,
            "metricx not installed. It pins transformers==4.30.2, so it needs its own "
            "environment: see envs/metricx-requirements.txt",
        )

    return status
