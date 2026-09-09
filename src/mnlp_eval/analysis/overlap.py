"""How much two subnetworks agree on which units to keep.

The number that matters is not the raw overlap. Two independently chosen
subnetworks that each keep half of every layer already share half of what they
keep, so a Jaccard index of 0.33 is what chance produces and reporting it as
evidence of shared structure would be wrong. Every figure here is therefore
paired with its chance expectation and the excess over it.

Comparisons are per component and per layer as well as pooled, because
"attention heads agree, FFN channels do not" is a finding and an average over
the two hides it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from mnlp_eval.analysis.subnetwork import Component, Subnetwork

__all__ = ["OverlapResult", "compare_subnetworks", "component_overlap", "overlap_report"]


@dataclass(frozen=True)
class OverlapResult:
    """Agreement between two selections of the same component."""

    component: str
    n_layers: int
    intersection: int
    union: int
    jaccard: float
    chance_jaccard: float
    #: Observed minus chance. Zero means the two selections agree no more than
    #: two independent selections of the same sizes would.
    excess: float
    overlap_coefficient: float
    per_layer_jaccard: list[float]

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "component": self.component,
            "n_layers": self.n_layers,
            "intersection": self.intersection,
            "union": self.union,
            "jaccard": round(self.jaccard, 6),
            "chance_jaccard": round(self.chance_jaccard, 6),
            "excess_over_chance": round(self.excess, 6),
            "overlap_coefficient": round(self.overlap_coefficient, 6),
        }
        if len(self.per_layer_jaccard) > 1:
            payload["per_layer_jaccard"] = {
                "min": round(min(self.per_layer_jaccard), 6),
                "max": round(max(self.per_layer_jaccard), 6),
                "mean": round(statistics.mean(self.per_layer_jaccard), 6),
            }
        return payload


def component_overlap(left: Component, right: Component) -> OverlapResult:
    """Compare two selections of the same component type.

    Positions are pooled across layers, so a head kept in layer 3 by one
    subnetwork and in layer 7 by the other does not count as agreement.
    """
    if left.total != right.total:
        msg = (
            f"component {left.name!r} has {left.total} positions per layer in one "
            f"descriptor and {right.total} in the other, so they describe different "
            "models and cannot be compared"
        )
        raise ValueError(msg)
    shared_layers = sorted(set(left.kept) & set(right.kept), key=_layer_sort_key)
    if not shared_layers:
        msg = f"component {left.name!r}: the two descriptors share no layer indices"
        raise ValueError(msg)

    left_positions = {(layer, index) for layer in shared_layers for index in left.kept[layer]}
    right_positions = {(layer, index) for layer in shared_layers for index in right.kept[layer]}
    intersection = len(left_positions & right_positions)
    union = len(left_positions | right_positions)

    per_layer: list[float] = []
    expected_intersection = 0.0
    for layer in shared_layers:
        a, b = set(left.kept[layer]), set(right.kept[layer])
        layer_union = len(a | b)
        per_layer.append(len(a & b) / layer_union if layer_union else 0.0)
        # Under independent uniform selection of |a| and |b| positions out of
        # `total`, the expected number in common is |a||b|/total.
        expected_intersection += len(a) * len(b) / left.total

    total_kept = len(left_positions) + len(right_positions)
    chance_jaccard = (
        expected_intersection / (total_kept - expected_intersection)
        if total_kept > expected_intersection
        else 0.0
    )
    smaller = min(len(left_positions), len(right_positions))
    return OverlapResult(
        component=left.name,
        n_layers=len(shared_layers),
        intersection=intersection,
        union=union,
        jaccard=intersection / union if union else 0.0,
        chance_jaccard=chance_jaccard,
        excess=(intersection / union if union else 0.0) - chance_jaccard,
        overlap_coefficient=intersection / smaller if smaller else 0.0,
        per_layer_jaccard=per_layer,
    )


def compare_subnetworks(
    subnetworks: list[Subnetwork],
) -> dict[str, Any]:
    """Compare every pair, per component.

    Returns a plain dictionary so the result can be written to JSON and read
    by a notebook without importing anything from this package.
    """
    if len(subnetworks) < 2:
        msg = "overlap analysis needs at least two subnetwork descriptors"
        raise ValueError(msg)

    components = sorted(set.intersection(*(set(item.components) for item in subnetworks)))
    skipped = sorted(set.union(*(set(item.components) for item in subnetworks)) - set(components))
    pairs: list[dict[str, Any]] = []
    problems: list[str] = []
    for index, left in enumerate(subnetworks):
        for right in subnetworks[index + 1 :]:
            results: dict[str, Any] = {}
            for name in components:
                try:
                    results[name] = component_overlap(
                        left.components[name], right.components[name]
                    ).as_dict()
                except ValueError as exc:
                    problems.append(f"{left.name} against {right.name}: {exc}")
            if results:
                pairs.append({"left": left.name, "right": right.name, "components": results})

    return {
        "subnetworks": [item.summary() for item in subnetworks],
        "components_compared": components,
        "components_skipped": skipped,
        "pairs": pairs,
        "problems": problems,
    }


def overlap_report(analysis: dict[str, Any]) -> str:
    """Render the comparison as markdown."""
    lines = [
        "# Subnetwork overlap",
        "",
        "## Subnetworks",
        "",
        "| Name | Selected for | Unit sparsity | Components |",
        "| --- | --- | --- | --- |",
    ]
    for item in analysis["subnetworks"]:
        components = ", ".join(
            f"{name} {payload['sparsity']:.1%} sparse"
            for name, payload in item["components"].items()
        )
        lines.append(
            f"| {item['name']} | {item['pruned_for'] or 'multi'} | "
            f"{item['overall_unit_sparsity']:.1%} | {components} |"
        )

    for component in analysis["components_compared"]:
        lines.extend(
            [
                "",
                f"## {component}",
                "",
                "| Pair | Jaccard | Chance | Excess | Overlap coefficient |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for pair in analysis["pairs"]:
            payload = pair["components"].get(component)
            if payload is None:
                continue
            lines.append(
                f"| {pair['left']} vs {pair['right']} | {payload['jaccard']:.3f} | "
                f"{payload['chance_jaccard']:.3f} | {payload['excess_over_chance']:+.3f} | "
                f"{payload['overlap_coefficient']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Reading this",
            "",
            "- Jaccard is the shared kept positions over the positions either kept.",
            "- Chance is what two independent selections of the same sizes would give: "
            "the expected intersection is the product of the two kept counts over the "
            "number of positions, computed per layer and summed. The Jaccard of an "
            "expected intersection is not quite the expectation of the Jaccard, but the "
            "difference is negligible at these layer widths.",
            "- Excess is the figure to quote. A value near zero means the two pruning "
            "runs agreed no more than chance, whatever the raw Jaccard says.",
            "- The overlap coefficient divides by the smaller selection instead, which "
            "is the right measure when the two subnetworks have different sparsities: "
            "it asks whether the sparser one is a subset of the denser one.",
        ]
    )
    if analysis["components_skipped"]:
        lines.append(
            f"- Skipped, present in only some descriptors: "
            f"{', '.join(analysis['components_skipped'])}."
        )
    for problem in analysis["problems"]:
        lines.append(f"- Not compared: {problem}")
    return "\n".join(lines) + "\n"


def _layer_sort_key(layer: str) -> tuple[int, int | str]:
    """Sort numeric layer keys numerically and anything else after them."""
    return (0, int(layer)) if layer.lstrip("-").isdigit() else (1, layer)
