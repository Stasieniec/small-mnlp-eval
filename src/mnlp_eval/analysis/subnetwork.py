"""Subnetwork descriptors: which units a pruning run decided to keep.

A pruned checkpoint records what survived but not which positions it came
from, and the overlap half of RQ3 is entirely a question about positions. Two
50 percent subnetworks have the same shape whether they kept the same heads or
disjoint ones, so the selection has to be written down at pruning time. This
module defines that file and refuses to load a malformed one.

The format is deliberately dumb: JSON, integer indices, no framework types. A
teammate's pruning script writes it in a few lines and the analysis never
needs to import their code. See docs/subnetworks.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SUBNETWORK_SCHEMA_VERSION",
    "Component",
    "Subnetwork",
    "SubnetworkError",
    "load_subnetwork",
    "load_subnetworks",
]

SUBNETWORK_SCHEMA_VERSION = 1


class SubnetworkError(ValueError):
    """Raised when a subnetwork descriptor is malformed or self-inconsistent."""


@dataclass(frozen=True)
class Component:
    """One prunable unit type, and which of its positions were kept.

    ``kept`` is keyed by layer index as a string, matching JSON's object keys.
    A component that is not per-layer, whole layers being the obvious case,
    uses the single key ``"*"``.
    """

    name: str
    total: int
    kept: dict[str, tuple[int, ...]]

    @property
    def n_kept(self) -> int:
        return sum(len(indices) for indices in self.kept.values())

    @property
    def n_total(self) -> int:
        return self.total * len(self.kept)

    @property
    def sparsity(self) -> float:
        """Fraction of positions removed."""
        return 1.0 - (self.n_kept / self.n_total) if self.n_total else 0.0

    def positions(self) -> set[tuple[str, int]]:
        """Kept positions as (layer, index) pairs, pooled across layers."""
        return {(layer, index) for layer, indices in self.kept.items() for index in indices}


@dataclass(frozen=True)
class Subnetwork:
    """A named selection of kept units, produced by one pruning run."""

    name: str
    components: dict[str, Component]
    base_model: str = ""
    pruned_for: str = ""
    method: str = ""
    notes: str = ""
    source: Path | None = field(default=None, compare=False)

    @property
    def overall_sparsity(self) -> float:
        """Sparsity across every component, weighted by position count.

        A crude summary: it counts an attention head and an FFN channel as one
        unit each, and they are not the same size. Reported so two descriptors
        can be checked for being roughly comparable, not as a compression
        ratio. Use the bench stage for that.
        """
        kept = sum(component.n_kept for component in self.components.values())
        total = sum(component.n_total for component in self.components.values())
        return 1.0 - (kept / total) if total else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_model": self.base_model,
            "pruned_for": self.pruned_for,
            "method": self.method,
            "overall_unit_sparsity": round(self.overall_sparsity, 6),
            "components": {
                name: {
                    "layers": len(component.kept),
                    "total_per_layer": component.total,
                    "kept": component.n_kept,
                    "sparsity": round(component.sparsity, 6),
                }
                for name, component in sorted(self.components.items())
            },
        }


def load_subnetwork(path: str | Path) -> Subnetwork:
    """Read and validate one descriptor."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        msg = f"subnetwork descriptor not found: {resolved}"
        raise FileNotFoundError(msg)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{resolved}: malformed JSON: {exc}"
        raise SubnetworkError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"{resolved}: top level must be an object"
        raise SubnetworkError(msg)

    version = payload.get("schema_version", SUBNETWORK_SCHEMA_VERSION)
    if version != SUBNETWORK_SCHEMA_VERSION:
        msg = (
            f"{resolved}: schema_version {version!r} is not "
            f"{SUBNETWORK_SCHEMA_VERSION}, which this version of the analysis understands"
        )
        raise SubnetworkError(msg)

    raw_components = payload.get("components")
    if not isinstance(raw_components, dict) or not raw_components:
        msg = f"{resolved}: 'components' must be a non-empty object"
        raise SubnetworkError(msg)

    components = {
        name: _parse_component(name, value, resolved)
        for name, value in sorted(raw_components.items())
    }
    return Subnetwork(
        name=str(payload.get("name") or resolved.stem),
        components=components,
        base_model=str(payload.get("base_model", "")),
        pruned_for=str(payload.get("pruned_for", "")),
        method=str(payload.get("method", "")),
        notes=str(payload.get("notes", "")),
        source=resolved,
    )


def load_subnetworks(paths: list[str | Path]) -> list[Subnetwork]:
    """Read several descriptors, rejecting duplicate names.

    Two descriptors under one name would silently overwrite each other in the
    overlap matrix, which reads as a subnetwork having perfect overlap with
    itself under a different label.
    """
    loaded = [load_subnetwork(path) for path in paths]
    seen: dict[str, Path | None] = {}
    for subnetwork in loaded:
        if subnetwork.name in seen:
            msg = (
                f"two descriptors are both named {subnetwork.name!r}: "
                f"{seen[subnetwork.name]} and {subnetwork.source}"
            )
            raise SubnetworkError(msg)
        seen[subnetwork.name] = subnetwork.source
    return loaded


def _parse_component(name: str, value: Any, source: Path) -> Component:
    if not isinstance(value, dict):
        msg = f"{source}: component {name!r} must be an object with 'total' and 'kept'"
        raise SubnetworkError(msg)
    total = value.get("total")
    if not isinstance(total, int) or total <= 0:
        msg = f"{source}: component {name!r}: 'total' must be a positive integer"
        raise SubnetworkError(msg)

    raw_kept = value.get("kept")
    if isinstance(raw_kept, list):
        # A component that is not per-layer, such as whole layers.
        per_layer: dict[str, Any] = {"*": raw_kept}
    elif isinstance(raw_kept, dict):
        per_layer = raw_kept
    else:
        msg = (
            f"{source}: component {name!r}: 'kept' must be a list of indices, or an "
            "object mapping a layer index to its list of indices"
        )
        raise SubnetworkError(msg)
    if not per_layer:
        msg = f"{source}: component {name!r}: 'kept' is empty"
        raise SubnetworkError(msg)

    kept: dict[str, tuple[int, ...]] = {}
    for layer, indices in per_layer.items():
        if not isinstance(indices, list):
            msg = f"{source}: component {name!r} layer {layer!r}: expected a list of indices"
            raise SubnetworkError(msg)
        if not all(isinstance(index, int) for index in indices):
            msg = f"{source}: component {name!r} layer {layer!r}: indices must be integers"
            raise SubnetworkError(msg)
        unique = set(indices)
        if len(unique) != len(indices):
            msg = f"{source}: component {name!r} layer {layer!r}: repeated indices"
            raise SubnetworkError(msg)
        out_of_range = sorted(index for index in unique if not 0 <= index < total)
        if out_of_range:
            msg = (
                f"{source}: component {name!r} layer {layer!r}: indices {out_of_range[:5]} "
                f"are outside [0, {total})"
            )
            raise SubnetworkError(msg)
        if not indices:
            msg = (
                f"{source}: component {name!r} layer {layer!r} kept nothing. An empty "
                "layer is a removed layer, which belongs in a 'layers' component "
                "rather than as an empty entry here."
            )
            raise SubnetworkError(msg)
        kept[str(layer)] = tuple(sorted(unique))
    return Component(name=name, total=total, kept=kept)
