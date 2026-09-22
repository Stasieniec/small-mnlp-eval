"""What a pruning run is, and running it.

Method, sparsity, allocation and calibration set are what tell two subnetworks
apart. All four reach the descriptor and the emitted model config, so a run is
interpretable without the config that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mnlp_eval.config import MULTI_DIRECTIONAL, ConfigError, _reject_unknown
from mnlp_eval.languages import parse_directions
from mnlp_eval.prune import PruneError
from mnlp_eval.prune.budget import ALLOCATIONS

__all__ = ["METHODS", "METHOD_NAMES", "PruneSpec", "run_pruning"]

#: The implemented criteria, by the name a config uses, and their display names.
METHOD_NAMES = {"flap": "FLAP", "llm-pruner": "LLM-Pruner", "slimgpt": "SlimGPT"}
METHODS = tuple(METHOD_NAMES)


@dataclass(frozen=True)
class PruneSpec:
    """One pruning run."""

    name: str
    method: str
    model_name_or_path: str
    #: A calibration set as ``mnlp-eval calibration`` writes it: one JSON Lines
    #: file per direction plus a manifest.
    calibration: str
    sparsity: float = 0.5
    allocation: str = "uniform"
    #: Which directions to use. Empty means all, a multi-directional subnetwork.
    directions: list[str] | None = None
    dtype: str = "bfloat16"
    batch_size: int = 4
    max_length: int = 512
    seed: int = 1234

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PruneSpec:
        payload = dict(data)
        _reject_unknown(cls, payload, "prune spec")
        for required in ("name", "method", "model_name_or_path", "calibration"):
            if not payload.get(required):
                msg = f"prune spec: {required!r} is required"
                raise ConfigError(msg)
        if payload.get("directions"):
            payload["directions"] = sorted(
                str(item) for item in parse_directions(payload["directions"])
            )
        spec = cls(**payload)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.method not in METHODS:
            msg = f"prune spec: method {self.method!r} is not one of {', '.join(METHODS)}"
            raise ConfigError(msg)
        if self.allocation not in ALLOCATIONS:
            msg = (
                f"prune spec: allocation {self.allocation!r} is not one of {', '.join(ALLOCATIONS)}"
            )
            raise ConfigError(msg)
        if not 0.0 <= self.sparsity < 1.0:
            msg = (
                f"prune spec: sparsity must be in [0, 1), got {self.sparsity}. "
                "It is the fraction removed, not the fraction kept."
            )
            raise ConfigError(msg)
        if self.batch_size < 1 or self.max_length < 1:
            msg = "prune spec: batch_size and max_length must both be at least 1"
            raise ConfigError(msg)

    @property
    def pruned_for(self) -> str:
        """What the report's transfer matrix keys this subnetwork on."""
        return ",".join(self.directions) if self.directions else MULTI_DIRECTIONAL


def run_pruning(
    spec: PruneSpec, out_dir: Path, *, subnetwork_dir: Path, config_dir: Path
) -> dict[str, Any]:
    """Prune a model and write the checkpoint, descriptor and model config.

    Returns a manifest describing what was produced.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mnlp_eval.models.loading import default_device, resolve_dtype
    from mnlp_eval.prune.collect import load_calibration_prompts, tokenized_batches
    from mnlp_eval.prune.compact import write_descriptor
    from mnlp_eval.prune.groups import describe_layers
    from mnlp_eval.seeding import seed_everything

    seed_everything(spec.seed)
    target = out_dir / spec.name
    target.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(spec.model_name_or_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        # Batches are padded to a common length; without this it pads with None.
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_name_or_path,
        torch_dtype=resolve_dtype(spec.dtype),
        low_cpu_mem_usage=True,
    )
    model.eval()
    model = model.to(default_device())

    prompts = load_calibration_prompts(spec.calibration, directions=spec.directions)
    groups = describe_layers(model)

    def batches() -> list[dict[str, Any]]:
        return list(
            tokenized_batches(
                prompts,
                tokenizer,
                batch_size=spec.batch_size,
                max_length=spec.max_length,
                device=str(next(model.parameters()).device),
            )
        )

    plan = _select(spec, model, groups, batches)

    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    subnetwork_dir.mkdir(parents=True, exist_ok=True)
    subnetwork_path = subnetwork_dir / f"{spec.name}.json"
    provenance = {
        "name": spec.name,
        "base_model": spec.model_name_or_path,
        "method": METHOD_NAMES[spec.method],
        "pruned_for": spec.pruned_for,
        "notes": f"{spec.allocation} budget over {len(prompts)} calibration segments",
    }
    # Beside the weights, so the checkpoint carries its own shapes, and under
    # subnetworks/, where the overlap analysis reads it.
    for path in (target / "subnetwork.json", subnetwork_path):
        descriptor = write_descriptor(path, plan, groups, **provenance)

    config_path = _write_model_config(spec, target, subnetwork_dir, config_dir)
    return {
        "name": spec.name,
        "method": METHOD_NAMES[spec.method],
        "checkpoint": str(target),
        "subnetwork": str(subnetwork_path),
        "model_config": str(config_path),
        "calibration_segments": len(prompts),
        "requested_sparsity": spec.sparsity,
        "unit_sparsity": round(descriptor.overall_sparsity, 6),
        "components": descriptor.summary()["components"],
    }


def _select(spec: PruneSpec, model: Any, groups: Any, batches: Any) -> Any:
    from mnlp_eval.prune.budget import allocate
    from mnlp_eval.prune.collect import collect_input_stats
    from mnlp_eval.prune.compact import compact_model
    from mnlp_eval.prune.methods import flap, llm_pruner, slimgpt

    if spec.method == "slimgpt":
        # SlimGPT compacts as it goes; see its module docstring.
        return slimgpt.prune(
            model, batches(), groups, sparsity=spec.sparsity, allocation=spec.allocation
        )

    if spec.method == "flap":
        stats = collect_input_stats(model, iter(batches()))
        heads, channels = flap.score(model, stats, groups)
        plan = allocate(heads, channels, groups, sparsity=spec.sparsity, allocation=spec.allocation)
        # Before compaction, while the dense channel indices still line up.
        flap.compensate(model, stats, groups, plan)
    elif spec.method == "llm-pruner":
        heads, channels = llm_pruner.score(model, batches(), groups)
        plan = allocate(heads, channels, groups, sparsity=spec.sparsity, allocation=spec.allocation)
    else:  # pragma: no cover - guarded by PruneSpec.validate
        msg = f"no implementation for method {spec.method!r}"
        raise PruneError(msg)

    compact_model(model, plan)
    return plan


def _write_model_config(
    spec: PruneSpec, checkpoint: Path, subnetwork_dir: Path, config_dir: Path
) -> Path:
    """Emit the model YAML that makes the checkpoint a system the harness runs.

    It goes beside the other model configs, not into the checkpoint, because
    ``extends`` resolves relative to the declaring file: a config sitting with
    the weights on scratch could never find ``alma-7b.yaml``.

    Always routes through the recipe loader. A uniform run would load without
    it, but one path keeps the sweep from being two artifacts that fail
    differently.
    """
    import yaml

    from mnlp_eval.artifacts import atomic_write_text

    payload = {
        "extends": "alma-7b.yaml",
        "name": spec.name,
        "baseline": "alma-7b",
        "loader": "custom",
        "entrypoint": "recipes.pruned:load",
        "kwargs": {"checkpoint": str(checkpoint)},
        "compression": {
            "family": "pruning",
            "method": f"{METHOD_NAMES[spec.method]}, {spec.allocation} budget",
            "nominal_sparsity": spec.sparsity,
            "pruned_for": spec.pruned_for,
            "subnetwork": str(subnetwork_dir / f"{spec.name}.json"),
            "calibration": spec.calibration,
        },
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{spec.name}.yaml"
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return path
