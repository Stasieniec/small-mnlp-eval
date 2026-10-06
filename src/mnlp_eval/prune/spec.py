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
from mnlp_eval.prune.budget import ALLOCATIONS, METHOD_ALLOCATIONS

__all__ = ["CALIBRATION_TEXTS", "METHODS", "METHOD_NAMES", "PruneSpec", "run_pruning"]

#: What of each calibration record goes through the model: the prompt alone,
#: or the prompt with its reference or dense-model continuation appended.
CALIBRATION_TEXTS = ("prompt", "prompt+target", "prompt+generated")

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
    #: ``prompt``, ``prompt+target`` or ``prompt+generated``.
    calibration_text: str = "prompt"
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
        known = (*ALLOCATIONS, *METHOD_ALLOCATIONS)
        if self.allocation not in known:
            msg = f"prune spec: allocation {self.allocation!r} is not one of {', '.join(known)}"
            raise ConfigError(msg)
        owner = METHOD_ALLOCATIONS.get(self.allocation)
        if owner is not None and self.method != owner:
            msg = (
                f"prune spec: allocation {self.allocation!r} is {METHOD_NAMES[owner]}'s own "
                f"structure search and works only with method {owner!r}, not {self.method!r}. "
                f"Use one of {', '.join(ALLOCATIONS)}."
            )
            raise ConfigError(msg)
        if not 0.0 <= self.sparsity < 1.0:
            msg = (
                f"prune spec: sparsity must be in [0, 1), got {self.sparsity}. "
                "It is the fraction removed, not the fraction kept."
            )
            raise ConfigError(msg)
        if self.calibration_text not in CALIBRATION_TEXTS:
            msg = (
                f"prune spec: calibration_text {self.calibration_text!r} is not one of "
                f"{', '.join(CALIBRATION_TEXTS)}"
            )
            raise ConfigError(msg)
        if self.batch_size < 1 or self.max_length < 1:
            msg = "prune spec: batch_size and max_length must both be at least 1"
            raise ConfigError(msg)

    @property
    def method_label(self) -> str:
        """How the report names the method; the default calibration text is implied."""
        label = f"{METHOD_NAMES[self.method]}, {self.allocation} budget"
        if self.calibration_text != "prompt":
            label += f", {self.calibration_text} calibration"
        return label

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

    prompts = load_calibration_prompts(
        spec.calibration,
        directions=spec.directions,
        with_target=spec.calibration_text == "prompt+target",
        with_generated=spec.calibration_text == "prompt+generated",
    )
    calibration_provenance: dict[str, Any] = {"text": spec.calibration_text}
    revision = None
    if spec.calibration_text == "prompt+generated":
        from mnlp_eval.data.generated_calibration import validate_cache

        cache = validate_cache(spec.calibration)
        identity = cache["identity"]["model"]
        if identity["model_name_or_path"] != spec.model_name_or_path:
            raise PruneError("generated calibration was produced by a different dense model")
        resolved = cache["identity"]["resolved_model"]
        if "inventory" in resolved:
            from mnlp_eval.config import ModelSpec
            from mnlp_eval.data.generated_calibration import _resolve_model

            _, current = _resolve_model(
                ModelSpec(
                    name="calibration-check",
                    loader="hf_causal",
                    model_name_or_path=spec.model_name_or_path,
                )
            )
            if current != resolved:
                raise PruneError("local dense checkpoint changed since calibration generation")
        revision = identity["revision"]
        calibration_provenance["cache_fingerprint"] = cache["fingerprint"]
        if any(len(sequence) > spec.max_length for sequence in prompts):
            raise PruneError("generated calibration exceeds max_length; use max_length: 768")

    seed_everything(spec.seed)
    target = out_dir / spec.name
    target.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_name_or_path, padding_side="left", revision=revision
    )
    if tokenizer.pad_token_id is None:
        # Batches are padded to a common length; without this it pads with None.
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_name_or_path,
        revision=revision,
        torch_dtype=resolve_dtype(spec.dtype),
        low_cpu_mem_usage=True,
    )
    model.eval()
    model = model.to(default_device())

    groups = describe_layers(model)

    def batches() -> list[dict[str, Any]]:
        return list(
            tokenized_batches(
                prompts,
                tokenizer,
                batch_size=spec.batch_size,
                max_length=spec.max_length,
                device=str(next(model.parameters()).device),
                append_eos=spec.calibration_text == "prompt+target",
            )
        )

    plan = _select(spec, model, groups, batches)
    parameter_sparsity = _parameter_sparsity(plan, groups)

    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    subnetwork_dir.mkdir(parents=True, exist_ok=True)
    subnetwork_path = subnetwork_dir / f"{spec.name}.json"
    provenance = {
        "name": spec.name,
        "base_model": spec.model_name_or_path,
        "method": METHOD_NAMES[spec.method],
        "pruned_for": spec.pruned_for,
        "notes": (
            f"{spec.allocation} budget over {len(prompts)} calibration segments, "
            f"{spec.calibration_text} text"
        ),
    }
    if "cache_fingerprint" in calibration_provenance:
        provenance["notes"] += f", cache {calibration_provenance['cache_fingerprint']}"
    provenance["notes"] += f"; removes {parameter_sparsity:.2%} of attention and MLP weights"
    # Beside the weights, so the checkpoint carries its own shapes, and under
    # subnetworks/, where the overlap analysis reads it.
    for path in (target / "subnetwork.json", subnetwork_path):
        descriptor = write_descriptor(path, plan, groups, **provenance)

    from mnlp_eval.artifacts import atomic_write_json

    atomic_write_json(
        target / "calibration_provenance.json",
        {
            **calibration_provenance,
            "path": spec.calibration,
            "directions": spec.directions,
            "segments": len(prompts),
            "max_length": spec.max_length,
        },
    )
    config_path = _write_model_config(spec, target, subnetwork_dir, config_dir)
    return {
        "name": spec.name,
        "method": METHOD_NAMES[spec.method],
        "checkpoint": str(target),
        "subnetwork": str(subnetwork_path),
        "model_config": str(config_path),
        "calibration_segments": len(prompts),
        "calibration_provenance": calibration_provenance,
        "requested_sparsity": spec.sparsity,
        "unit_sparsity": round(descriptor.overall_sparsity, 6),
        # What al-am's sparsity means, and what makes it comparable with the
        # unit budgets, under which it equals the unit sparsity up to rounding.
        "parameter_sparsity": round(parameter_sparsity, 6),
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
        if spec.allocation == flap.AL_AM:
            # Before the calibration pass, so a grouped-query model fails fast.
            flap.require_multi_head(groups)
        stats = collect_input_stats(model, iter(batches()))
        if spec.allocation == flap.AL_AM:
            plan = flap.al_am(model, stats, groups, sparsity=spec.sparsity)
        else:
            heads, channels = flap.score(model, stats, groups)
            plan = allocate(
                heads, channels, groups, sparsity=spec.sparsity, allocation=spec.allocation
            )
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


def _parameter_sparsity(plan: Any, groups: Any) -> float:
    """Fraction of the decoder's attention and MLP projection weights removed.

    Counted in units of the hidden size, which pruning never touches and which
    every projection has on one side: a query head holds ``head_dim`` of them
    in each of q_proj and o_proj, a key/value head ``head_dim`` in each of
    k_proj and v_proj, an FFN channel one in each of gate_proj, up_proj and
    down_proj. Biases, norms and embeddings are left out.
    """
    dense = kept = 0
    for group, layer in zip(groups, plan, strict=True):
        dense += 2 * group.head_dim * (group.num_heads + group.num_kv_heads)
        dense += 3 * group.intermediate
        kv_heads = len(group.kept_kv_heads(layer.heads))
        kept += 2 * group.head_dim * (len(layer.heads) + kv_heads) + 3 * len(layer.channels)
    return 1.0 - kept / dense


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
            "method": spec.method_label,
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
