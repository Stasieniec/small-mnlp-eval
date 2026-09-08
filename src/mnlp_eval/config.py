"""Typed configuration objects, YAML loading, and run identity.

Every stage of the pipeline is driven by these objects, and a run's identity is
a hash of the semantic subset of them. That gives two guarantees the project
depends on:

* jobs are idempotent, so a resubmitted Slurm array does not duplicate work;
* two systems cannot be compared unless they were evaluated under identical
  data and decoding settings, because differing settings produce different run
  identities and the report stage refuses to mix them.

Unknown keys are rejected rather than ignored. A typo in a teammate's YAML
should fail immediately, not silently fall back to a default and quietly
invalidate a comparison.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from mnlp_eval.languages import Direction, parse_directions
from mnlp_eval.prompts import get_prompt, prompt_hash

__all__ = [
    "AdapterSpec",
    "BenchSpec",
    "DataSpec",
    "DecodeSpec",
    "MetricsSpec",
    "ModelSpec",
    "QuantizationSpec",
    "RunConfig",
    "SuiteSpec",
    "canonical_json",
    "load_yaml_config",
]

T = TypeVar("T")

DEFAULT_DATASET = "haoranxu/WMT22-Test"

_VALID_DTYPES = frozenset({"float32", "float16", "bfloat16", "auto"})
_VALID_QUANT_METHODS = frozenset(
    {"bitsandbytes", "gptq", "awq", "compressed_tensors", "hqq", "torchao"}
)


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or self-inconsistent."""


def _reject_unknown(cls: type, data: dict[str, Any], context: str) -> None:
    known = {field_info.name for field_info in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        msg = (
            f"{context}: unknown key(s) {', '.join(unknown)}. "
            f"Valid keys are: {', '.join(sorted(known))}"
        )
        raise ConfigError(msg)


# --------------------------------------------------------------------------
# Model description
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantizationSpec:
    """How to quantize at load time, or how a saved checkpoint is quantized."""

    method: str
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantizationSpec:
        payload = dict(data)
        method = payload.pop("method", None)
        if not method:
            msg = "quantization: 'method' is required"
            raise ConfigError(msg)
        if method not in _VALID_QUANT_METHODS:
            msg = (
                f"quantization: unsupported method {method!r}; "
                f"supported methods are {', '.join(sorted(_VALID_QUANT_METHODS))}"
            )
            raise ConfigError(msg)
        options = payload.pop("options", None)
        # Allow either a nested "options" mapping or inline keys, since inline
        # reads better for the common bitsandbytes case.
        merged = dict(options or {})
        merged.update(payload)
        return cls(method=method, options=merged)


@dataclass(frozen=True)
class AdapterSpec:
    """A PEFT adapter applied on top of a base checkpoint."""

    path: str
    revision: str | None = None
    merge: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdapterSpec:
        _reject_unknown(cls, data, "adapter")
        if "path" not in data:
            msg = "adapter: 'path' is required"
            raise ConfigError(msg)
        return cls(**data)


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build one translation system.

    ``name`` and ``notes`` are labels and take no part in run identity, so
    renaming a system does not orphan its existing runs.
    """

    name: str
    loader: str
    prompt: str = "alma"
    model_name_or_path: str | None = None
    revision: str | None = None
    tokenizer_name_or_path: str | None = None
    dtype: str = "bfloat16"
    device_map: str | None = None
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    quantization: QuantizationSpec | None = None
    adapter: AdapterSpec | None = None
    entrypoint: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    baseline: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSpec:
        payload = dict(data)
        _reject_unknown(cls, payload, "model spec")
        for key in ("name", "loader"):
            if not payload.get(key):
                msg = f"model spec: {key!r} is required"
                raise ConfigError(msg)
        if (raw_quant := payload.get("quantization")) is not None:
            payload["quantization"] = QuantizationSpec.from_dict(raw_quant)
        if (raw_adapter := payload.get("adapter")) is not None:
            payload["adapter"] = AdapterSpec.from_dict(raw_adapter)
        spec = cls(**payload)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.dtype not in _VALID_DTYPES:
            msg = (
                f"model {self.name}: dtype {self.dtype!r} is not one of "
                f"{', '.join(sorted(_VALID_DTYPES))}"
            )
            raise ConfigError(msg)
        # Fails fast on an unknown prompt name rather than at generation time,
        # an hour into a Slurm job.
        try:
            get_prompt(self.prompt)
        except KeyError as exc:
            msg = f"model {self.name}: {exc.args[0]}"
            raise ConfigError(msg) from None
        if self.loader == "custom":
            if not self.entrypoint:
                msg = f"model {self.name}: loader 'custom' requires 'entrypoint'"
                raise ConfigError(msg)
            if ":" not in self.entrypoint:
                msg = (
                    f"model {self.name}: entrypoint {self.entrypoint!r} must have the form "
                    "'package.module:function'"
                )
                raise ConfigError(msg)
        elif not self.model_name_or_path:
            msg = f"model {self.name}: 'model_name_or_path' is required for loader {self.loader!r}"
            raise ConfigError(msg)

    @property
    def prompt_fingerprint(self) -> str:
        return prompt_hash(get_prompt(self.prompt))

    def identity(self) -> dict[str, Any]:
        """Return the fields that make this model a distinct system."""
        payload: dict[str, Any] = dict(_plain(self))
        for label in ("name", "notes", "device_map", "baseline"):
            payload.pop(label, None)
        payload["prompt_fingerprint"] = self.prompt_fingerprint
        return payload


# --------------------------------------------------------------------------
# Data and decoding
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSpec:
    """Which test set, which directions, and how much of it."""

    directions: list[str]
    dataset: str = DEFAULT_DATASET
    split: str = "test"
    limit: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataSpec:
        payload = dict(data)
        _reject_unknown(cls, payload, "data spec")
        raw = payload.get("directions")
        if not raw:
            msg = "data spec: 'directions' is required"
            raise ConfigError(msg)
        payload["directions"] = [str(d) for d in parse_directions(raw)]
        spec = cls(**payload)
        if spec.limit is not None and spec.limit <= 0:
            msg = f"data spec: limit must be positive, got {spec.limit}"
            raise ConfigError(msg)
        return spec

    @property
    def parsed_directions(self) -> list[Direction]:
        return parse_directions(self.directions)


@dataclass(frozen=True)
class DecodeSpec:
    """Decoding settings, shared by every system in a comparison.

    ``do_sample`` defaults to false and must be enabled explicitly. Several
    ALMA-family checkpoints ship a ``generation_config.json`` inherited from
    Llama 2 that sets ``do_sample``, ``temperature`` and ``top_p``, so the
    generator always passes these values explicitly rather than inheriting
    them. ALMA's own README quick-start samples, while its evaluation scripts
    use pure beam search; the scripts are what produced the published numbers.
    """

    num_beams: int = 5
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_new_tokens: int = 256
    max_source_length: int = 256
    batch_size: int = 4
    seed: int = 42
    length_penalty: float = 1.0
    early_stopping: bool = False
    # Pinned to neutral rather than inherited. Qwen2.5 ships
    # repetition_penalty 1.1 in its generation config, which suppresses the
    # degenerate-repetition collapse mode this framework measures as a
    # behavioural metric. Inheriting it silently would mean the checkpoint
    # decides how visible its own failure mode is.
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    # Length-bucketed batching is a large speed win but changes how much
    # padding each sequence sees, which under beam search can perturb output.
    # It is part of run identity so a bucketed run is never silently compared
    # against an unbucketed one. Set false for the ALMA reproduction suite.
    sort_by_length: bool = True
    respect_alma_source_length_override: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecodeSpec:
        payload = dict(data)
        _reject_unknown(cls, payload, "decode spec")
        spec = cls(**payload)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.repetition_penalty <= 0:
            msg = f"decode spec: repetition_penalty must be positive, got {self.repetition_penalty}"
            raise ConfigError(msg)
        if self.no_repeat_ngram_size < 0:
            msg = "decode spec: no_repeat_ngram_size cannot be negative"
            raise ConfigError(msg)
        if self.num_beams < 1:
            msg = f"decode spec: num_beams must be at least 1, got {self.num_beams}"
            raise ConfigError(msg)
        for label in ("max_new_tokens", "max_source_length", "batch_size"):
            if getattr(self, label) < 1:
                msg = f"decode spec: {label} must be at least 1"
                raise ConfigError(msg)
        if not self.do_sample:
            offenders = [
                label
                for label in ("temperature", "top_p", "top_k")
                if getattr(self, label) is not None
            ]
            if offenders:
                msg = (
                    f"decode spec: {', '.join(offenders)} set while do_sample is false. "
                    "Remove them, or set do_sample: true and accept that results are "
                    "no longer deterministic."
                )
                raise ConfigError(msg)

    @property
    def is_deterministic(self) -> bool:
        return not self.do_sample

    def describe(self) -> str:
        """Return a short human-readable label, for tables and log lines."""
        if self.do_sample:
            mode = "sample"
        elif self.num_beams == 1:
            mode = "greedy"
        else:
            mode = f"beam{self.num_beams}"
        return f"{mode}, max_new_tokens={self.max_new_tokens}"


@dataclass(frozen=True)
class SuiteSpec:
    """A named pairing of a test set with decoding settings."""

    name: str
    data: DataSpec
    decode: DecodeSpec = field(default_factory=DecodeSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuiteSpec:
        payload = dict(data)
        _reject_unknown(cls, payload, "suite")
        if not payload.get("name"):
            msg = "suite: 'name' is required"
            raise ConfigError(msg)
        payload["data"] = DataSpec.from_dict(payload.get("data") or {})
        payload["decode"] = DecodeSpec.from_dict(payload.get("decode") or {})
        return cls(**payload)


# --------------------------------------------------------------------------
# Metrics and efficiency
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricsSpec:
    """Which metrics to compute, and with which checkpoints.

    Groups map onto the three incompatible environments described in
    docs/environments.md: ``surface`` runs anywhere, ``neural`` needs the COMET
    environment, ``metricx`` needs the MetricX environment.
    """

    groups: list[str] = field(default_factory=lambda: ["surface", "neural"])
    comet_models: list[str] = field(default_factory=lambda: ["Unbabel/wmt22-comet-da"])
    metricx_model: str = "google/metricx-24-hybrid-large-v2p6-bfloat16"
    metricx_tokenizer: str = "google/mt5-xl"
    comet_batch_size: int = 32
    comet_gpus: int = 1
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 12345
    lid_backend: str = "auto"
    compute_ter: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricsSpec:
        payload = dict(data)
        _reject_unknown(cls, payload, "metrics spec")
        spec = cls(**payload)
        unknown = sorted(set(spec.groups) - {"surface", "neural", "metricx"})
        if unknown:
            msg = f"metrics spec: unknown group(s) {', '.join(unknown)}"
            raise ConfigError(msg)
        if spec.bootstrap_seed == 0:
            # sacreBLEU treats a zero seed as unseeded, so the surface
            # p-values would silently stop being reproducible.
            msg = "metrics spec: bootstrap_seed must be non-zero (sacreBLEU treats 0 as unseeded)"
            raise ConfigError(msg)
        if spec.bootstrap_samples < 1:
            msg = "metrics spec: bootstrap_samples must be at least 1"
            raise ConfigError(msg)
        return spec


@dataclass(frozen=True)
class BenchSpec:
    """The efficiency measurement protocol.

    Fixed subset, fixed token budget, warmup discarded, repeats taken. Batch
    size 1 and a batched setting are both measured, because compression methods
    trade off differently in the memory-bound and compute-bound regimes.
    """

    subset_size: int = 128
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8])
    warmup_batches: int = 2
    repeats: int = 3
    max_new_tokens: int = 128
    direction: str = "de-en"
    # Force min_new_tokens == max_new_tokens so latency does not reward a model
    # for stopping early. See mnlp_eval.bench.efficiency for the reasoning.
    force_fixed_length: bool = True
    measure_power: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchSpec:
        payload = dict(data)
        _reject_unknown(cls, payload, "bench spec")
        spec = cls(**payload)
        if spec.repeats < 1:
            msg = "bench spec: repeats must be at least 1"
            raise ConfigError(msg)
        if not spec.batch_sizes or any(b < 1 for b in spec.batch_sizes):
            msg = "bench spec: batch_sizes must be a non-empty list of positive integers"
            raise ConfigError(msg)
        return spec


# --------------------------------------------------------------------------
# Run configuration and identity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """A model plus a suite, which together identify a run."""

    model: ModelSpec
    suite: SuiteSpec
    output_root: Path = Path("runs")

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any], output_root: Path) -> RunConfig:
        """Rebuild a run configuration from a manifest.

        Lets the bench and score stages act on a run directory alone, which is
        what a Slurm job chain has to work with.
        """
        config = cls(
            model=ModelSpec.from_dict(manifest["model"]),
            suite=SuiteSpec.from_dict(manifest["suite"]),
            output_root=output_root,
        )
        if config.run_id != manifest.get("run_id"):
            msg = (
                f"manifest run_id {manifest.get('run_id')!r} does not match the identity "
                f"of its own contents ({config.run_id!r}). The manifest was edited by hand, "
                "or was written by an incompatible version."
            )
            raise ConfigError(msg)
        return config

    def identity(self) -> dict[str, Any]:
        return {
            "model": self.model.identity(),
            "data": _plain(self.suite.data),
            "decode": _plain(self.suite.decode),
        }

    @property
    def run_id(self) -> str:
        return hashlib.sha256(canonical_json(self.identity()).encode()).hexdigest()[:12]

    @property
    def slug(self) -> str:
        return f"{self.model.name}__{self.suite.name}__{self.run_id}"

    @property
    def directory(self) -> Path:
        return self.output_root / self.slug


def canonical_json(payload: Any) -> str:
    """Serialise to JSON deterministically, for hashing and for manifests."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _plain(value: Any) -> Any:
    """Convert dataclasses and paths into JSON-safe plain data."""
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


# --------------------------------------------------------------------------
# YAML loading with single-level inheritance
# --------------------------------------------------------------------------


def load_yaml_config(path: str | Path, *, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML file, resolving an optional ``extends`` chain.

    ``extends`` holds a path relative to the file that declares it. Values in
    the child override the parent, and nested mappings are merged rather than
    replaced. This keeps a compression sweep readable: one base model spec plus
    a short file per variant.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved in _seen:
        chain = " -> ".join(str(item) for item in (*_seen, resolved))
        msg = f"circular 'extends' chain: {chain}"
        raise ConfigError(msg)
    if not resolved.is_file():
        msg = f"config file not found: {resolved}"
        raise ConfigError(msg)

    with resolved.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        msg = f"{resolved}: top level must be a mapping, got {type(loaded).__name__}"
        raise ConfigError(msg)

    parent_ref = loaded.pop("extends", None)
    if parent_ref is None:
        return loaded
    parent = load_yaml_config(resolved.parent / str(parent_ref), _seen=(*_seen, resolved))
    return _deep_merge(parent, loaded)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
