"""Dense-model continuations cached once and replayed for pruning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from mnlp_eval.artifacts import atomic_write_json, read_jsonl_dicts, write_jsonl
from mnlp_eval.config import ModelSpec, SuiteSpec
from mnlp_eval.languages import parse_direction
from mnlp_eval.models import build_translator
from mnlp_eval.postprocess import extract_hypothesis
from mnlp_eval.prompts import get_prompt
from mnlp_eval.seeding import seed_everything

MANIFEST = "calibration.json"
SCHEMA_VERSION = 1


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def prompt_id(direction: str, prompt: str) -> str:
    return fingerprint([direction, prompt])


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation() -> dict[str, Any]:
    # Invalidate when generation/token handling changes, including uncommitted edits.
    root = Path(__file__).parents[1]
    files = (
        "data/generated_calibration.py",
        "models/base.py",
        "models/hf_causal.py",
        "models/loading.py",
        "prompts.py",
        "languages.py",
        "postprocess.py",
        "seeding.py",
        "config.py",
    )
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "tokenizers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"files": {name: _file_hash(root / name) for name in files}, "packages": packages}


def _resolve_model(spec: ModelSpec) -> tuple[ModelSpec, dict[str, Any]]:
    """Pin Hub weights/tokenizer together before checking or creating a cache."""
    reference = str(spec.model_name_or_path)
    local = Path(reference).expanduser()
    if local.is_dir():
        # Local checkpoints have no immutable Hub revision. Inventory detects edits
        # without rereading GB of weights on every cache validation.
        inventory = {
            str(path.relative_to(local)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in sorted(local.rglob("*"))
            if path.is_file()
        }
        return spec, {"path": str(local.resolve()), "inventory": inventory}
    # Offline jobs resolve the prefetched snapshot; online jobs resolve the current
    # revision. Load that SHA afterwards so main cannot move between the two calls.
    import os

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            reference,
            revision=spec.revision,
            allow_patterns=["config.json"],
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "0").upper() in {"1", "TRUE", "YES"},
        )
    )
    revision = snapshot.name
    return replace(spec, revision=revision), {"repository": reference, "revision": revision}


def validate_cache(directory: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Check completeness and every direction's content before consuming a cache."""
    root = Path(directory)
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get("generated_schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{root}: not a supported generated calibration cache")
    if fingerprint(manifest["identity"]) != manifest.get("cache_key"):
        raise ValueError(f"{root}: invalid cache identity")
    if require_complete and not manifest.get("complete"):
        raise ValueError(
            f"{root}: generated calibration is incomplete; resume calibration-generate"
        )
    entries = manifest["directions"]
    if require_complete and set(entries) != set(manifest["identity"]["input"]):
        raise ValueError(f"{root}: generated cache is missing directions")
    if require_complete and {path.stem for path in root.glob("*.jsonl")} != set(entries):
        raise ValueError(f"{root}: unexpected or missing generated direction files")
    for direction, entry in entries.items():
        path = root / f"{direction}.jsonl"
        if not path.is_file() or _file_hash(path) != entry["sha256"]:
            raise ValueError(f"{path}: generated cache content changed or is missing")
        records = list(read_jsonl_dicts(path))
        if len(records) != entry["n_segments"]:
            raise ValueError(f"{path}: wrong segment count")
        for record in records:
            if record.get("direction") != direction or record.get("cache_id") != prompt_id(
                direction, record["prompt"]
            ):
                raise ValueError(f"{path}: prompt/source alignment is invalid")
            for field in ("input_token_ids", "generated_token_ids"):
                ids = record.get(field)
                if not isinstance(ids, list) or any(
                    type(token) is not int or token < 0 for token in ids
                ):
                    raise ValueError(f"{path}: invalid {field}")
            if not record["input_token_ids"] or not isinstance(record.get("generated"), str):
                raise ValueError(f"{path}: missing generated calibration sequence")
    expected = fingerprint({"cache_key": manifest["cache_key"], "directions": entries})
    if manifest.get("fingerprint") != expected:
        raise ValueError(f"{root}: invalid cache fingerprint")
    return dict(manifest)


def generate_calibration(
    calibration: str | Path,
    model: ModelSpec,
    suite: SuiteSpec,
    out: str | Path,
) -> dict[str, Any]:
    """Generate the union once, loading one dense model for all missing directions.

    The input directory is already the union (e.g. multi-10dir). Repeated prompts
    within a direction are deduplicated; conflicting reference records are refused.
    Completed directions survive interruptions. A mismatched cache requires a new
    output directory, rather than silently mixing translations from different runs.
    """
    from mnlp_eval.prune.collect import refuse_contaminated

    if model.loader != "hf_causal" or model.compression.family != "none" or model.adapter:
        raise ValueError("calibration-generate requires a dense hf_causal model without an adapter")
    if (
        model.quantization
        or model.tokenizer_name_or_path
        or get_prompt(model.prompt).requires_chat_template
    ):
        raise ValueError(
            "generated calibration currently requires a plain prompt and the model's tokenizer"
        )
    if suite.decode.do_sample or suite.decode.num_beams != 1:
        raise ValueError("calibration-generate currently requires greedy decoding (num_beams=1)")
    root, destination = Path(calibration), Path(out)
    if root.resolve() == destination.resolve():
        raise ValueError("generated cache must use a different directory from its input")
    refuse_contaminated(root)
    source_manifest = json.loads((root / MANIFEST).read_text())
    pool: dict[str, list[dict[str, Any]]] = {}
    template = get_prompt(model.prompt)
    files = sorted(root.glob("*.jsonl"))
    declared = source_manifest.get("directions")
    if declared is not None and set(declared) != {path.stem for path in files}:
        raise ValueError(f"{root}: calibration files do not match the input manifest")
    for path in files:
        direction = parse_direction(path.stem)
        if str(direction) not in suite.data.directions:
            raise ValueError(f"{direction}: direction absent from decoding suite")
        unique: dict[str, dict[str, Any]] = {}
        input_rows = list(read_jsonl_dicts(path))
        if declared is not None and len(input_rows) != declared[path.stem]["n_segments"]:
            raise ValueError(f"{path}: segment count does not match the input manifest")
        for row in input_rows:
            if row.get("direction") != str(direction) or row.get("prompt") != template.render(
                direction, row["source"]
            ):
                raise ValueError(f"{path}: prompt or direction does not match the dense model")
            key = prompt_id(str(direction), row["prompt"])
            if key in unique and unique[key]["target"] != row["target"]:
                raise ValueError(f"{path}: duplicate prompt has conflicting references")
            unique.setdefault(key, {**row, "cache_id": key})
        if not unique:
            raise ValueError(f"{path}: empty calibration direction")
        pool[str(direction)] = list(unique.values())
    if not pool:
        raise ValueError(f"{root}: no calibration records")
    pinned, resolved = _resolve_model(model)
    identity = {
        "input": {direction: fingerprint(rows) for direction, rows in pool.items()},
        "source_manifest": fingerprint(source_manifest),
        "model": pinned.identity(),
        "resolved_model": resolved,
        "decode": asdict(suite.decode),
        "implementation": _implementation(),
    }
    key = fingerprint(identity)
    if (destination / MANIFEST).exists():
        manifest = validate_cache(destination, require_complete=False)
        if manifest["cache_key"] != key:
            raise ValueError(
                f"{destination}: cache settings/input changed; choose a new --out directory"
            )
        if manifest["complete"]:
            return manifest
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"{destination}: nonempty directory has no generated cache manifest")
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generated_schema_version": SCHEMA_VERSION,
            "identity": identity,
            "cache_key": key,
            "complete": False,
            "directions": {},
            "contamination": source_manifest.get("contamination", {}),
            "source_calibration": str(root),
            "total_segments": sum(map(len, pool.values())),
        }

    def save() -> None:
        manifest["fingerprint"] = fingerprint(
            {"cache_key": key, "directions": manifest["directions"]}
        )
        atomic_write_json(destination / MANIFEST, manifest)

    save()
    seed_everything(suite.decode.seed)
    translator = build_translator(pinned)
    try:
        for name, rows in pool.items():
            if name in manifest["directions"]:
                continue
            direction = parse_direction(name)
            outputs = translator.translate(
                direction, [row["source"] for row in rows], suite.decode, capture_token_ids=True
            )
            if len(outputs) != len(rows):
                raise ValueError(f"{name}: model returned the wrong number of translations")
            records = []
            for row, output in zip(rows, outputs, strict=True):
                if output.input_token_ids is None or output.generated_token_ids is None:
                    raise ValueError("translator did not return calibration token IDs")
                parsed = extract_hypothesis(
                    output.raw_text, target_marker=template.target_marker(direction)
                )
                records.append(
                    {
                        **row,
                        "generated": output.raw_text,
                        "hypothesis": parsed.hypothesis,
                        "flags": list(parsed.flags),
                        "input_token_ids": output.input_token_ids,
                        "generated_token_ids": output.generated_token_ids,
                        "n_source_tokens": output.n_source_tokens,
                        "n_generated_tokens": output.n_generated_tokens,
                        "source_truncated": output.source_truncated,
                        "hit_token_budget": output.hit_token_budget,
                    }
                )
            path = destination / f"{name}.jsonl"
            write_jsonl(path, records)
            manifest["directions"][name] = {
                "sha256": _file_hash(path),
                "n_segments": len(records),
                "source_tokens": sum(row["n_source_tokens"] for row in records),
                "generated_tokens": sum(row["n_generated_tokens"] for row in records),
                "empty_outputs": sum(not row["generated"].strip() for row in records),
                "source_truncated": sum(row["source_truncated"] for row in records),
                "hit_token_budget": sum(row["hit_token_budget"] for row in records),
            }
            save()
        manifest["complete"] = True
        save()
    finally:
        translator.close()
    return validate_cache(destination)
