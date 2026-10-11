"""Download datasets, models and metric checkpoints into ``HF_HOME`` before a run.

Meant for a login node; batch jobs run with ``HF_HUB_OFFLINE=1``. Each
environment prefetches the metric models it can load and reports the rest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mnlp_eval.config import MetricsSpec, ModelSpec, SuiteSpec

__all__ = ["prefetch"]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def prefetch(
    models: list[ModelSpec],
    suites: list[SuiteSpec],
    metrics: MetricsSpec | None = None,
) -> dict[str, Any]:
    """Populate the local cache. Returns a per-item status report."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        _log(
            "HF_HUB_OFFLINE=1 is set, so nothing can be downloaded. Unset it on the "
            "login node and set it only inside batch jobs."
        )
    home = os.environ.get("HF_HOME")
    _log(f"HF_HOME={home or '(default, ~/.cache/huggingface)'}")

    report: dict[str, Any] = {"datasets": {}, "models": {}, "metrics": {}}

    for suite in suites:
        for direction in suite.data.directions:
            key = f"{suite.data.dataset}:{direction}"
            if key in report["datasets"]:
                continue
            report["datasets"][key] = _fetch_dataset(
                suite.data.dataset, direction, suite.data.split
            )

    for model in models:
        for label, reference, revision in _model_references(model):
            key = f"{model.name}:{label}"
            report["models"][key] = _fetch_repo(reference, revision)

    if metrics is not None:
        report["metrics"] = _fetch_metrics(metrics)

    warnings = [
        f"{key}: {value['warning']}"
        for section in report.values()
        for key, value in section.items()
        if value.get("warning")
    ]
    if warnings:
        _log(f"{len(warnings)} item(s) could not be prefetched but are not errors:")
        for warning in warnings:
            _log(f"  {warning}")

    failures = [
        f"{key}: {value['reason']}"
        for section in report.values()
        for key, value in section.items()
        if not value.get("ok")
    ]
    if failures:
        _log(f"{len(failures)} item(s) could not be prefetched:")
        for failure in failures:
            _log(f"  {failure}")
    elif warnings:
        _log("everything downloadable is in the local cache")
    else:
        _log("all items present in the local cache")
    return report


def _model_references(model: ModelSpec) -> list[tuple[str, str, str | None]]:
    references: list[tuple[str, str, str | None]] = []
    if model.model_name_or_path:
        references.append(("model", model.model_name_or_path, model.revision))
    if model.tokenizer_name_or_path:
        references.append(("tokenizer", model.tokenizer_name_or_path, model.revision))
    if model.adapter is not None:
        references.append(("adapter", model.adapter.path, model.adapter.revision))
    return references


def _fetch_dataset(dataset: str, direction: str, split: str) -> dict[str, Any]:
    if dataset.startswith("local:"):
        return {"ok": True, "note": "local file, nothing to download"}
    try:
        from datasets import load_dataset

        loaded = load_dataset(dataset, direction, split=split)
        _log(f"  dataset {dataset} [{direction}]: {len(loaded)} rows")
        return {"ok": True, "rows": len(loaded)}
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def _looks_like_a_filesystem_path(reference: str) -> bool:
    """Whether a model reference is a filesystem path (Hub ids are ``owner/name``)."""
    return reference.startswith(("/", "./", "../", "~"))


def _fetch_repo(
    reference: str, revision: str | None, allow_patterns: list[str] | None = None
) -> dict[str, Any]:
    if Path(reference).expanduser().is_dir():
        return {"ok": True, "note": "local directory"}
    if _looks_like_a_filesystem_path(reference):
        # A warning only: pruned configs point at checkpoints that may not exist yet.
        return {
            "ok": True,
            "note": "local path",
            "warning": (
                f"{reference} is not a Hub repository and does not exist yet. "
                "The job that uses it will fail unless the checkpoint is in place "
                "by then."
            ),
        }
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(reference, revision=revision, allow_patterns=allow_patterns)
        _log(f"  model {reference}: cached")
        return {"ok": True, "path": path}
    except Exception as exc:
        hint = ""
        if "gated" in str(exc).lower() or "401" in str(exc):
            hint = (
                " This repository is gated: accept its licence on the Hub and export "
                "HF_TOKEN before prefetching."
            )
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}{hint}"}


def _comet_encoder(checkpoint: str | Path) -> str | None:
    """The Hub name of the encoder a COMET checkpoint builds on, from hparams.yaml."""
    import yaml

    hparams = Path(checkpoint).parents[1] / "hparams.yaml"
    if not hparams.is_file():
        return None
    try:
        loaded = yaml.safe_load(hparams.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    encoder = loaded.get("pretrained_model") if isinstance(loaded, dict) else None
    return str(encoder) if encoder else None


def _fetch_metrics(metrics: MetricsSpec) -> dict[str, Any]:
    report: dict[str, Any] = {}

    if "neural" in metrics.groups:
        try:
            from comet import download_model
        except ImportError:
            for model_name in metrics.comet_models:
                report[f"comet:{model_name}"] = {
                    "ok": False,
                    "reason": (
                        "unbabel-comet not installed in this environment. Prefetch COMET "
                        "checkpoints from the COMET environment: "
                        ".venv-comet/bin/mnlp-eval prefetch"
                    ),
                }
        else:
            for model_name in metrics.comet_models:
                try:
                    path = download_model(model_name)
                    _log(f"  comet {model_name}: cached")
                    report[f"comet:{model_name}"] = {"ok": True, "path": str(path)}
                    encoder = _comet_encoder(path)
                    if encoder:
                        # COMET loads the encoder config and tokenizer from the Hub.
                        report[f"comet-encoder:{encoder}"] = _fetch_repo(
                            encoder,
                            None,
                            allow_patterns=["*.json", "*.model", "*.txt"],
                        )
                except Exception as exc:
                    hint = ""
                    if "gated" in str(exc).lower() or "401" in str(exc):
                        hint = " Gated: accept the licence on the Hub and export HF_TOKEN."
                    report[f"comet:{model_name}"] = {
                        "ok": False,
                        "reason": f"{type(exc).__name__}: {exc}{hint}",
                    }

    if "metricx" in metrics.groups:
        report[f"metricx:{metrics.metricx_model}"] = _fetch_repo(metrics.metricx_model, None)
        # Tokenizer files only; the mt5-xl weights are not needed.
        report[f"metricx-tokenizer:{metrics.metricx_tokenizer}"] = _fetch_repo(
            metrics.metricx_tokenizer,
            None,
            allow_patterns=[
                "*.json",
                "*.model",
                "*.txt",
                "spiece.model",
                "tokenizer.json",
            ],
        )

    return report
