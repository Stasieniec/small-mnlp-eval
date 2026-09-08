"""Download everything a run needs, ahead of time.

Snellius compute nodes have no internet access. Anything a batch job touches
has to be in ``HF_HOME`` before the job starts, or the job fails several
minutes in with a network error and burns an allocation.

This runs on a login node. Because the three environments hold different metric
libraries, each one prefetches what it can and reports the rest, in the same
way the score stage does.
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


def _fetch_repo(reference: str, revision: str | None) -> dict[str, Any]:
    if Path(reference).expanduser().is_dir():
        return {"ok": True, "note": "local directory"}
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(reference, revision=revision)
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
                except Exception as exc:
                    hint = ""
                    if "gated" in str(exc).lower() or "401" in str(exc):
                        hint = " Gated: accept the licence on the Hub and export HF_TOKEN."
                    report[f"comet:{model_name}"] = {
                        "ok": False,
                        "reason": f"{type(exc).__name__}: {exc}{hint}",
                    }

    if "metricx" in metrics.groups:
        for label, reference in (
            ("metricx", metrics.metricx_model),
            ("metricx-tokenizer", metrics.metricx_tokenizer),
        ):
            report[f"{label}:{reference}"] = _fetch_repo(reference, None)

    return report
