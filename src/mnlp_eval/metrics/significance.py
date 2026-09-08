"""Paired bootstrap resampling against a baseline.

A compression report full of deltas without significance tests invites the
reader to over-read noise. A 0.3 BLEU drop on 1000 segments is not evidence of
anything, and saying so is more useful than reporting it as a finding.

Two paths. Both are paired bootstrap tests and both are add-one smoothed, but
they are **not the same estimator**, and a p-value from one is not directly
comparable with a p-value from the other:

* Surface metrics use sacreBLEU's own ``PairedTest``, which recomputes BLEU and
  chrF++ from sufficient statistics on each resample. That is correct in a way
  that averaging per-sentence BLEU is not, and it is what the MT literature
  reports, so it is the right choice despite the incomparability. Its statistic
  centres the absolute score difference on its own bootstrap mean and counts
  one-sided.
* Neural metrics are resampled from the per-segment scores stored at scoring
  time, so no metric model has to be reloaded. Its statistic is the two-sided
  centred difference, ``P(|d_b - d| >= |d|)``.

An earlier version of this module claimed both used the same statistic. They do
not, the two rejection rates under the null differ, and the report labels the
columns accordingly rather than pretending one asterisk threshold means the
same thing in both.

Both are genuinely paired: the same resampled segment indices are applied to
both systems, which removes test-set difficulty as a source of variance and is
why this test is far more sensitive than comparing independent confidence
intervals.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

__all__ = ["bootstrap_segment_delta", "paired_bootstrap_surface"]

DEFAULT_SAMPLES = 1000
DEFAULT_SEED = 12345


def paired_bootstrap_surface(
    baseline_hypotheses: Sequence[str],
    system_hypotheses: Sequence[str],
    references: Sequence[str],
    target_language: str,
    *,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    compute_ter: bool = False,
) -> dict[str, Any]:
    """Test a system against a baseline on BLEU and chrF++."""
    if seed == 0:
        # sacreBLEU reads SACREBLEU_SEED then does `self._seed if self._seed
        # else None`, so zero is falsy and silently means "draw entropy from
        # the OS". The recorded signature would still claim seed 0, giving a
        # provenance record that says seeded while the surface p-values change
        # between identical runs.
        msg = (
            "bootstrap_seed 0 is not usable: sacreBLEU treats it as unseeded, so "
            "surface p-values would not be reproducible while claiming to be. "
            "Use any non-zero seed."
        )
        raise ValueError(msg)

    # sacreBLEU seeds its resampler from the environment, and reads it when
    # PairedTest is constructed, so it has to be set before that and restored
    # afterwards rather than leaking into the rest of the process.
    previous_seed = os.environ.get("SACREBLEU_SEED")
    os.environ["SACREBLEU_SEED"] = str(seed)
    from sacrebleu.metrics import BLEU, CHRF, TER
    from sacrebleu.significance import PairedTest

    from mnlp_eval.languages import sacrebleu_tokenizer

    if not (len(baseline_hypotheses) == len(system_hypotheses) == len(references)):
        msg = (
            f"length mismatch: {len(baseline_hypotheses)} baseline, "
            f"{len(system_hypotheses)} system, {len(references)} references"
        )
        raise ValueError(msg)

    metrics: dict[str, Any] = {
        "bleu": BLEU(tokenize=sacrebleu_tokenizer(target_language)),
        "chrf2pp": CHRF(word_order=2),
    }
    if compute_ter:
        metrics["ter"] = TER()

    # PairedTest takes a list of (name, hypotheses); the first entry is the
    # baseline every other system is tested against.
    named_systems: list[tuple[str, Sequence[str]]] = [
        ("baseline", list(baseline_hypotheses)),
        ("system", list(system_hypotheses)),
    ]
    test = PairedTest(
        named_systems,
        metrics,
        references=[list(references)],
        test_type="bs",
        n_samples=n_samples,
    )
    try:
        signatures, results = test()
    finally:
        if previous_seed is None:
            os.environ.pop("SACREBLEU_SEED", None)
        else:
            os.environ["SACREBLEU_SEED"] = previous_seed

    output: dict[str, Any] = {
        "method": "paired bootstrap resampling",
        "n_samples": n_samples,
        "seed": seed,
        "n_segments": len(references),
        "metrics": {},
    }
    for metric_name, values in results.items():
        # sacreBLEU puts the system names under a "System" key alongside the
        # metric rows, so entries are filtered by shape, not by key name.
        if metric_name == "System" or len(values) < 2:
            continue
        baseline_result, system_result = values[0], values[1]
        if isinstance(baseline_result, str) or isinstance(system_result, str):
            continue
        key = metric_name.lower().replace("2++", "2pp").replace("+", "p")
        output["metrics"][key] = {
            "baseline_score": _round(baseline_result.score),
            "system_score": _round(system_result.score),
            "delta": _round(
                system_result.score - baseline_result.score
                if system_result.score is not None and baseline_result.score is not None
                else None
            ),
            "p_value": _round(system_result.p_value, 6),
            "significant_at_0.05": (
                system_result.p_value is not None and system_result.p_value < 0.05
            ),
            "bootstrap_mean": _round(system_result.mean),
            # sacreBLEU returns a half-width around the system's own score, not
            # an interval on the difference. Named to say so, because the
            # neural path's bootstrap_ci_95 is an interval on the delta and the
            # two were previously a rename apart.
            "bootstrap_score_ci_halfwidth": _round(system_result.ci),
            # sacreBLEU returns signature objects here, not strings, and a
            # signature object is not JSON serialisable.
            "signature": _signature_text(signatures.get(metric_name)),
        }
    return output


def bootstrap_segment_delta(
    baseline_scores: Sequence[float],
    system_scores: Sequence[float],
    *,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Paired bootstrap on stored per-segment metric scores.

    Uses the centred p-value, the same convention sacreBLEU applies: the share
    of resamples whose deviation from the observed difference is at least as
    large as the observed difference itself, with add-one smoothing so a
    p-value of exactly zero is never reported from a finite number of samples.
    """
    import numpy as np

    if len(baseline_scores) != len(system_scores):
        msg = f"{len(baseline_scores)} baseline scores but {len(system_scores)} system scores"
        raise ValueError(msg)
    if not baseline_scores:
        msg = "no segment scores to resample"
        raise ValueError(msg)

    baseline = np.asarray(baseline_scores, dtype=np.float64)
    system = np.asarray(system_scores, dtype=np.float64)
    observed = float(system.mean() - baseline.mean())

    rng = np.random.default_rng(seed)
    n = len(baseline)
    indices = rng.integers(0, n, size=(n_samples, n))
    deltas = system[indices].mean(axis=1) - baseline[indices].mean(axis=1)

    centred = np.abs(deltas - observed) >= abs(observed)
    p_value = float((1 + int(centred.sum())) / (n_samples + 1))
    improved = observed > 0 if higher_is_better else observed < 0

    return {
        "method": "paired bootstrap resampling on per-segment scores",
        "n_samples": n_samples,
        "seed": seed,
        "n_segments": n,
        "baseline_score": round(float(baseline.mean()), 6),
        "system_score": round(float(system.mean()), 6),
        "delta": round(observed, 6),
        "improved": bool(improved),
        "higher_is_better": higher_is_better,
        "p_value": round(p_value, 6),
        "significant_at_0.05": p_value < 0.05,
        "bootstrap_ci_95": [
            round(float(np.percentile(deltas, 2.5)), 6),
            round(float(np.percentile(deltas, 97.5)), 6),
        ],
    }


def _signature_text(signature: Any) -> str | None:
    if signature is None:
        return None
    formatter = getattr(signature, "format", None)
    return str(formatter()) if callable(formatter) else str(signature)


def _round(value: Any, digits: int = 4) -> Any:
    if value is None or isinstance(value, str):
        return value
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value
