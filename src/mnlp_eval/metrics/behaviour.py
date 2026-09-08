"""Behavioural failure metrics: how a model breaks, not just how well it scores.

An averaged COMET score is a poor instrument for compression research. A model
that translates 90 percent of segments well and emits Icelandic-looking noise
on the rest can score similarly to one that is uniformly mediocre, and the two
call for completely different conclusions. Off-target output in particular is
the signature failure of aggressively compressed multilingual models, and it is
invisible in every quality metric this project reports.

Language identification uses a classifier restricted to the languages that are
plausible for the direction, namely the target, the source, and English. A
restricted decision is far more reliable on single sentences than an open
choice among 142 languages, and it answers the question that actually matters:
did the model translate, echo the source, or fall back to English. The open
142-way rate is reported alongside it for transparency.
"""

from __future__ import annotations

import string
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mnlp_eval.artifacts import Segment
from mnlp_eval.languages import Direction
from mnlp_eval.postprocess import FLAG_REPETITION

__all__ = ["BehaviourScores", "score_behaviour"]

#: Hypotheses shorter than this are excluded from language identification.
#: Language ID on a two-word fragment is noise, and counting that noise as
#: off-target output would manufacture a finding.
MIN_LID_CHARACTERS = 12

_PUNCTUATION = str.maketrans("", "", string.punctuation + string.whitespace)


@dataclass
class BehaviourScores:
    """Failure-mode rates for one direction."""

    n_segments: int
    empty_rate: float
    source_copy_rate: float
    length_ratio: float
    truncation_rate: float
    budget_hit_rate: float
    repetition_rate: float
    wasted_token_fraction: float
    parse_flag_rates: dict[str, float] = field(default_factory=dict)
    off_target_rate: float | None = None
    source_language_rate: float | None = None
    english_fallback_rate: float | None = None
    off_target_rate_open: float | None = None
    lid: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise(text: str) -> str:
    """Casefold and strip punctuation, for near-exact copy detection."""
    decomposed = unicodedata.normalize("NFKC", text).casefold()
    return decomposed.translate(_PUNCTUATION)


def score_behaviour(
    segments: Sequence[Segment],
    direction: Direction,
    *,
    lid_backend: str = "auto",
) -> BehaviourScores:
    """Compute behavioural failure rates for one direction."""
    if not segments:
        msg = "no segments to score"
        raise ValueError(msg)

    total = len(segments)
    hypotheses = [segment.hypothesis for segment in segments]
    empty = sum(1 for text in hypotheses if not text.strip())

    copies = sum(
        1
        for segment in segments
        if segment.hypothesis.strip()
        and _normalise(segment.hypothesis) == _normalise(segment.source)
    )

    hypothesis_chars = sum(len(text) for text in hypotheses)
    reference_chars = sum(len(segment.reference) for segment in segments)

    flag_counts: dict[str, int] = {}
    for segment in segments:
        for flag in segment.parse_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    generated = sum(segment.n_generated_tokens for segment in segments)
    wasted = sum(segment.n_wasted_tokens for segment in segments)

    scores = BehaviourScores(
        n_segments=total,
        empty_rate=round(empty / total, 4),
        source_copy_rate=round(copies / total, 4),
        length_ratio=round(hypothesis_chars / reference_chars, 4) if reference_chars else 0.0,
        truncation_rate=round(sum(1 for s in segments if s.truncated) / total, 4),
        budget_hit_rate=round(sum(1 for s in segments if s.hit_token_budget) / total, 4),
        repetition_rate=round(
            sum(1 for s in segments if FLAG_REPETITION in s.parse_flags) / total, 4
        ),
        wasted_token_fraction=round(wasted / generated, 4) if generated else 0.0,
        parse_flag_rates={
            flag: round(count / total, 4) for flag, count in sorted(flag_counts.items())
        },
    )

    _add_language_scores(scores, hypotheses, direction, lid_backend)
    return scores


def _add_language_scores(
    scores: BehaviourScores,
    hypotheses: Sequence[str],
    direction: Direction,
    lid_backend: str,
) -> None:
    if lid_backend == "none":
        scores.lid = {"backend": "none", "reason": "disabled by configuration"}
        return

    try:
        from py3langid.langid import MODEL_FILE, LanguageIdentifier
    except ImportError:
        scores.lid = {
            "backend": None,
            "reason": "py3langid not installed; pip install -e '.[surface]'",
        }
        return

    candidates = list(dict.fromkeys([direction.target, direction.source, "en"]))
    restricted = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    restricted.set_languages(candidates)
    open_model = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    # Every branch below reports the same keys, so two runs' lid blocks can be
    # diffed directly.
    base_report: dict[str, Any] = {
        "backend": "py3langid",
        "languages": candidates,
        "min_characters": MIN_LID_CHARACTERS,
    }

    evaluated = 0
    off_target = 0
    source_language = 0
    english = 0
    off_target_open = 0

    for text in hypotheses:
        stripped = text.strip()
        if len(stripped) < MIN_LID_CHARACTERS:
            continue
        evaluated += 1
        detected, _ = restricted.classify(stripped)
        if detected != direction.target:
            off_target += 1
            if detected == direction.source:
                source_language += 1
            if detected == "en":
                english += 1
        detected_open, _ = open_model.classify(stripped)
        if detected_open != direction.target:
            off_target_open += 1

    if not evaluated:
        scores.lid = {
            **base_report,
            "n_evaluated": 0,
            "n_skipped_too_short": len(hypotheses),
            "reason": f"no hypothesis reached {MIN_LID_CHARACTERS} characters",
        }
        return

    scores.off_target_rate = round(off_target / evaluated, 4)
    scores.source_language_rate = round(source_language / evaluated, 4)
    scores.english_fallback_rate = (
        round(english / evaluated, 4) if direction.target != "en" else None
    )
    scores.off_target_rate_open = round(off_target_open / evaluated, 4)
    scores.lid = {
        **base_report,
        "n_evaluated": evaluated,
        "n_skipped_too_short": len(hypotheses) - evaluated,
    }
