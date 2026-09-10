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
choice among all 142 languages it knows, and it answers what matters:
did the model translate, echo the source, or fall back to English. The open
rate over every language the classifier knows is reported alongside it.

Every rate is divided by the total segment count, never by the subset that
could be classified. Dividing by the subset makes a model that emits nothing on
most segments report an off-target rate near zero, since the few segments it
does produce are the ones it got right. ``on_target_rate``, ``off_target_rate``
and ``unverifiable_rate`` sum to one by construction.
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
    truncation_rate: float
    budget_hit_rate: float
    repetition_rate: float
    length_ratio: float | None = None
    wasted_token_fraction: float | None = None
    parse_flag_rates: dict[str, float] = field(default_factory=dict)
    #: Share of all segments confirmed to be in the target language.
    on_target_rate: float | None = None
    #: Share of all segments confirmed to be in some other language.
    off_target_rate: float | None = None
    #: Share of all segments too short or too empty to identify. The three
    #: rates above sum to one.
    unverifiable_rate: float | None = None
    source_language_rate: float | None = None
    english_fallback_rate: float | None = None
    off_target_rate_open: float | None = None
    #: Off-target share among classifiable hypotheses only. Comparable with the
    #: figure usually quoted in the literature, but it says nothing about
    #: segments that produced no output, so it is never the headline.
    off_target_rate_among_scorable: float | None = None
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

    # _normalise strips punctuation, so two punctuation-only strings would
    # otherwise compare equal and a model emitting "!!!" would be scored as
    # copying the source rather than as producing garbage.
    copies = sum(
        1
        for segment in segments
        if _normalise(segment.hypothesis)
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
        length_ratio=round(hypothesis_chars / reference_chars, 4) if reference_chars else None,
        truncation_rate=round(sum(1 for s in segments if s.truncated) / total, 4),
        budget_hit_rate=round(sum(1 for s in segments if s.hit_token_budget) / total, 4),
        repetition_rate=round(
            sum(1 for s in segments if FLAG_REPETITION in s.parse_flags) / total, 4
        ),
        wasted_token_fraction=round(wasted / generated, 4) if generated else None,
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
    total = len(hypotheses)
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

    scorable = 0
    on_target = 0
    off_target = 0
    source_language = 0
    english = 0
    off_target_open = 0

    for text in hypotheses:
        stripped = text.strip()
        if len(stripped) < MIN_LID_CHARACTERS:
            # Language identification on a two-word fragment is noise. These
            # segments are counted as unverifiable rather than assigned to
            # either side.
            continue
        scorable += 1
        detected, _ = restricted.classify(stripped)
        if detected == direction.target:
            on_target += 1
        else:
            off_target += 1
            if detected == direction.source:
                source_language += 1
            if detected == "en":
                english += 1
        detected_open, _ = open_model.classify(stripped)
        if detected_open != direction.target:
            off_target_open += 1

    scores.lid = {
        "backend": "py3langid",
        "languages": candidates,
        "n_segments": total,
        "n_scorable": scorable,
        "n_unverifiable": total - scorable,
        "min_characters": MIN_LID_CHARACTERS,
        "denominator": "n_segments for every rate except off_target_rate_among_scorable",
    }
    if not total:
        return

    # Divided by the full segment count, so a model that emits nothing cannot
    # score zero off-target. The three shares sum to one.
    scores.on_target_rate = round(on_target / total, 4)
    scores.off_target_rate = round(off_target / total, 4)
    scores.unverifiable_rate = round((total - scorable) / total, 4)
    scores.source_language_rate = round(source_language / total, 4)
    scores.off_target_rate_open = round(off_target_open / total, 4)
    scores.off_target_rate_among_scorable = round(off_target / scorable, 4) if scorable else None

    # When the source is English, "fell back to English" and "echoed the source
    # language" are the same event, and reporting one number twice under two
    # headings invents a second independent failure mode. Only report it when
    # it says something the source-language rate does not.
    if direction.target != "en" and direction.source != "en":
        scores.english_fallback_rate = round(english / total, 4)
