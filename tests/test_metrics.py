"""Surface metrics, behavioural metrics, and significance testing."""

from __future__ import annotations

import pytest

from mnlp_eval.artifacts import Segment
from mnlp_eval.languages import parse_direction
from mnlp_eval.metrics.base import METRIC_GROUPS, MetricScore, group_availability
from mnlp_eval.metrics.behaviour import MIN_LID_CHARACTERS, score_behaviour
from mnlp_eval.metrics.significance import bootstrap_segment_delta, paired_bootstrap_surface
from mnlp_eval.metrics.surface import score_surface

DE_EN = parse_direction("de-en")
EN_DE = parse_direction("en-de")

REFERENCES = [
    "The goods cost less than 20 euros.",
    "The fee would equal 40% of the value of the goods.",
    "I am a serious customer and that is why it is not a problem.",
    "Thank you for contacting us today, I am happy to help.",
]


def _segments(hypotheses: list[str], sources: list[str] | None = None, **kwargs: object):
    sources = sources or [f"Quelle {index}" for index in range(len(hypotheses))]
    return [
        Segment(
            index=index,
            source=sources[index],
            reference=REFERENCES[index % len(REFERENCES)],
            hypothesis=hypothesis,
            n_generated_tokens=max(len(hypothesis.split()), 1),
            **kwargs,  # type: ignore[arg-type]
        )
        for index, hypothesis in enumerate(hypotheses)
    ]


# --------------------------------------------------------------------------
# Surface metrics
# --------------------------------------------------------------------------


def test_perfect_output_scores_100_bleu() -> None:
    scores = score_surface(REFERENCES, REFERENCES, "en")
    assert scores["bleu"].score == pytest.approx(100.0)
    assert scores["chrf2pp"].score == pytest.approx(100.0)


def test_signatures_are_recorded() -> None:
    # A score without its signature cannot be compared with a score from
    # anywhere else, including ALMA's published tables.
    scores = score_surface(REFERENCES, REFERENCES, "en")
    assert "tok:13a" in (scores["bleu"].signature or "")
    assert "version:" in (scores["bleu"].signature or "")
    assert "nw:2" in (scores["chrf2pp"].signature or "")


def test_target_language_selects_almas_tokenizer() -> None:
    chinese = ["这是一个测试。"] * 2
    scores = score_surface(chinese, chinese, "zh")
    assert scores["bleu"].extra["tokenizer"] == "zh"
    assert "tok:zh" in (scores["bleu"].signature or "")


def test_ter_is_optional_and_lower_is_better() -> None:
    assert "ter" not in score_surface(REFERENCES, REFERENCES, "en")
    scores = score_surface(REFERENCES, REFERENCES, "en", compute_ter=True)
    assert scores["ter"].higher_is_better is False


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="references"):
        score_surface(REFERENCES[:2], REFERENCES, "en")


def test_empty_hypotheses_score_near_zero_rather_than_being_skipped() -> None:
    # A collapsed model must be penalised by the metric, not excused by the
    # harness.
    scores = score_surface([""] * len(REFERENCES), REFERENCES, "en")
    assert scores["bleu"].score == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Behavioural metrics
# --------------------------------------------------------------------------


def test_healthy_output_has_no_failure_rates() -> None:
    scores = score_behaviour(_segments(REFERENCES), DE_EN)
    assert scores.empty_rate == 0.0
    assert scores.source_copy_rate == 0.0
    assert scores.truncation_rate == 0.0
    assert scores.off_target_rate == 0.0


def test_empty_output_is_counted() -> None:
    scores = score_behaviour(_segments([REFERENCES[0], "", "  ", REFERENCES[3]]), DE_EN)
    assert scores.empty_rate == 0.5


def test_source_copy_is_detected_ignoring_case_and_punctuation() -> None:
    sources = ["Das ist ein Test.", "Der Hund schlaeft."]
    hypotheses = ["das ist ein test", "The dog is sleeping."]
    scores = score_behaviour(_segments(hypotheses, sources), DE_EN)
    assert scores.source_copy_rate == 0.5


def test_off_target_output_is_detected() -> None:
    # The signature failure of an aggressively compressed multilingual model,
    # and invisible in every quality metric.
    hypotheses = [
        "Das ist ein deutscher Satz mit mehreren Woertern.",
        "This is a perfectly ordinary English sentence.",
    ]
    scores = score_behaviour(_segments(hypotheses), DE_EN)
    assert scores.off_target_rate == 0.5
    assert scores.on_target_rate == 0.5
    assert scores.source_language_rate == 0.5


def test_language_shares_always_sum_to_one() -> None:
    # The invariant that makes a collapsed system impossible to hide. Divide
    # the language rates by the classifiable subset instead and this breaks.
    for hypotheses in (
        REFERENCES,
        ["", "", "Ok", "This is a perfectly ordinary English sentence."],
        ["Das ist ein deutscher Satz mit mehreren Woertern."] * 4,
    ):
        scores = score_behaviour(_segments(hypotheses), DE_EN)
        total = (
            (scores.on_target_rate or 0.0)
            + (scores.off_target_rate or 0.0)
            + (scores.unverifiable_rate or 0.0)
        )
        assert total == pytest.approx(1.0), hypotheses


def test_a_collapsed_model_cannot_score_zero_off_target() -> None:
    # 95 empty hypotheses and 5 good ones. Dividing by the classifiable subset
    # reported off_target_rate 0.0 here, the most flattering possible value for
    # the most broken possible system.
    hypotheses = [""] * 95 + ["This is a German sentence here."] * 5
    scores = score_behaviour(_segments(hypotheses), DE_EN)
    assert scores.unverifiable_rate == 0.95
    assert scores.on_target_rate == 0.05
    # Retained under an explicit name, so the misleading figure is still
    # available for comparison with the literature but is never the headline.
    assert scores.off_target_rate_among_scorable == 0.0


def test_short_hypotheses_count_as_unverifiable_not_on_target() -> None:
    # Language ID on a two-word fragment is noise, so it is neither credited
    # nor blamed. It is reported as unverifiable.
    scores = score_behaviour(_segments(["Ok", "Ja", "Nein"]), DE_EN)
    assert scores.lid["n_scorable"] == 0
    assert scores.unverifiable_rate == 1.0
    assert scores.on_target_rate == 0.0
    assert scores.off_target_rate_among_scorable is None
    assert scores.lid["min_characters"] == MIN_LID_CHARACTERS


def test_language_identification_can_be_disabled() -> None:
    scores = score_behaviour(_segments(REFERENCES), DE_EN, lid_backend="none")
    assert scores.off_target_rate is None
    assert scores.on_target_rate is None
    assert scores.lid["backend"] == "none"


def test_english_fallback_is_not_reported_when_it_duplicates_source_language() -> None:
    # For an out-of-English direction, "fell back to English" and "echoed the
    # source language" are the same event. Reporting one number under two
    # headings would invent a second independent failure mode.
    into_german = score_behaviour(
        _segments(
            ["This is a perfectly ordinary English sentence."],
            sources=["This is a perfectly ordinary English sentence."],
        ),
        EN_DE,
    )
    assert into_german.source_language_rate == 1.0
    assert into_german.english_fallback_rate is None

    # Into English, an English fallback is simply on-target.
    into_english = score_behaviour(_segments(REFERENCES), DE_EN)
    assert into_english.english_fallback_rate is None


def test_punctuation_only_output_is_not_a_source_copy() -> None:
    # _normalise strips punctuation, so without a guard "!!!" and "..." are
    # equal and garbage gets scored as a faithful copy.
    scores = score_behaviour(_segments(["!!!", "!!!"], sources=["...", "..."]), DE_EN)
    assert scores.source_copy_rate == 0.0


def test_zero_denominators_report_none_rather_than_zero() -> None:
    segments = _segments(["something"])
    segments[0].reference = ""
    segments[0].n_generated_tokens = 0
    segments[0].n_wasted_tokens = 0
    scores = score_behaviour(segments, DE_EN)
    assert scores.length_ratio is None
    assert scores.wasted_token_fraction is None


def test_truncation_and_budget_rates_are_separate() -> None:
    # A model that finished translating and then rambled hit the budget without
    # truncating its translation.
    segments = _segments(REFERENCES)
    segments[0].hit_token_budget = True
    segments[0].truncated = True
    segments[1].hit_token_budget = True
    scores = score_behaviour(segments, DE_EN)
    assert scores.budget_hit_rate == 0.5
    assert scores.truncation_rate == 0.25


def test_wasted_token_fraction_is_computed_over_tokens_not_segments() -> None:
    segments = _segments(REFERENCES)
    for segment in segments:
        segment.n_generated_tokens = 10
        segment.n_wasted_tokens = 4
    assert score_behaviour(segments, DE_EN).wasted_token_fraction == pytest.approx(0.4)


def test_repetition_flag_is_aggregated() -> None:
    segments = _segments(REFERENCES)
    segments[0].parse_flags = ["repetition"]
    assert score_behaviour(segments, DE_EN).repetition_rate == 0.25


def test_parse_flag_rates_are_reported() -> None:
    segments = _segments(REFERENCES)
    segments[0].parse_flags = ["extra_lines"]
    segments[1].parse_flags = ["extra_lines", "stripped_marker"]
    rates = score_behaviour(segments, DE_EN).parse_flag_rates
    assert rates == {"extra_lines": 0.5, "stripped_marker": 0.25}


def test_empty_segment_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="no segments"):
        score_behaviour([], DE_EN)


# --------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------


def test_paired_bootstrap_finds_a_real_difference() -> None:
    degraded = ["", "", "wrong", "wrong"]
    result = paired_bootstrap_surface(REFERENCES, degraded, REFERENCES, "en", n_samples=200)
    bleu = result["metrics"]["bleu"]
    assert bleu["delta"] < 0
    assert bleu["significant_at_0.05"]
    assert "bs:200" in bleu["signature"]


def test_paired_bootstrap_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        paired_bootstrap_surface(REFERENCES, REFERENCES[:2], REFERENCES, "en")


def test_segment_bootstrap_does_not_cry_wolf() -> None:
    import random

    random.seed(0)
    baseline = [0.85 + random.gauss(0, 0.05) for _ in range(400)]
    identical_noise = [value + random.gauss(0, 0.05) for value in baseline]
    result = bootstrap_segment_delta(baseline, identical_noise, n_samples=500)
    assert not result["significant_at_0.05"]


def test_segment_bootstrap_detects_a_real_drop() -> None:
    import random

    random.seed(1)
    baseline = [0.85 + random.gauss(0, 0.05) for _ in range(400)]
    degraded = [value - 0.03 + random.gauss(0, 0.05) for value in baseline]
    result = bootstrap_segment_delta(baseline, degraded, n_samples=500)
    assert result["delta"] < 0
    assert not result["improved"]
    assert result["significant_at_0.05"]
    low, high = result["bootstrap_ci_95"]
    assert low < result["delta"] < high
    assert high < 0


def test_segment_bootstrap_honours_metric_direction() -> None:
    # MetricX predicts an error score, so a lower value is an improvement.
    baseline = [5.0] * 50
    better = [4.0] * 50
    assert bootstrap_segment_delta(baseline, better, higher_is_better=False)["improved"]
    assert not bootstrap_segment_delta(baseline, better, higher_is_better=True)["improved"]


def test_segment_bootstrap_p_value_is_never_exactly_zero() -> None:
    # Add-one smoothing: a finite number of resamples cannot prove p = 0.
    result = bootstrap_segment_delta([0.0] * 100, [1.0] * 100, n_samples=100)
    assert result["p_value"] > 0


def test_segment_bootstrap_is_reproducible() -> None:
    baseline = [0.8, 0.9, 0.7, 0.85, 0.6]
    system = [0.7, 0.8, 0.6, 0.9, 0.5]
    first = bootstrap_segment_delta(baseline, system, n_samples=200, seed=7)
    second = bootstrap_segment_delta(baseline, system, n_samples=200, seed=7)
    assert first == second


def test_segment_bootstrap_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="baseline scores"):
        bootstrap_segment_delta([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="no segment scores"):
        bootstrap_segment_delta([], [])


# --------------------------------------------------------------------------
# Group availability
# --------------------------------------------------------------------------


def test_group_availability_covers_every_group_with_a_reason() -> None:
    availability = group_availability()
    assert set(availability) == set(METRIC_GROUPS)
    for group, (served, reason) in availability.items():
        assert served or reason, f"{group} is unavailable without explaining why"


def test_metric_score_omits_segments_when_asked() -> None:
    score = MetricScore(name="x", score=1.0, segment_scores=[1.0, 2.0])
    assert "segment_scores" in score.to_dict()
    assert "segment_scores" not in score.to_dict(include_segments=False)
