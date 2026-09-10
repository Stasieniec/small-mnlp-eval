"""Hypothesis extraction, including the cases that break naive parsing."""

from __future__ import annotations

from mnlp_eval.postprocess import (
    FLAG_EXTRA_LINES,
    FLAG_REPETITION,
    FLAG_SKIPPED_BLANK,
    FLAG_STRIPPED_MARKER,
    STATUS_EMPTY,
    STATUS_OK,
    detect_repetition,
    extract_hypothesis,
)

MARKER = "English:"


def test_clean_output_needs_no_recovery() -> None:
    parsed = extract_hypothesis(" This is a test.", MARKER)
    assert parsed.hypothesis == "This is a test."
    assert parsed.status == STATUS_OK
    assert parsed.flags == ()


def test_empty_output_is_reported_not_absorbed() -> None:
    parsed = extract_hypothesis("", MARKER)
    assert parsed.hypothesis == ""
    assert parsed.status == STATUS_EMPTY
    assert parsed.is_empty


def test_whitespace_only_output_is_empty() -> None:
    assert extract_hypothesis("   \n  \n ", MARKER).status == STATUS_EMPTY


def test_leading_blank_lines_are_flagged() -> None:
    parsed = extract_hypothesis("\n\nThis is a test.", MARKER)
    assert parsed.hypothesis == "This is a test."
    assert FLAG_SKIPPED_BLANK in parsed.flags


def test_echoed_marker_is_stripped_and_flagged() -> None:
    parsed = extract_hypothesis(" English: This is a test.", MARKER)
    assert parsed.hypothesis == "This is a test."
    assert FLAG_STRIPPED_MARKER in parsed.flags


def test_marker_on_its_own_line_moves_to_the_next_line() -> None:
    parsed = extract_hypothesis("\nEnglish:\nThis is a test.", MARKER)
    assert parsed.hypothesis == "This is a test."
    assert FLAG_STRIPPED_MARKER in parsed.flags


def test_marker_with_nothing_after_it_is_empty() -> None:
    parsed = extract_hypothesis("English:", MARKER)
    assert parsed.status == STATUS_EMPTY
    assert FLAG_STRIPPED_MARKER in parsed.flags


def test_trailing_commentary_is_discarded_and_flagged() -> None:
    # Instruction-tuned models routinely keep talking after the translation.
    parsed = extract_hypothesis(
        " This is a test.\nExplanation: the German sentence means...", MARKER
    )
    assert parsed.hypothesis == "This is a test."
    assert FLAG_EXTRA_LINES in parsed.flags


def test_degenerate_repetition_is_flagged() -> None:
    parsed = extract_hypothesis("the the the the the", MARKER)
    assert FLAG_REPETITION in parsed.flags


def test_repetition_detection_finds_the_shortest_unit() -> None:
    assert detect_repetition("Hallo. Hallo. Hallo. Hallo.") == "Hallo."
    # "a" alone does not repeat contiguously here, so the two-token cycle is
    # the shortest unit that does.
    assert detect_repetition("a b a b a b a b") == "a b"
    assert detect_repetition("a a a a a a") == "a"


def test_repetition_detection_leaves_healthy_text_alone() -> None:
    assert detect_repetition("This is an ordinary sentence with distinct words.") is None


def test_repetition_detection_needs_enough_tokens() -> None:
    assert detect_repetition("the the") is None


def test_repetition_detection_can_be_disabled() -> None:
    parsed = extract_hypothesis("the the the the the", MARKER, detect_repeats=False)
    assert FLAG_REPETITION not in parsed.flags


def test_no_marker_still_extracts_the_first_line() -> None:
    parsed = extract_hypothesis("Dies ist ein Test.\nnoise", "")
    assert parsed.hypothesis == "Dies ist ein Test."
