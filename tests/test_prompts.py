"""Prompt templates and the fingerprint that pins them into a manifest."""

from __future__ import annotations

import pytest

from mnlp_eval.languages import parse_direction
from mnlp_eval.prompts import (
    AlmaPrompt,
    PassthroughPrompt,
    available_prompts,
    get_prompt,
    prompt_hash,
)

DE_EN = parse_direction("de-en")
EN_IS = parse_direction("en-is")


def test_alma_prompt_is_byte_identical_to_upstream() -> None:
    # Reproduced from utils/utils.py:get_prompt in fe1ixxu/ALMA. The upstream
    # README says "into" while the code says "to"; the code produced the
    # published scores.
    rendered = AlmaPrompt().render(DE_EN, "Das ist ein Test.")
    assert rendered == (
        "Translate this from German to English:\nGerman: Das ist ein Test.\nEnglish:"
    )


def test_alma_prompt_matches_upstream_readme_example() -> None:
    zh_en = parse_direction("zh-en")
    rendered = AlmaPrompt().render(zh_en, "我爱机器翻译。")
    assert rendered.startswith("Translate this from Chinese to English:\nChinese: ")
    assert rendered.endswith("\nEnglish:")


def test_target_marker_is_the_language_cue() -> None:
    assert AlmaPrompt().target_marker(EN_IS) == "Icelandic:"


def test_passthrough_returns_the_source_untouched() -> None:
    prompt = PassthroughPrompt()
    assert prompt.render(DE_EN, "Das ist ein Test.") == "Das ist ein Test."
    assert prompt.target_marker(DE_EN) == ""


def test_registry_lists_and_builds_every_template() -> None:
    for name in available_prompts():
        assert get_prompt(name).name == name


def test_unknown_template_lists_the_alternatives() -> None:
    with pytest.raises(KeyError, match="unknown prompt template"):
        get_prompt("nope")


def test_fingerprints_are_stable_and_distinct() -> None:
    assert prompt_hash(AlmaPrompt()) == prompt_hash(AlmaPrompt())
    assert prompt_hash(AlmaPrompt()) != prompt_hash(PassthroughPrompt())


def test_fingerprint_changes_when_the_template_changes() -> None:
    class Edited(AlmaPrompt):
        name = "edited"

        def render(self, direction, source):  # type: ignore[no-untyped-def]
            return super().render(direction, source) + " "

    # An accidental whitespace edit must invalidate existing comparisons.
    assert prompt_hash(Edited()) != prompt_hash(AlmaPrompt())


def test_chat_template_flag_affects_the_fingerprint() -> None:
    assert prompt_hash(get_prompt("alma")) != prompt_hash(get_prompt("alma_chat"))
