"""Language tables and language-dependent evaluation settings."""

from __future__ import annotations

import pytest

from mnlp_eval.languages import (
    LANG_TABLE,
    NLLB_CODE,
    lang_name,
    max_source_length_for,
    parse_direction,
    parse_directions,
    sacrebleu_tokenizer,
)


def test_tables_cover_the_same_languages() -> None:
    assert set(LANG_TABLE) == set(NLLB_CODE)


def test_alma_language_names_are_exact() -> None:
    # These strings go into the prompt verbatim, so they are part of the
    # evaluation protocol and must match fe1ixxu/ALMA.
    assert lang_name("de") == "German"
    assert lang_name("is") == "Icelandic"
    assert lang_name("zh") == "Chinese"
    assert lang_name("en") == "English"


def test_unknown_language_lists_the_alternatives() -> None:
    with pytest.raises(KeyError, match="unknown language code"):
        lang_name("xx")


@pytest.mark.parametrize("spec", ["de-en", "en-is", "zh-en"])
def test_valid_directions_round_trip(spec: str) -> None:
    assert str(parse_direction(spec)) == spec


@pytest.mark.parametrize("spec", ["deen", "de-de", "xx-en", "de-", "", "de-en-fr"])
def test_malformed_directions_are_rejected(spec: str) -> None:
    with pytest.raises(ValueError, match="direction"):
        parse_direction(spec)


def test_duplicate_directions_are_rejected() -> None:
    # A duplicated direction would be scored twice and skew a macro average.
    with pytest.raises(ValueError, match="more than once"):
        parse_directions("de-en,en-de,de-en")


def test_directions_accept_string_or_list() -> None:
    from_string = parse_directions("de-en,en-de")
    from_list = parse_directions(["de-en", "en-de"])
    assert from_string == from_list


def test_directions_preserve_order() -> None:
    assert [str(d) for d in parse_directions("en-is,de-en,ru-en")] == ["en-is", "de-en", "ru-en"]


@pytest.mark.parametrize(
    ("target", "expected"),
    [("zh", "zh"), ("ja", "ja-mecab"), ("ko", "ko-mecab"), ("de", "13a"), ("en", "13a")],
)
def test_tokenizer_matches_alma(target: str, expected: str) -> None:
    assert sacrebleu_tokenizer(target) == expected


def test_zh_en_gets_almas_raised_source_length() -> None:
    # ALMA reruns zh-en at 512 because some tokenized Chinese sources exceed 256.
    assert max_source_length_for(parse_direction("zh-en"), 256) == 512
    assert max_source_length_for(parse_direction("de-en"), 256) == 256


def test_source_length_override_never_lowers_the_cap() -> None:
    assert max_source_length_for(parse_direction("zh-en"), 1024) == 1024
