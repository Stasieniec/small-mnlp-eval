"""Language code tables and language-dependent evaluation settings.

The tables here are deliberately identical to ``utils/utils.py`` in
``fe1ixxu/ALMA``. Reproducing ALMA's published scores requires the same
language names in the prompt, so this module is a compatibility surface and
should not be "improved" without a matching change to the prompt hash.
"""

from __future__ import annotations

from typing import Final, NamedTuple

__all__ = [
    "LANG_TABLE",
    "NLLB_CODE",
    "Direction",
    "lang_name",
    "max_source_length_for",
    "parse_direction",
    "parse_directions",
    "sacrebleu_tokenizer",
]

# ALMA's language name table. The prompt embeds these names verbatim, so the
# exact spelling is part of the evaluation protocol.
LANG_TABLE: Final[dict[str, str]] = {
    "en": "English",
    # Group 1
    "da": "Danish",
    "nl": "Dutch",
    "de": "German",
    "is": "Icelandic",
    "no": "Norwegian",
    "sv": "Swedish",
    "af": "Afrikaans",
    # Group 2
    "ca": "Catalan",
    "ro": "Romanian",
    "gl": "Galician",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
    # Group 3
    "bg": "Bulgarian",
    "mk": "Macedonian",
    "sr": "Serbian",
    "uk": "Ukrainian",
    "ru": "Russian",
    # Group 4
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
    "mg": "Malagasy",
    "fr": "French",
    # Group 5
    "hu": "Hungarian",
    "el": "Greek",
    "cs": "Czech",
    "pl": "Polish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    # Group 6
    "ka": "Georgian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fi": "Finnish",
    "et": "Estonian",
    # Group 7
    "gu": "Gujarati",
    "hi": "Hindi",
    "mr": "Marathi",
    "ne": "Nepali",
    "ur": "Urdu",
    # Group 8
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "tr": "Turkish",
    "uz": "Uzbek",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
}

# FLORES-200 style codes, used by the NLLB baseline loader to set
# ``forced_bos_token_id``. Also taken verbatim from ALMA.
NLLB_CODE: Final[dict[str, str]] = {
    "en": "eng_Latn",
    "da": "dan_Latn",
    "nl": "nld_Latn",
    "de": "deu_Latn",
    "is": "isl_Latn",
    "no": "nob_Latn",
    "sv": "swe_Latn",
    "af": "afr_Latn",
    "ca": "cat_Latn",
    "ro": "ron_Latn",
    "gl": "glg_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "es": "spa_Latn",
    "bg": "bul_Cyrl",
    "mk": "mkd_Cyrl",
    "sr": "srp_Cyrl",
    "uk": "ukr_Cyrl",
    "ru": "rus_Cyrl",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "mg": "plt_Latn",
    "fr": "fra_Latn",
    "hu": "hun_Latn",
    "el": "ell_Grek",
    "cs": "ces_Latn",
    "pl": "pol_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "ka": "kat_Geor",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fi": "fin_Latn",
    "et": "est_Latn",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "ur": "urd_Arab",
    "az": "azj_Latn",
    "kk": "kaz_Cyrl",
    "ky": "kir_Cyrl",
    "tr": "tur_Latn",
    "uz": "uzn_Latn",
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
}

# sacreBLEU applies a language-specific tokenizer for these target languages.
# Every other target uses the default ``13a``, which mimics mteval-v13a.
#
# ALMA's evals/eval_generation.sh overrides only zh and ja. Korean is included
# here because 13a is as wrong for Korean as it is for Japanese, and no ALMA
# suite evaluates Korean, so the deviation cannot affect a reproduction. It is
# a deliberate departure from upstream rather than an oversight.
#
# The ja and ko tokenizers need MeCab, which the ``surface`` extra does not
# install: sacreBLEU raises a clear error naming ``sacrebleu[ja]`` or
# ``sacrebleu[ko]``. Add those extras before evaluating either language.
_TOKENIZER_BY_TARGET: Final[dict[str, str]] = {
    "zh": "zh",
    "ja": "ja-mecab",
    "ko": "ko-mecab",
}

DEFAULT_TOKENIZER: Final[str] = "13a"

# ALMA raises the source-length cap for zh-en only, because some tokenized
# Chinese sources exceed 256 tokens. See evals/alma_7b.sh in fe1ixxu/ALMA.
_SOURCE_LENGTH_OVERRIDES: Final[dict[str, int]] = {"zh-en": 512}


class Direction(NamedTuple):
    """A single translation direction, for example German into English."""

    source: str
    target: str

    def __str__(self) -> str:
        return f"{self.source}-{self.target}"

    @property
    def source_name(self) -> str:
        return lang_name(self.source)

    @property
    def target_name(self) -> str:
        return lang_name(self.target)


def lang_name(code: str) -> str:
    """Return ALMA's English name for a two-letter language code."""
    try:
        return LANG_TABLE[code]
    except KeyError:
        known = ", ".join(sorted(LANG_TABLE))
        msg = f"unknown language code {code!r}; known codes are: {known}"
        raise KeyError(msg) from None


def parse_direction(spec: str) -> Direction:
    """Parse ``"de-en"`` into a validated :class:`Direction`."""
    parts = spec.strip().split("-")
    if len(parts) != 2 or not all(parts):
        msg = f"malformed direction {spec!r}; expected the form 'de-en'"
        raise ValueError(msg)
    source, target = parts
    for code in (source, target):
        if code not in LANG_TABLE:
            msg = f"unknown language code {code!r} in direction {spec!r}"
            raise ValueError(msg)
    if source == target:
        msg = f"direction {spec!r} has the same source and target language"
        raise ValueError(msg)
    return Direction(source, target)


def parse_directions(specs: str | list[str]) -> list[Direction]:
    """Parse a comma-separated string or a list into validated directions.

    Order is preserved and duplicates are rejected, so a suite cannot silently
    evaluate the same direction twice and skew a macro average.
    """
    items = specs.split(",") if isinstance(specs, str) else list(specs)
    directions = [parse_direction(item) for item in items if item.strip()]
    if not directions:
        msg = "no translation directions given"
        raise ValueError(msg)
    seen: set[str] = set()
    for direction in directions:
        key = str(direction)
        if key in seen:
            msg = f"direction {key} is listed more than once"
            raise ValueError(msg)
        seen.add(key)
    return directions


def sacrebleu_tokenizer(target: str) -> str:
    """Return the sacreBLEU tokenizer name for a target language.

    Matches ALMA's ``evals/eval_generation.sh``: ``zh`` for Chinese,
    ``ja-mecab`` for Japanese, ``13a`` otherwise. Korean is included for
    completeness even though ALMA never evaluates it.
    """
    return _TOKENIZER_BY_TARGET.get(target, DEFAULT_TOKENIZER)


def max_source_length_for(direction: Direction, default: int) -> int:
    """Return the source-length cap ALMA uses for a direction.

    ALMA evaluates every direction at 256 source tokens except zh-en, which it
    reruns at 512. Encoding the quirk here keeps suite configs honest.
    """
    override = _SOURCE_LENGTH_OVERRIDES.get(str(direction))
    if override is None:
        return default
    return max(default, override)
