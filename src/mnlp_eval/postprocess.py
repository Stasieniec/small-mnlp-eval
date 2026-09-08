"""Turn raw model output into a single hypothesis, with an audit trail.

ALMA's ``clean_outputstring`` recovers a translation by splitting the full
decoded sequence on the target-language cue inside three nested ``try`` blocks,
and returns an empty string when all of them fail. For a project whose subject
is degradation that is the worst possible behaviour: a collapsed model scores a
legitimate-looking zero and nobody notices.

Here, extraction operates on the generated continuation only (the caller slices
the prompt off by input length), and every hypothesis carries flags describing
what had to be done to recover it. The counts surface in the behavioural
metrics, so parsing trouble is reported rather than absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "FLAG_EXTRA_LINES",
    "FLAG_REPETITION",
    "FLAG_SKIPPED_BLANK",
    "FLAG_STRIPPED_MARKER",
    "STATUS_EMPTY",
    "STATUS_OK",
    "Parsed",
    "alma_legacy_clean",
    "detect_repetition",
    "extract_hypothesis",
]

STATUS_OK: Final = "ok"
STATUS_EMPTY: Final = "empty"

#: Blank lines were skipped before a non-empty line was found.
FLAG_SKIPPED_BLANK: Final = "skipped_blank"
#: The model echoed the target-language cue, for example a leading "English:".
FLAG_STRIPPED_MARKER: Final = "stripped_marker"
#: Non-empty lines after the hypothesis were discarded.
FLAG_EXTRA_LINES: Final = "extra_lines"
#: The output ends in a repeated n-gram, a common collapse mode.
FLAG_REPETITION: Final = "repetition"


@dataclass(frozen=True)
class Parsed:
    """The result of extracting one hypothesis from raw model output."""

    hypothesis: str
    status: str = STATUS_OK
    flags: tuple[str, ...] = field(default=())

    @property
    def is_empty(self) -> bool:
        return self.status == STATUS_EMPTY


def extract_hypothesis(
    raw: str,
    target_marker: str = "",
    *,
    detect_repeats: bool = True,
) -> Parsed:
    """Extract a single-line hypothesis from a generated continuation.

    Args:
        raw: The decoded continuation, containing only newly generated tokens.
        target_marker: The template's trailing cue, for example ``"English:"``.
            Stripped if the model echoes it back.
        detect_repeats: Whether to flag degenerate trailing repetition.

    Returns:
        A :class:`Parsed` whose ``flags`` record every recovery step taken.
    """
    flags: list[str] = []
    lines = raw.split("\n")
    hypothesis = ""
    consumed = 0

    for index, line in enumerate(lines):
        candidate = line.strip()
        if not candidate:
            continue
        if target_marker and candidate.startswith(target_marker):
            candidate = candidate[len(target_marker) :].strip()
            if FLAG_STRIPPED_MARKER not in flags:
                flags.append(FLAG_STRIPPED_MARKER)
            if not candidate:
                continue
        hypothesis = candidate
        consumed = index + 1
        break

    if not hypothesis:
        return Parsed("", STATUS_EMPTY, tuple(flags))

    if any(not line.strip() for line in lines[: consumed - 1]):
        flags.append(FLAG_SKIPPED_BLANK)
    if any(line.strip() for line in lines[consumed:]):
        flags.append(FLAG_EXTRA_LINES)
    if detect_repeats and detect_repetition(hypothesis) is not None:
        flags.append(FLAG_REPETITION)

    return Parsed(hypothesis, STATUS_OK, tuple(flags))


def detect_repetition(
    text: str,
    *,
    min_repeats: int = 4,
    max_unit_tokens: int = 8,
) -> str | None:
    """Return the repeated unit if ``text`` ends in a repeated n-gram.

    Detects the classic collapse mode of aggressively compressed models, where
    generation degenerates into a cycle. Returns the shortest repeating unit,
    or ``None`` when the tail looks healthy.
    """
    tokens = text.split()
    if min_repeats < 2 or len(tokens) < min_repeats:
        return None
    for size in range(1, max_unit_tokens + 1):
        span = size * min_repeats
        if span > len(tokens):
            break
        tail = tokens[-span:]
        unit = tail[:size]
        if all(tail[i : i + size] == unit for i in range(0, span, size)):
            return " ".join(unit)
    return None


def alma_legacy_clean(full_output: str, key_word: str, split_idx: int = 1) -> str:
    """Faithful reimplementation of ALMA's ``clean_outputstring``.

    Kept so the framework can quantify how often upstream parsing would have
    silently produced an empty string on the same outputs. Not used for
    scoring. ``full_output`` must be the whole decoded sequence including the
    prompt, and ``key_word`` the suffix such as ``"\\nEnglish:"``.
    """
    try:
        parts = full_output.split(key_word)[split_idx].split("\n")
        for candidate in parts[:3]:
            if candidate.strip():
                return candidate.strip()
    except IndexError:
        pass
    try:
        return full_output.split(key_word)[2].split("\n")[0].strip()
    except IndexError:
        return ""
