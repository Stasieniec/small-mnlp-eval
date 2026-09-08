#!/usr/bin/env python3
"""Fail if tracked text files contain emoji or em-dashes.

The project rules that this repository contains no emoji and no em-dashes are
enforced here rather than left to review, because these characters arrive
easily through copy-paste and generated text, and a reviewer will not reliably
spot one in a 600 line diff.

Binary files are skipped by attempting a UTF-8 decode. En-dashes are not
checked: they appear legitimately in numeric ranges.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import unicodedata
from pathlib import Path

# Codepoints banned outright.
BANNED_CHARACTERS: dict[str, str] = {
    "—": "em-dash",
    "―": "horizontal bar",
}

# Inclusive codepoint ranges treated as emoji or decorative pictographs.
# Deliberately excludes arrows and mathematical symbols, which have
# legitimate uses in prose about metrics.
EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F000, 0x1FAFF),  # emoticons, pictographs, transport, flags, symbols
    (0x2600, 0x27BF),  # miscellaneous symbols and dingbats
    (0x2B00, 0x2BFF),  # miscellaneous symbols and arrows
    (0xFE0F, 0xFE0F),  # variation selector 16, the emoji presentation marker
    (0x1F3FB, 0x1F3FF),  # skin tone modifiers
)


def is_emoji(character: str) -> bool:
    codepoint = ord(character)
    return any(low <= codepoint <= high for low, high in EMOJI_RANGES)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return sorted(
            path for path in Path().rglob("*") if path.is_file() and ".git" not in path.parts
        )
    return [Path(line) for line in result.stdout.splitlines() if line]


def describe(character: str) -> str:
    label = BANNED_CHARACTERS.get(character)
    if label:
        return label
    try:
        return unicodedata.name(character).lower()
    except ValueError:
        return "unnamed character"


def check(paths: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable, nothing to check
        if path.name == "check_style.py":
            continue  # this file necessarily names the characters it bans
        for line_number, line in enumerate(text.splitlines(), start=1):
            for column, character in enumerate(line, start=1):
                if character in BANNED_CHARACTERS or is_emoji(character):
                    problems.append(
                        f"{path}:{line_number}:{column}: "
                        f"{describe(character)} (U+{ord(character):04X})"
                    )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="files to check; default is every tracked file")
    args = parser.parse_args(argv)

    paths = [Path(item) for item in args.paths] if args.paths else tracked_files()
    problems = check(paths)
    if problems:
        print(f"{len(problems)} style violation(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nThis repository contains no emoji and no em-dashes. "
            "Replace an em-dash with a comma, a colon, or two sentences.",
            file=sys.stderr,
        )
        return 1
    print(f"style check passed on {len(paths)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
