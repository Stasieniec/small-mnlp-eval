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

# Dash-family codepoints banned outright. Every codepoint whose Unicode name
# contains "em dash", plus the horizontal bar. Listing only U+2014 and U+2015
# let four of them through, all of which render as an em-dash.
BANNED_CHARACTERS: dict[str, str] = {
    "—": "em-dash",
    "―": "horizontal bar",
    "⸺": "two-em dash",
    "⸻": "three-em dash",
    "︱": "vertical em-dash",
    "︲": "vertical en-dash",
    "﹘": "small em-dash",
}

# Inclusive codepoint ranges treated as emoji or decorative pictographs.
#
# These blocks contain a few characters with legitimate typographic uses, tick
# and cross marks among them. They are banned anyway: the rule is that this
# repository carries no decorative symbols, and a rule with exceptions is one
# nobody can apply without consulting a table. Mathematical operators, arrows
# below U+2190 and box drawing are outside these ranges and stay allowed.
EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x203C, 0x203C),  # double exclamation mark
    (0x2049, 0x2049),  # exclamation question mark
    (0x2122, 0x2122),  # trade mark sign
    (0x2139, 0x2139),  # information source
    (0x231A, 0x231B),  # watch, hourglass
    (0x2328, 0x2328),  # keyboard
    (0x23CF, 0x23CF),  # eject
    (0x23E9, 0x23F3),  # media controls, timers
    (0x23F8, 0x23FA),  # pause, stop, record
    (0x24C2, 0x24C2),  # circled M
    (0x25AA, 0x25AB),  # small squares used as emoji
    (0x25B6, 0x25B6),  # play
    (0x25C0, 0x25C0),  # reverse play
    (0x25FB, 0x25FE),  # medium squares used as emoji
    (0x2600, 0x27BF),  # miscellaneous symbols and dingbats
    (0x2B00, 0x2BFF),  # miscellaneous symbols and arrows
    (0x3030, 0x3030),  # wavy dash
    (0x303D, 0x303D),  # part alternation mark
    (0x3297, 0x3299),  # circled ideographs used as emoji
    (0xFE0F, 0xFE0F),  # variation selector 16, the emoji presentation marker
    (0x1F000, 0x1FAFF),  # emoticons, pictographs, transport, flags, symbols
)

#: Directories never walked when git is unavailable. Without this the fallback
#: walked .venv and reported hundreds of violations from site-packages.
SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        ".venv-comet",
        ".venv-metricx",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "build",
        "dist",
    }
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
    if result.returncode == 0:
        return [Path(line) for line in result.stdout.splitlines() if line]
    return sorted(
        path
        for path in Path().rglob("*")
        if path.is_file() and not SKIP_DIRECTORIES.intersection(path.parts)
    )


def expand(paths: list[Path]) -> tuple[list[Path], list[str]]:
    """Resolve the requested paths into files, reporting anything unusable.

    A missing path used to be skipped in silence, so a typo'd argument printed
    "style check passed", and a directory argument checked nothing inside it.
    """
    files: list[Path] = []
    problems: list[str] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(
                    item
                    for item in path.rglob("*")
                    if item.is_file() and not SKIP_DIRECTORIES.intersection(item.parts)
                )
            )
        elif path.is_file():
            files.append(path)
        else:
            problems.append(f"{path}: no such file or directory")
    return files, problems


def describe(character: str) -> str:
    label = BANNED_CHARACTERS.get(character)
    if label:
        return label
    try:
        return unicodedata.name(character).lower()
    except ValueError:
        return "unnamed character"


def check(paths: list[Path]) -> tuple[list[str], int]:
    problems: list[str] = []
    skipped = 0
    for path in paths:
        if not path.is_file():
            continue
        if path.name == "check_style.py":
            continue  # this file necessarily names the characters it bans
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Binary or not UTF-8. Counted rather than ignored, because a
            # cp1252 file can hold an em-dash this cannot see.
            skipped += 1
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for column, character in enumerate(line, start=1):
                if character in BANNED_CHARACTERS or is_emoji(character):
                    problems.append(
                        f"{path}:{line_number}:{column}: "
                        f"{describe(character)} (U+{ord(character):04X})"
                    )
    return problems, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="files to check; default is every tracked file")
    args = parser.parse_args(argv)

    requested = [Path(item) for item in args.paths] if args.paths else tracked_files()
    paths, unusable = expand(requested)
    problems, skipped = check(paths)

    for problem in unusable:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        print(f"{len(problems)} style violation(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nThis repository contains no emoji and no em-dashes. "
            "Replace an em-dash with a comma, a colon, or two sentences.",
            file=sys.stderr,
        )
    if problems or unusable:
        return 1

    note = f", {skipped} binary or non-UTF-8 file(s) skipped" if skipped else ""
    print(f"style check passed on {len(paths)} file(s){note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
