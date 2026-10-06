"""How far the dense model's own translations are from the references it replaces.

Reads every generated-calibration cache under data/calibration/*-generated and
reports, per cache and direction, chrF++ and BLEU of the cached greedy
translation against the record's reference, the share of exact matches,
empty outputs and outputs that hit the token budget, and token counts.

ALMA-7B was fine-tuned on these very pairs, so its greedy output may be close
to the reference. If it is, a reference-versus-generated calibration contrast
is small by construction, and this table is what says so.

    ./.venv/bin/python scripts/calibration_diagnostics.py --out results/grid-2026-10-06
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/calibration")
    parser.add_argument("--out", default="results/grid-2026-10-06")
    args = parser.parse_args()

    from sacrebleu.metrics import BLEU, CHRF

    chrf = CHRF(word_order=2)
    rows = []
    for cache in sorted(Path(args.root).glob("*-generated")):
        for path in sorted(cache.glob("*.jsonl")):
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            direction = path.stem
            target = direction.split("-")[1]
            bleu = BLEU(tokenize="zh" if target == "zh" else "13a")
            hyps = [str(record.get("hypothesis") or "") for record in records]
            refs = [str(record["target"]) for record in records]
            rows.append(
                {
                    "cache": cache.name,
                    "direction": direction,
                    "n": len(records),
                    "chrf++": round(chrf.corpus_score(hyps, [refs]).score, 2),
                    "bleu": round(bleu.corpus_score(hyps, [refs]).score, 2),
                    "exact_match": round(
                        sum(h.strip() == r.strip() for h, r in zip(hyps, refs, strict=True))
                        / len(records),
                        4,
                    ),
                    "empty": sum(not h.strip() for h in hyps),
                    "hit_token_budget": sum(bool(record["hit_token_budget"]) for record in records),
                    "source_truncated": sum(bool(record["source_truncated"]) for record in records),
                    "generated_tokens": sum(
                        len(record["generated_token_ids"]) for record in records
                    ),
                    "input_tokens": sum(len(record["input_token_ids"]) for record in records),
                }
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration_diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    columns = list(rows[0]) if rows else []
    lines = [
        "# Generated calibration against the references",
        "",
        "Dense ALMA-7B greedy translations of the calibration sources, scored against",
        "the reference each record carries. High agreement means the reference and",
        "generated calibration arms see similar target text.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows]
    (out / "calibration_diagnostics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
