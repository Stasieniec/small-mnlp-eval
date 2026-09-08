"""sacreBLEU metrics, with signatures recorded for reproducibility.

Signatures are not decoration. BLEU shifts between sacreBLEU releases and
between tokenizers, so a score without its signature cannot be compared with a
score from anywhere else, including ALMA's published tables.
"""

from __future__ import annotations

from collections.abc import Sequence

from mnlp_eval.languages import sacrebleu_tokenizer
from mnlp_eval.metrics.base import MetricScore

__all__ = ["score_surface"]


def score_surface(
    hypotheses: Sequence[str],
    references: Sequence[str],
    target_language: str,
    *,
    compute_ter: bool = False,
) -> dict[str, MetricScore]:
    """Compute BLEU, chrF++ and optionally TER for one direction.

    The tokenizer follows ALMA's ``evals/eval_generation.sh``: ``zh`` for
    Chinese targets, ``ja-mecab`` for Japanese, ``13a`` otherwise.
    """
    from sacrebleu.metrics import BLEU, CHRF, TER

    if len(hypotheses) != len(references):
        msg = f"{len(hypotheses)} hypotheses but {len(references)} references"
        raise ValueError(msg)

    tokenizer = sacrebleu_tokenizer(target_language)
    scores: dict[str, MetricScore] = {}

    bleu = BLEU(tokenize=tokenizer)
    bleu_result = bleu.corpus_score(list(hypotheses), [list(references)])
    # get_signature() only works after scoring, since the signature records the
    # number of references it actually saw.
    scores["bleu"] = MetricScore(
        name="bleu",
        score=round(bleu_result.score, 4),
        signature=bleu.get_signature().format(),
        extra={
            "precisions": [round(value, 4) for value in bleu_result.precisions],
            "brevity_penalty": round(bleu_result.bp, 4),
            "length_ratio": round(bleu_result.sys_len / bleu_result.ref_len, 4)
            if bleu_result.ref_len
            else None,
            "hypothesis_length": bleu_result.sys_len,
            "reference_length": bleu_result.ref_len,
            "tokenizer": tokenizer,
        },
    )

    chrf = CHRF(word_order=2)
    chrf_result = chrf.corpus_score(list(hypotheses), [list(references)])
    scores["chrf2pp"] = MetricScore(
        name="chrf2pp",
        score=round(chrf_result.score, 4),
        signature=chrf.get_signature().format(),
    )

    if compute_ter:
        ter = TER()
        ter_result = ter.corpus_score(list(hypotheses), [list(references)])
        scores["ter"] = MetricScore(
            name="ter",
            score=round(ter_result.score, 4),
            signature=ter.get_signature().format(),
            higher_is_better=False,
        )

    return scores
