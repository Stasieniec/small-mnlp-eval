# Direction-specific pruning pilot - 5 October 2026

Completed ALMA-7B / FLAP pilot, calibration seed **1234**, 20% nominal pruning,
uniform allocation, prompt+reference calibration, no repair. Eight models were
evaluated on the same first 200 segments in each of de-en, en-de, is-en and en-is
(6,400 translations), with greedy decoding and a 256-token generation limit.
All eight GPU jobs and the report job completed successfully on H100 GPUs.

## Results

- [Statistical comparison](reports/comparison.md): all eight primary contrasts,
  paired bootstrap confidence intervals and Holm-adjusted p-values.
- [All model metrics](reports/systems.csv): quality and behavioural failure rates.
- [Per-direction quality](reports/by-direction.csv).
- [Full statistical results](reports/comparisons.json).
- [Pruning budget audit](reports/budget-audit.json): all seven pruned models retain
  5,470,162,944 parameters, removing 192 heads and 70,464 FFN channels.

Direction-only calibration improves English to Icelandic over multilingual
calibration (COMET +0.05720, Holm p=0.0480). Its advantage over pair calibration
is not significant after correction (+0.04519, p=0.1259). Icelandic to English
gets worse against both multilingual (-0.09627, p=0.0040) and pair calibration
(-0.06114, p=0.0315). Neither German direction shows a significant difference
against either control. This does not support direction-only calibration as a
general default.

**Absolute quality is poor for every pruned model.** Macro COMET ranges from
0.5248 to 0.5956, versus 0.8478 for dense ALMA; BLEU ranges from 3.24 to 4.85,
versus 30.17. Repetition rates are 29.87%-42.25%, and truncation rates are
31.62%-43.13%, versus zero for dense ALMA. Relative gains therefore do not
establish usable translation quality. Investigate these failures before
adopting a scope, and confirm promising contrasts with another seed.

## Limitations

One calibration seed; segment bootstrap measures evaluation uncertainty, not
calibration variability. All pruned models use 1,280 calibration segments,
but token budgets differ. Evaluation uses the first segments rather than a
random sample. Other-direction transfer results are exploratory. This pilot
has no repair or inference benchmark and is not directly comparable to the
300-segment, ten-direction SlimGPT layer-protection experiment.

## Archived evidence

`provenance/` preserves the original experiment manifest with source/input
hashes, calibration fingerprints, evaluation provenance, final job IDs and
resolved configs. Absolute paths refer to the original Snellius run and need
adapting for reproduction; see [experiment instructions](../../docs/direction-scope.md).

`runs/` contains sentence-level COMET scores, surface scores, manifests,
runtime environments and stage metadata for all eight models. There are
6,400 sentence-level COMET scores. `segment-alignment.json` preserves original
indices and SHA-256 hashes of exact UTF-8 source/reference strings; alignment
was verified equal across all models. Pair scores by direction and array
position, and weight the four direction means equally for macro COMET.

Checkpoints, translations, calibration/evaluation text and Slurm logs remain
on Snellius at `/scratch-shared/scur0535/direction-scope-seed1234/`.
This compact archive supports score-based analysis; recomputing text-based
metrics or rerunning the report generator requires the original text artifacts.
Scratch storage is temporary; preserve large artifacts separately as needed.

Verify archive integrity from this directory:

```bash
sha256sum -c SHA256SUMS
```
