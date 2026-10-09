# Full-suite runs started on 6 October 2026

Superseded by the complete full-suite evaluation of 8 October 2026:
[../full-2026-10-08/README.md](../full-2026-10-08/README.md).

`configs/suites/alma10-greedy.yaml` (all of WMT22, WMT21 for Icelandic, greedy),
through `slurm/submit_eval.sh`. Prerequisites for the full evaluation of the chosen
systems, not the choice itself.

| system | directions | COMET-22 | BLEU | chrF++ |
| --- | --- | --- | --- | --- |
| alma-7b (dense) | all ten, 17,491 segments | 0.8475 | 30.32 | 51.13 |
| alma-7b-slimgpt40-gen-pair-is-lora | is-en / en-is | 0.8367 / 0.8137 | 28.34 / 18.73 | |

The pair model is scored on its own two directions only (`--allow-partial`).
