# The experiment

A project on structured pruning of
[ALMA-7B](https://github.com/fe1ixxu/ALMA): whether it contains smaller
multi-directional or pair-specific subnetworks that still translate, how that
interacts with resource level across its ten directions, and how much a LoRA
repair recovers.

| | |
| --- | --- |
| Baseline and pruning target | `haoranxu/ALMA-7B`, fully fine-tuned |
| Directions | all ten ALMA supports: English against cs, de, is, ru, zh |
| Test sets | `haoranxu/WMT22-Test` (17,491 segments; WMT21 for Icelandic) |
| Calibration and repair data | `haoranxu/ALMA-Human-Parallel` |
| Quality | BLEU, chrF++, COMET-22, COMETKiwi, XCOMET-XL, MetricX-24 |
| Efficiency | size, FLOPs, MFU, latency, throughput, three compression ratios |

## Decisions not recoverable from the code

**ALMA-7B rather than ALMA-7B-R.** The pruning criterion and the repair adapter
both act on dense weights. ALMA-R adds contrastive preference optimization, a
second change in training objective that would be confounded with the effect
of compression. It stays available as an upper reference.

**The Icelandic test set is WMT21.** The `is-en` and `en-is` configs of
`haoranxu/WMT22-Test` hold 1,000 segments each because WMT22 had no Icelandic
general-MT task and ALMA evaluates Icelandic on WMT21. The dataset name is
misleading for two of its ten configs; quote the per-direction provenance the
generate stage records.

**Calibration is balanced, not proportional.** Icelandic has 2,009 parallel
training pairs against Russian's 15,000. A proportional draw would leave the
multi-directional calibration set nearly free of the language most likely to
break. `segments_per_direction` is also the same for the pair-specific sets, so
the comparison is scope at a fixed per-direction budget rather than scope
confounded with calibration size.

**FLORES-200 is not a test set for ALMA.** ALMA-Human-Parallel, ALMA's
fine-tuning data and the pool every calibration and repair set here is drawn
from, contains the FLORES-200 sentences: all 1,012 test sources of de-en,
is-en and zh-en appear in it (checked 8 October 2026). ALMA-7B was trained on
them, and a calibration draw can include them, so FLORES scores measure
memorisation, not out-of-domain translation. `configs/suites/flores200-10dir-greedy.yaml`
stays only for completeness.

## What the report contains

Quality per direction and macro-averaged, with paired bootstrap significance
against the baseline. Behavioural failure rates: off-target language, empty,
source copy, repetition, truncation, length ratio. Efficiency with separate
compression ratios for disk bytes, parameter count and resident VRAM, because
they diverge. Per-layer structure and the share of weights that are exactly
zero, which is how a mask that was applied but never compacted shows up.
Degradation split by resource tier, and a cross-direction transfer matrix for
pair-specific subnetworks.

[protocol.md](protocol.md) records what each of those means and how it is
measured.

## Verification

- `./scripts/smoke_local.sh` runs all four evaluation stages and asserts the
  results are plausible, so a broken pipeline cannot pass as a bad model.
- `mnlp-eval verify-testset` checks the Hub test sets against ALMA's committed
  `human_written_data` files segment by segment.
- `./scripts/reproduce_alma_baseline.sh` reproduces ALMA-7B on all ten
  directions for comparison against the published table. Until that delta is
  known, every result here rests on an unverified harness.
- `pytest tests/` needs no GPU, network or weights.
