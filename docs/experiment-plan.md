# Experiment plan

What the project is trying to establish, and which command produces each piece
of evidence. This is the document to update when the design changes, and the
one to check a result against before it goes in the report.

## The claim under test

ALMA-7B is one model trained across five language pairs and ten translation
directions. The question is whether it contains smaller subnetworks that still
translate, whether those subnetworks are shared across directions or specific
to them, and how much of the damage a lightweight LoRA repair can undo.

Compression is by structured pruning: FFN intermediate channels, attention
heads, whole layers, or a combination. Structured rather than unstructured
because the units are physically removed, so memory and latency actually fall,
and because a removed unit has an index, which is what makes the overlap
analysis in RQ3 possible at all.

## Research questions and the evidence for each

### RQ1: compressibility and repair

*How far can ALMA be structurally pruned before translation quality goes, does
pair-specific pruning permit more compression than multi-directional pruning,
and how much does LoRA repair recover?*

Three axes, so three sets of runs:

| Axis | Systems |
| --- | --- |
| Sparsity | one system per compression level, all with `compression.nominal_sparsity` set |
| Calibration scope | `pruned_for: multi` against one system per language pair |
| Repair | each pruned system twice, with and without its merged LoRA adapter |

Evidence: the quality table, the efficiency table with its three compression
ratios, the significance table, and the Pareto plots. The unrepaired system is
not optional. Without it, a repaired result cannot separate "pruning did little
damage" from "repair fixed a lot", and those are different findings.

```bash
mnlp-eval run --model configs/models/alma-7b-prune50-multi.yaml \
              --suite configs/suites/alma10-greedy.yaml
mnlp-eval report --baseline alma-7b
```

### RQ2: cross-language robustness

*Does multi-directional pruning hurt some directions more than others, and are
lower-resource directions more vulnerable and harder to repair?*

Evidence: the per-direction columns of the quality table, the per-direction
significance table, and the "Degradation by resource tier" table, which splits
the change against the baseline into high-resource and low-resource means.

The macro average over ten directions is dominated by the eight high-resource
ones. A system that has lost Icelandic entirely still reads as mildly degraded
on the average, which is exactly the result the tier split exists to prevent.

Resource tier is set by the parallel training data ALMA has per language, from
`haoranxu/ALMA-Human-Parallel`: Czech 12,076, German 14,211, Icelandic 2,009,
Russian 15,000, Chinese 15,406. Icelandic is the only low-resource language in
ALMA's set, has roughly a seventh of what the others have, and has no
validation split, so it is both the hardest direction to preserve and the one
with the least data available to repair it.

Behavioural metrics matter more here than anywhere else. Off-target
translation is the classic collapse mode of a compressed multilingual model,
and a macro-averaged COMET score hides it completely. The behaviour table
reports on-target, off-target and unverifiable rates that sum to 100 percent of
segments, so a system that emits nothing cannot score a flattering 0 percent
off-target.

### RQ3: subnetwork specialization and transfer

*How much do pair-specific subnetworks overlap, and how well does a subnetwork
pruned for one pair transfer to the others?*

Two separate measurements that are easy to conflate.

**Overlap** is about which units were kept, and is answered without running the
models at all. Each pruning run writes a subnetwork descriptor listing the
kept indices, and `mnlp-eval overlap` compares them pairwise.

```bash
mnlp-eval overlap --subnetwork-dir subnetworks/ --out reports/
```

The figure to quote is the excess over chance, not the raw Jaccard index. Two
independently chosen 50 percent subnetworks already share about a third of what
they keep, so a Jaccard index of 0.33 is the null result. See
[subnetworks.md](subnetworks.md).

**Transfer** is about behaviour, and needs every subnetwork evaluated on all
ten directions rather than only on the pair it was selected for. The
"Cross-direction transfer" table is that matrix, with the matched cells
bracketed and a specialization figure that is the matched mean minus the
mismatched mean.

Reading it: compare a pair-specific row against the multi-directional row
column by column. Comparing specialization figures across systems is only
meaningful at equal sparsity, because a deeper cut lowers every column at once.

## Systems

| Config | Role |
| --- | --- |
| `alma-7b` | baseline and pruning target, fully fine-tuned |
| `alma-7b-r` | optional upper reference, CPO-refined, not a pruning target |
| `alma-7b-prune<N>-multi` | multi-directional subnetwork at sparsity N |
| `alma-7b-prune<N>-multi-lora` | the same subnetwork after repair |
| `alma-7b-prune<N>-<pair>` | pair-specific subnetwork, one per pair |
| `alma-7b-bnb-nf4`, `alma-7b-bnb-int8` | quantization contrast points |

ALMA-7B rather than ALMA-7B-R because it is fully fine-tuned. A pruning
criterion and a repair adapter both act on dense weights, and ALMA-R's
contrastive preference optimization is a second change in training objective
that would be confounded with the effect of compression.

## Test sets

`haoranxu/WMT22-Test`, all ten directions, 17,491 segments per system:
cs-en 1,448, de-en 1,984, is-en 1,000, ru-en 2,016, zh-en 1,875, en-cs 2,037,
en-de 2,037, en-is 1,000, en-ru 2,037, en-zh 2,037.

The Icelandic configs hold 1,000 segments each because they are the WMT21
Icelandic test set. WMT22 had no Icelandic general-MT task, and ALMA evaluates
Icelandic on WMT21 for that reason, so the dataset name is misleading for two
of its ten configs. The per-direction provenance recorded by the generate
stage is what should be quoted in the report.

`haoranxu/FLORES-200` covers the same ten directions and is the out-of-domain
check. Pruning is calibrated on ALMA's news-domain parallel data, so a
subnetwork that has overfitted its calibration data should show up on
Wikipedia-derived text before it shows up on WMT.

## Suites

| Suite | Decoding | Use |
| --- | --- | --- |
| `alma10-beam5` | beam 5, max 256 | headline table, comparable with ALMA's published numbers |
| `alma10-greedy` | greedy, max 256 | the sparsity sweep and the transfer matrix, roughly 3x cheaper |
| `flores200-10dir-greedy` | greedy, max 256 | out-of-domain check |
| `wmt22-alma-repro` | beam 5, no length bucketing | the one-off baseline reproduction |

Scores from different suites are not comparable, and the report stage refuses
to place runs from two suites in one table. Run the sweep under one suite.

## Calibration and repair data

Both come from `haoranxu/ALMA-Human-Parallel` and both are part of the
experimental design rather than of whoever runs the pruning script. If the
multi-directional and pair-specific subnetworks are calibrated on sets of
different sizes, the comparison between them measures calibration data as much
as pruning criterion.

```bash
mnlp-eval calibration --spec configs/calibration/multi-10dir.yaml
mnlp-eval calibration --spec configs/calibration/pair-de.yaml
```

Each set is balanced by construction, drawn with a per-direction seed so
adding a direction does not re-draw the others, fingerprinted, and checked
against the test sets for contamination. Records carry the exact ALMA prompt,
because a criterion calibrated on bare source sentences is calibrated on a
distribution the model is never asked to run on.

`segments_per_direction` is the same for the multi-directional and the
pair-specific sets, so a pair-specific subnetwork sees a tenth of the segments.
That asymmetry is deliberate: the comparison is between calibrating on one pair
and on ten at a fixed per-direction budget. Equalising the totals instead would
confound scope with calibration size.

## What the pruning side has to produce

For a checkpoint to become a fully reported system, three things:

1. A saved checkpoint, or a loader function reachable through `recipes/`.
2. A model config with its `compression` block filled in. Without
   `pruned_for` the transfer matrix cannot be built, and without
   `nominal_sparsity` the sweep has no x-axis.
3. A subnetwork descriptor, if the overlap analysis is to include it.

See [plugging-in-a-model.md](plugging-in-a-model.md) and
[subnetworks.md](subnetworks.md).

## Order of work

The evaluation layer is complete ahead of the checkpoints, so the sequence is
set by what unblocks whom:

1. Build the calibration sets and commit their fingerprints. Everything
   downstream is calibrated on them, so they should not change afterwards.
2. Run `scripts/reproduce_alma_baseline.sh` on Snellius and record the delta
   against ALMA's published table. Until that number exists, every compression
   result rests on an unverified harness. This is the first job to submit.
3. Generate and score the `alma-7b` baseline under `alma10-greedy`, which is
   the reference every ratio, p-value and transfer figure is computed against.
4. Sweep sparsity levels for the multi-directional subnetwork.
5. Add the five pair-specific subnetworks at the sparsity level the sweep
   picks out, then the overlap analysis and the transfer matrix.
6. Repair the subnetworks that are worth repairing and re-evaluate.

Steps 3 to 6 are the expensive ones. At beam 5 the ten-direction suite is
roughly 6 to 10 A100-hours per system, and greedy is roughly a third of that,
which is why the sweep uses greedy and only the final systems are re-run at
beam 5.
