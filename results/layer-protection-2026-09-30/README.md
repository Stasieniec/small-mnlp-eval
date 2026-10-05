# Layer protection experiment — 30 September 2026

Completed ALMA-7B / SlimGPT comparison, calibration seed **1234**. Evaluation:
300 sentences per direction across 10 translation directions, greedy decoding,
20% nominal pruning, prompt+target calibration, no repair.

## Start here

- [Statistical comparison](reports/comparison.md): protection versus unprotected uniform pruning.
- [All model metrics](reports/systems.csv): quality, size, behaviour, and efficiency.
- [Readable tables versus the control](reports/against-control/report.md).
- [Readable tables versus dense ALMA](reports/against-dense/report.md).
- [Per-direction differences](reports/per-direction-deltas.csv).
- [Full statistical results](reports/comparisons.json).
- [Budget audit](reports/budget-audit.json): all five uniform conditions have exactly matching budgets.

The table folders also include CSV and LaTeX exports for analysis and writing.

## Findings

| Protected layers | COMET | Delta versus control | Holm-adjusted p | BLEU | chrF++ |
| --- | ---: | ---: | ---: | ---: | ---: |
| None (uniform control) | 0.8344 | — | — | 27.60 | 48.86 |
| First 4 | 0.8362 | +0.00180 | 0.1439 | 27.59 | 48.91 |
| Last 2 | 0.8349 | +0.00055 | 0.5265 | 28.35 | 49.43 |
| First 4 + last 2 | 0.8384 | +0.00404 | 0.0040 | 28.12 | 49.31 |
| First 3 + last 1 | 0.8367 | +0.00235 | 0.0599 | 27.98 | 49.13 |
| Global pruning reference | 0.8367 | — | — | 28.33 | 49.33 |
| Dense reference | 0.8474 | — | — | 30.35 | 51.14 |

First-4/last-2 is the leading candidate on the primary metric, COMET. Its
improvement over control has a pointwise 95% CI of [+0.00200, +0.00621] and
remains significant after correcting the four comparisons. It improves COMET
in all ten directions and recovers about 31% of the dense-to-control COMET gap.
Last-2 has the best BLEU/chrF++ among the uniform pruned models.

All five uniform conditions retain 5,469,900,800 parameters and remove 192 heads
and 70,464 FFN channels. The global reference has a different budget and is a
contextual comparison. Pruning reduces model size; these measurements do not
show a clear inference speedup over dense ALMA.

**Limitations:** one calibration seed; uncertainty covers evaluation segments,
not calibration variability. Per-direction tests are exploratory and unadjusted.
Significance versus control does not establish superiority over every other
mask. Repeat across calibration seeds before adopting a default.

## Provenance and reproduction

See [experiment instructions](../../docs/layer-protection.md) for the method and
submission commands. The `provenance/` folder preserves the original experiment
manifest (seed, systems, calibration fingerprint, and code/input SHA-256 hashes),
evaluation provenance, software versions, Slurm job IDs, and resolved configs.
The captured Git revision may precede the experiment implementation: the run
used a source snapshot including uncommitted changes. The manifest's source
hashes are the record of the exact files used, not the Git revision alone.

Reports and resolved configs are copied unchanged. Absolute paths in them point
to the original Snellius run; they are provenance, not portable rerun paths.
Create a new experiment using the documented launcher when reproducing it.

Original complete run on Snellius:
`/scratch-shared/scur0535/layer-protection-seed1234-retry1/`

## Data for follow-up analysis

The [runs/](runs/) folder contains one directory per model, retaining the original
run names and IDs. Each includes:

- `scores.neural.json`: all 300 sentence-level COMET scores per direction, metric
  settings and versions, direction means and aggregates.
- `scores.surface.json`: BLEU/chrF++ scores and metric signatures.
- `bench.json`: individual timing repeats (five per batch size), memory usage,
  throughput, benchmark protocol, device information and model structure.
- `manifest.json`, `env.json`, `stages/`: run identity, configuration, runtime
  environment, input fingerprints and stage completion metadata.
- `segment-alignment.json`: original segment indices and SHA-256 hashes of source
  and reference strings, in the same order as the COMET score arrays. These were
  verified equal across all seven models, enabling paired COMET analysis without
  including dataset text. Hashes use exact UTF-8 strings without normalization.

There are 21,000 sentence-level COMET scores (7 models × 10 directions × 300
segments). Use direction plus array position to pair scores across models;
`segment-alignment.json` provides the corresponding original index and text
hashes. For macro COMET, weight the ten direction means equally. The report's
bootstrap resamples within each direction and pairs the same positions between
models; preserve that design when recomputing comparisons.

Model checkpoints, translation text, calibration/evaluation text, Slurm logs,
and the full source snapshot remain on Snellius. Recomputing text-based metrics
such as BLEU/chrF++ or inspecting translation errors still requires those text
artifacts. The report generator expects the original hypotheses and cannot run
unchanged on this reduced archive. Preserve large artifacts separately if needed;
scratch storage is not a permanent archive.

`SHA256SUMS` checks the integrity of every other file in this bundle:

```bash
cd results/layer-protection-2026-09-30
sha256sum -c SHA256SUMS
```
