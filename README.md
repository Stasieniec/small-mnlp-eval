# small-mnlp-eval

Quality and efficiency evaluation for compressed machine translation LLMs.

The evaluation layer for a project on structured pruning of
[ALMA-7B](https://github.com/fe1ixxu/ALMA): whether it contains smaller
multi-directional or pair-specific subnetworks that still translate, how that
interacts with resource level across its ten directions, and how much a LoRA
repair recovers. Pruning happens elsewhere. This repository produces the
calibration data it runs on and every number the report quotes.

A compressed checkpoint becomes a fully evaluated system by adding one YAML
file, and two systems can only appear in the same table if they were measured
the same way.

## The experiment

| | |
| --- | --- |
| Baseline and pruning target | `haoranxu/ALMA-7B`, fully fine-tuned |
| Directions | all ten ALMA supports: English against cs, de, is, ru, zh |
| Test sets | `haoranxu/WMT22-Test` (17,491 segments), FLORES-200 out of domain |
| Calibration and repair data | `haoranxu/ALMA-Human-Parallel` |
| Quality | BLEU, chrF++, COMET-22, COMETKiwi, XCOMET-XL, MetricX-24 |
| Efficiency | size, FLOPs, MFU, latency, throughput, three compression ratios |

Three decisions that are not recoverable from the code:

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

## Pipeline

`unbabel-comet` pins `numpy<2` and `torchmetrics<0.11`; MetricX pins
`transformers==4.30.2`. Neither can share a process with a current generation
stack, so the four stages are joined by files on disk:

```
generate  (GPU, gen env)     ->  runs/<slug>/hyps/<direction>.{jsonl,txt}
bench     (GPU, gen env)     ->  runs/<slug>/bench.json
score     (GPU, metric env)  ->  runs/<slug>/scores.<group>.json
report    (CPU, any env)     ->  reports/
```

A run directory holds `manifest.json` (identity, written once), `env.json`,
`stages/`, `hyps/`, `bench.json` and `scores.*.json`, and is interpretable on
its own without the configs that produced it. An old run can be re-scored with
a new metric without regenerating. See [docs/environments.md](docs/environments.md).

## Use

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gen,surface,dev]"

./scripts/smoke_local.sh          # all four stages on a small slice, 6 GB VRAM
```

```bash
mnlp-eval calibration --spec configs/calibration/multi-10dir.yaml
mnlp-eval run --model configs/models/alma-7b.yaml \
              --suite configs/suites/alma10-greedy.yaml
.venv-comet/bin/mnlp-eval score --groups neural
mnlp-eval report --baseline alma-7b --formats md,csv,tex
mnlp-eval overlap --subnetwork-dir subnetworks/
```

`mnlp-eval info` prints what the current environment can do. For Snellius see
[slurm/README.md](slurm/README.md).

`alma10-greedy` is the sweep suite; `alma10-beam5` matches ALMA's decode
settings and is for the headline table. Results from the two are not
comparable and the report refuses to mix them.

## What the report contains

Quality per direction and macro-averaged, with paired bootstrap significance
against the baseline. Behavioural failure rates: off-target language, empty,
source copy, repetition, truncation, length ratio. Efficiency with separate
compression ratios for disk bytes, parameter count and resident VRAM, because
they diverge. Per-layer structure and the share of weights that are exactly
zero, which is how a mask that was applied but never compacted shows up.
Degradation split by resource tier, and a cross-direction transfer matrix for
pair-specific subnetworks.

[docs/protocol.md](docs/protocol.md) records what each of those means and how
it is measured.

## Adding a model

Usually one YAML file:

```yaml
extends: alma-7b.yaml
name: alma-7b-prune50-multi
baseline: alma-7b
model_name_or_path: /scratch-shared/$USER/alma-7b-prune50-multi
compression:
  family: pruning
  nominal_sparsity: 0.5
  pruned_for: multi
  subnetwork: subnetworks/prune50-multi.json
```

A model needing custom modelling code supplies one function instead. See
[docs/plugging-in-a-model.md](docs/plugging-in-a-model.md) and, for the
subnetwork descriptor, [docs/subnetworks.md](docs/subnetworks.md).

## Layout

```
src/mnlp_eval/        config, data, models, metrics, bench, analysis, report
configs/              models, suites, calibration, metrics
subnetworks/          kept-unit descriptors, one per pruning run (see docs)
recipes/              loaders for models that need their own code
envs/                 requirements for the COMET and MetricX environments
slurm/                Snellius job scripts
scripts/              smoke test, ALMA reproduction, style guard
tests/                no GPU, no network, no weights
```

## Verification

- `./scripts/smoke_local.sh` runs all four stages and asserts the results are
  plausible, so a broken pipeline cannot pass as a bad model.
- `mnlp-eval verify-testset` checks the Hub test sets against ALMA's committed
  `human_written_data` files segment by segment.
- `./scripts/reproduce_alma_baseline.sh` reproduces ALMA-7B on all ten
  directions for comparison against the published table. Until that delta is
  known, every result here rests on an unverified harness.
- `pytest tests/` needs no GPU, network or weights.

No emojis and no em-dashes anywhere; `scripts/check_style.py` enforces it. See
[CONTRIBUTING.md](CONTRIBUTING.md).
