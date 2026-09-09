# small-mnlp-eval

Quality and efficiency evaluation for compressed machine translation LLMs.

This is the evaluation and experimental layer for a model-compression project
on the [ALMA](https://github.com/fe1ixxu/ALMA) translation models. The project
asks whether ALMA-7B can be structurally pruned into smaller multi-directional
or pair-specific subnetworks, how that interacts with resource level across its
ten translation directions, and how far a lightweight LoRA repair recovers what
pruning removed. Pruning happens elsewhere; this repository produces the
calibration data it runs on and every number the report quotes.

The design goal is that a compressed checkpoint becomes a fully evaluated
system by adding one YAML file, and that any two systems in the same table are
guaranteed to have been measured the same way.

[docs/experiment-plan.md](docs/experiment-plan.md) maps each research question
to the commands and artifacts that answer it. Read that first.

## What it measures

**Translation quality**, following ALMA's own evaluation protocol so numbers
stay comparable with the published baselines:

- sacreBLEU BLEU and chrF++, with full signatures recorded
- COMET-22 (`Unbabel/wmt22-comet-da`), the primary metric
- COMETKiwi, reference-free
- XCOMET-XL and MetricX-24-Hybrid, the WMT24 metrics task winners
- paired bootstrap significance against a named baseline run

**Behavioural failure modes**, which averaged quality scores hide:

- empty output rate
- off-target language rate, the classic collapse mode of compressed
  multilingual models
- source-copy rate
- hypothesis to reference length ratio
- truncation rate and degenerate-repetition rate

**Efficiency.** The course reference survey (Zhu et al., "A Survey on Model
Compression for Large Language Models") lists six metrics in its Section 2.1,
and all six are reported: model size, FLOPs, mean FLOPS utilisation, inference
time, speedup ratio, compression ratio. The following go beyond that list,
because a compression report needs them:

- checkpoint size on disk and resident VRAM, so compression is reported as
  three separate ratios (bytes, parameters, VRAM) because they diverge
- peak memory from both the torch allocator and NVML
- time to first token and per-sentence latency
- decode throughput under a forced token budget, at two batch sizes

FLOPs are an analytic estimate, not a traced operator count, and are not
estimated at all for encoder-decoder models. Utilisation is computed against
the device's dense bf16 peak and named `mfu_bf16_equivalent` to say so.

**Structure**, because for structured pruning "how much smaller" is not the
question. Measured from the loaded weights rather than from the config that
produced them:

- attention and FFN width per layer, and whether the pruning budget was spent
  evenly across depth
- the share of floating-point weights that are exactly zero, which is how a
  mask that was applied but never compacted shows up: full parameter count,
  full checkpoint size, full latency, no saving delivered

**Subnetwork overlap**, for the question of whether two pair-specific
subnetworks kept the same units. Reported as excess over chance, because two
independently chosen 50 percent subnetworks already share about a third of what
they keep. See [docs/subnetworks.md](docs/subnetworks.md).

## Why there are three environments

`unbabel-comet` pins `numpy<2` and `torchmetrics<0.11`. MetricX pins
`transformers==4.30.2`. Neither can share a process with a current generation
stack. So the pipeline is four stages joined by files on disk, not by function
calls:

```
generate  (GPU, gen env)      ->  runs/<slug>/hyps/<direction>.{jsonl,txt}
bench     (GPU, gen env)      ->  runs/<slug>/bench.json
score     (GPU, metric env)   ->  runs/<slug>/scores.<group>.json
report    (CPU, any env)      ->  reports/
```

A run directory holds `manifest.json` (the run's identity, written once),
`env.json`, `stages/<stage>.json` (one file per stage, and per direction where
a stage is sharded across a Slurm array), `hyps/`, `bench.json` and
`scores.*.json`. It is designed to be interpretable on its own, months later,
without the configs that produced it.

This is a constraint rather than a preference, but it pays for itself: an old
run can be re-scored with a new metric without regenerating anything, and
scoring runs as its own Slurm job. See [docs/environments.md](docs/environments.md).

MetricX needs a clone rather than an install, because upstream ships no
`pyproject.toml`. Run `scripts/setup_metricx_env.sh`.

## Quickstart

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gen,surface,dev]"

# End to end on a small local slice. Takes a couple of minutes on 6 GB of
# VRAM and asserts the results are sane, so a broken pipeline cannot pass as
# a bad model.
./scripts/smoke_local.sh
```

Then, for a real evaluation:

```bash
# The calibration data every pruning run shares. Deterministic, balanced
# across directions, fingerprinted, and checked against the test sets.
.venv/bin/mnlp-eval calibration --spec configs/calibration/multi-10dir.yaml

# Generate, measure efficiency, and score surface metrics in one command.
.venv/bin/mnlp-eval run --model configs/models/alma-7b.yaml \
                        --suite configs/suites/alma10-greedy.yaml

# Neural metrics live in their own environment. See docs/environments.md.
.venv-comet/bin/mnlp-eval score --groups neural

.venv/bin/mnlp-eval report --baseline alma-7b --formats md,csv,tex

# Which units the pair-specific subnetworks kept, and whether they agree
# more than chance would.
.venv/bin/mnlp-eval overlap --subnetwork-dir subnetworks/
```

`mnlp-eval info` prints what the current environment can do. On Snellius, see
[slurm/README.md](slurm/README.md), which submits one generation array per
model sharded by direction, then efficiency, then scoring, then the report.

## What a result looks like

Casting the reference MT system to fp16. Quality on 200 segments of WMT22
de-en at beam 4; efficiency on 64 segments at a forced 64-token budget, 9
timed repeats, one RTX 1000 Ada (6 GB). Conditions matter and are stated
because this document elsewhere insists on exactly that.

| System | BLEU | COMET-22 | Resident | Bits/param | Compression (disk) | Compression (VRAM) |
| --- | --- | --- | --- | --- | --- | --- |
| opus-mt-de-en | 31.89 | 0.8237 | 284.60 MiB | 32.08 | 1.00x | 1.00x |
| opus-mt-de-en-fp16 | 31.94 | 0.8243 | 142.04 MiB | 16.01 | 1.00x | 2.00x |

| System | tok/s b1 | stdev | tok/s b8 | stdev |
| --- | --- | --- | --- | --- |
| opus-mt-de-en | 237.6 | 9.7% | 1552.8 | 2.3% |
| opus-mt-de-en-fp16 | 203.6 | 6.6% | 1373.6 | 10.0% |

What this table is here to show:

- The disk ratio reads 1.00x while the VRAM ratio reads 2.00x, because the cast
  happens at load time and the checkpoint is untouched. Reporting a single
  compression ratio would have been misleading either way. Bits per parameter
  reading 32.08 and 16.01 is the cheapest available check that the size
  accounting every ratio depends on is right.
- Halving the weights bought no speed here. fp16 is slower at both batch
  sizes, on a 74M-parameter encoder-decoder model on a laptop GPU. Compression
  that shrinks a model is not compression that accelerates it, and the
  framework reports the two separately for that reason.
- The run-to-run spread is 2.3 to 10 percent of the median, which is the same
  size as the differences above. That is why the two batch sizes and the
  repeat count are reported next to the numbers, and why a throughput claim
  from a shared cluster node needs an exclusive allocation before it means
  anything.
- The quality difference is not significant: the paired bootstrap over those
  200 segments gives p = 0.169 for BLEU and p = 0.186 for COMET-22, so the
  framework reports it as indistinguishable rather than as an improvement.

An earlier version of this table quoted a batch-size crossover, fp16 slower at
batch 1 and faster at batch 8, from three repeats. It did not reproduce at
nine. The two batch sizes are still measured because the memory-bound and
compute-bound regimes genuinely can disagree, but this measurement does not
demonstrate it.

## Adding your model

Most models need no code. Write a YAML file in `configs/models/`:

```yaml
name: alma-7b-bnb-nf4
loader: hf_causal
prompt: alma
model_name_or_path: haoranxu/ALMA-7B
dtype: bfloat16
quantization:
  method: bitsandbytes
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
```

A model that needs custom modelling code, as most pruning does, supplies one
function instead:

```yaml
name: alma-7b-wanda-50
loader: custom
prompt: alma
baseline: alma-7b
entrypoint: recipes.wanda:load     # load(**kwargs) -> (model, tokenizer)
kwargs:
  sparsity: 0.5
  checkpoint: /scratch-shared/$USER/wanda-50
compression:
  family: pruning
  nominal_sparsity: 0.5
  pruned_for: multi                # or a direction list, for a pair-specific cut
  subnetwork: subnetworks/wanda-50.json
```

The `compression` block takes no part in run identity, but the structure,
resource tier and transfer tables are built from it and are omitted when it is
absent. Full details in
[docs/plugging-in-a-model.md](docs/plugging-in-a-model.md).

## The comparability contract

A compression comparison is worthless if the systems were not measured
identically, so the framework enforces this rather than documenting it:

- a run's identity is a hash of the model, data, and decode specifications, so
  changing any of them produces a different run rather than overwriting one
- the prompt template is fingerprinted and the fingerprint is stored
- `do_sample` is false unless explicitly enabled, and sampling parameters are
  rejected when it is off
- `report` refuses to place two runs in the same table when their decode
  specifications differ

See [docs/protocol.md](docs/protocol.md) for the full protocol, including the
efficiency measurement procedure.

## Layout

```
src/mnlp_eval/         the package: config, data, models, metrics, bench, report
src/mnlp_eval/analysis/  subnetwork descriptors and overlap against chance
configs/models/        one file per system, including compression variants
configs/suites/        test set plus decode settings
configs/calibration/   pruning and repair data, balanced and fingerprinted
configs/metrics/       which metrics, which checkpoints
subnetworks/           kept-unit descriptors, one per pruning run
recipes/               loaders for models that need their own code
envs/                  requirements for the COMET and MetricX environments
slurm/                 Snellius job scripts and a submit_sweep driver
scripts/               smoke test, ALMA reproduction, style guard
docs/                  experiment plan, environments, plugin contract, protocol
tests/                 no GPU, no network, no weights
```

## Verification

- `./scripts/smoke_local.sh` runs all four stages on 6 GB and asserts the
  results are plausible, including that fp32 reads 32 bits per parameter and
  fp16 reads 16.
- `mnlp-eval verify-testset` confirms the Hub test sets match ALMA's committed
  `human_written_data` files segment by segment. Row counts across the ten
  directions: cs-en 1448, de-en 1984, is-en 1000, ru-en 2016, zh-en 1875,
  en-cs 2037, en-de 2037, en-is 1000, en-ru 2037, en-zh 2037, for 17,491
  segments per system. The Icelandic configs hold the WMT21 test set, because
  WMT22 had no Icelandic task and ALMA evaluates it on WMT21 for that reason.
- `./scripts/reproduce_alma_baseline.sh` reproduces ALMA-7B on all ten
  directions and prints scores to compare against the published table. Until
  that delta is known, every compression result here rests on an unverified
  harness.
- `pytest tests/` needs no GPU, network or model weights, and runs in seconds.

## Repository conventions

No emojis and no em-dashes, anywhere. `scripts/check_style.py` enforces this
and CI runs it. Commit messages and pull requests must not credit AI agents as
authors or co-authors. See [CONTRIBUTING.md](CONTRIBUTING.md).
