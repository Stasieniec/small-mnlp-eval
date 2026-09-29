# small-mnlp-eval: pruning pilot

Structured pruning of [ALMA-7B](https://github.com/fe1ixxu/ALMA) across its
ten translation directions (English against cs, de, is, ru, zh), and the
harness that evaluates the result. This branch builds on `feat/bart/pruning`
with bug fixes found by running it, a reduced pilot suite, and the first
results. Study design: [docs/experiment.md](docs/experiment.md).

## Results so far

Suite `alma10-greedy-300`: first 300 segments of each WMT22 direction,
greedy decoding, no repair fine-tuning. Macro means over the ten
directions; COMET is `Unbabel/wmt22-comet-da`. These are the runs of
2026-09-29 on scur0560; per-direction tables, significance tests and the
overlap analysis are in
[docs/overnight-2026-09-29.md](docs/overnight-2026-09-29.md). All are
SlimGPT with a global budget unless the name says otherwise.

| system | sparsity | calibration | BLEU | chrF++ | COMET | length ratio | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALMA-7B dense | 0 | | 30.35 | 51.14 | 0.8474 | 0.96 | 0.0% |
| slimgpt20-multi | 20% | 1,280 prompts | 20.65 | 41.71 | 0.8148 | 0.91 | 0.1% |
| slimgpt20-multi-uniform | 20% | 1,280 prompts | 23.01 | 43.81 | 0.8222 | 0.89 | 0.0% |
| slimgpt20-multi-log | 20% | 1,280 prompts | 21.92 | 42.92 | 0.8223 | 0.90 | 0.1% |
| **slimgpt20-multi-target** | 20% | 1,280 prompt+reference | **28.33** | **49.33** | **0.8367** | 0.95 | 0.0% |
| slimgpt20-multi-256 | 20% | 2,560 prompts | 21.22 | 42.28 | 0.8195 | 0.90 | 0.2% |
| slimgpt20-cs | 20% | 256 prompts, cs | 12.28 | 31.73 | 0.6845 | 0.96 | 2.1% |
| slimgpt20-de | 20% | 256 prompts, de | 10.37 | 29.08 | 0.6482 | 1.00 | 1.8% |
| slimgpt20-is | 20% | 256 prompts, is | 13.05 | 33.36 | 0.7105 | 0.97 | 1.3% |
| slimgpt20-ru | 20% | 256 prompts, ru | 10.86 | 30.18 | 0.6523 | 0.95 | 0.6% |
| slimgpt20-zh | 20% | 256 prompts, zh | 16.15 | 36.25 | 0.7340 | 0.96 | 1.6% |
| flap20-multi | 20% | 1,280 prompts | 19.02 | 38.22 | 0.8031 | 0.82 | 0.3% |
| flap20-de | 20% | 256 prompts, de | 19.62 | 39.30 | 0.8043 | 0.86 | 0.4% |
| slimgpt50-multi | 50% | 1,280 prompts | 4.91 | 19.08 | 0.5567 | 0.65 | 0.8% |
| slimgpt50-multi-log | 50% | 1,280 prompts | 3.82 | 17.26 | 0.5645 | 0.61 | 0.0% |

- **Calibrating on prompt plus reference recovers most of the 20% loss:**
  28.33 BLEU and 0.8367 COMET, against 20.65 and 0.8148 for prompts alone.
  The 2,560-prompt control, which sees more tokens, gains only +0.57 BLEU,
  so the gain comes from the target side. It is still below dense overall
  (COMET -0.0107, p=0.001), and it helps into English more than out of it.
- **Per-layer schedules beat the global budget at 20%:** uniform +2.4 BLEU,
  log-increase +1.3, both +0.007 COMET (p=0.001), the gain being into
  English. Uniform and log-increase tie on COMET.
- **Pair-specific SlimGPT subnetworks collapse:** 10.4-16.2 BLEU, below the
  multi subnetwork on every direction including their own pair, worst on
  Icelandic. This is confounded with calibration size (256 against 1,280
  segments); FLAP at 256 segments ties FLAP at 1,280 (COMET +0.0012,
  p=0.441).
- **50% without repair is unusable** with either schedule. SlimGPT writes
  short, fluent, loosely related English (length ratio 0.61-0.65), where
  FLAP ran on without stopping.
- **SlimGPT beats FLAP** at 20% with multi calibration: +1.6 BLEU, COMET
  +0.0116 (p=0.001).
- Every pruned system is significantly below dense in every direction on
  BLEU, chrF++ and COMET, except five cells of slimgpt20-multi-target.
- **Pruning saves memory, not time, with this harness.** Parameters and
  resident memory shrink 1.23x at 20% and 1.90x at 50%, but throughput is
  0.94-1.00x dense at batch sizes 1 and 8. Eager Hugging Face generation
  is bound by per-step overhead here (model FLOP utilisation under 2%), so removing
  weights does not shorten a step. The report's disk ratios compare a
  float32 dense checkpoint with bfloat16 pruned ones; use the parameter or
  VRAM ratios.
- **No repair has run:** the repair set failed its contamination check (see
  below).

### Pilot, on Jan's account

Run before COMET worked: BLEU and chrF++ only, no significance testing.
Tonight's re-runs of dense, SlimGPT multi and both FLAP 20% systems
reproduce these numbers exactly.

| system | sparsity | BLEU | chrF++ | length ratio | truncated |
| --- | --- | --- | --- | --- | --- |
| ALMA-7B dense | 0 | 30.35 | 51.14 | 0.96 | 0.0% |
| **SlimGPT, multi calibration** | 20% | **20.65** | **41.71** | 0.91 | 0.1% |
| FLAP, de calibration | 20% | 19.62 | 39.30 | 0.86 | 0.4% |
| FLAP, multi calibration | 20% | 19.02 | 38.22 | 0.81 | 0.3% |
| FLAP, multi calibration | 50% | 1.16 | 9.56 | 3.60 | 18.8% |
| FLAP, de calibration | 50% | 1.05 | 8.93 | 4.24 | 20.7% |

BLEU per direction at 20% sparsity:

| | cs-en | de-en | is-en | ru-en | zh-en | en-cs | en-de | en-is | en-ru | en-zh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 42.0 | 31.3 | 38.2 | 36.9 | 19.8 | 24.9 | 28.0 | 23.8 | 25.0 | 33.5 |
| SlimGPT multi | 28.5 | 21.3 | 26.9 | 24.8 | 12.0 | 15.5 | 19.4 | 15.9 | 16.4 | 25.8 |
| FLAP de | 28.3 | 20.9 | 26.7 | 25.4 | 10.0 | 14.0 | 19.5 | 15.0 | 16.2 | 20.2 |
| FLAP multi | 25.8 | 20.8 | 26.0 | 25.1 | 9.4 | 12.7 | 18.8 | 14.5 | 16.6 | 20.6 |

- **50% without repair is unusable.** FLAP output starts plausibly, then
  fails to stop: about 4x the reference length, invented numbers, tag loops.
- **20% keeps about two thirds of BLEU and 80% of chrF++.** Output is
  fluent and in the right language, but short (length ratio 0.81-0.91), so
  part of the loss is omission.
- **SlimGPT beats FLAP** on 7 of 10 directions (the rest by 0.6 BLEU or
  less). Its gains are largest where FLAP is weakest: en-zh (+5.2), zh-en
  (+2.0), en-cs (+1.5).
- **By language:** German and Russian are most robust. Chinese is hit hardest
  (zh-en keeps half its BLEU at best). Out of English suffers more than into
  English for cs, is and zh.
- **German-only calibration is not worse than all-ten** despite 256 against
  1,280 segments, and helps Czech most. Treat as a tie until significance is
  tested. (Tested on 2026-09-29: a tie, COMET +0.0012, p=0.441.)

## Problems and solutions

| problem | status | cause and fix |
| --- | --- | --- |
| SlimGPT collapsed at 50% (`nobody nobody ...`) | **fixed** | Pass one scored every layer on the embedding output instead of its own input, so layers 1-31 were ranked on the wrong activations. Pass one now advances a dense copy layer by layer. The 50% SlimGPT run predates the fix and was discarded. |
| SlimGPT Hessian included padding | **fixed** | Left-padded eos positions reached `X X^T`. The 2D mask now travels with each batch and filters the Hessian. |
| FLAP checkpoint failed to load | **fixed** | FLAP adds a bias only where units were dropped; the loader assumed all or none. `reshape_model` takes a per-projection predicate read from the saved tensors. |
| Every GPU job died at the first kernel (`Python.h` missing) | **fixed** | Venvs built on system Python, whose headers are absent on GPU nodes. Build with `uv venv --managed-python`; module is now `Python/3.13.1-GCCcore-14.2.0`. |
| LLM-Pruner OOM on A100-40GB and H100-94GB | open | A float32 `W * dL/dW` accumulator for every prunable weight (about 26 GB) plus activations. Proposed: reduce to per-head and per-channel sums after each backward (exact, sums are linear), and enable gradient checkpointing. |
| COMET: `Model 'Unbabel/wmt22-comet-da' not supported` | **fixed, verified on Snellius** | COMET raises this for any failure to fetch the checkpoint, and compute nodes are offline, so the checkpoint was most likely never in `HF_HOME`. Run `.venv-comet/bin/mnlp-eval prefetch --metrics configs/metrics/default.yaml` on a login node. Prefetch now also caches the XLM-R encoder files COMET loads next, and the error now names the cache. On 2026-09-29 every score job on a GPU node wrote `scores.neural.json`. |
| Report job is always cancelled | **fixed** | `submit_pilot.sh` chains the report `afterok` on scoring, and scoring exited 1 on the COMET error after writing `scores.surface.json`. With COMET working, scoring succeeds and the 2026-09-29 report ran. |
| SlimGPT's global budget cuts layer 0 hard | **run** | Per-layer standardisation makes heavy-tailed layers lose most. `allocation: log-increase` is the paper's Incremental Pruning Ratio, layer 0 whole rising to 1.36 times the target at layer 31. At 20% both uniform (+2.4 BLEU) and log-increase (+1.3) beat global, +0.007 COMET each (p=0.001), and tie with each other on COMET. At 50% all three are unusable without repair. See [docs/pruning.md](docs/pruning.md#budget). |
| Pruned models stop short at 20% | **blocked**: repair not run | Length ratio 0.81-0.91. `mnlp-eval repair` trains LoRA with ALMA's own recipe on a pruned checkpoint and merges it; configs in `configs/repair/`. The smoke run on Qwen2.5-0.5B works end to end. The ALMA-7B runs wait on the repair-set decision in the next row. See [docs/pruning.md](docs/pruning.md#repair). |
| Repair set overlaps the test set | **open, needs a decision** | `repair-multi` has 2 exact-source collisions with WMT22-Test in 117,404 segments: en-cs `Amazing.` and ru-en `Ну и что?`, test segments 1356 and 311, both outside the 300-segment pilot slice. A hard failure by design, so the set is quarantined as `data/calibration/repair-multi.CONTAMINATED` and no repair ran. |
| A failed calibration build left a usable set on disk | **fixed** | `mnlp-eval calibration` writes the set before exiting 1 on a collision, under the name the configs read. `load_calibration_prompts` and `load_repair_data` now refuse a set whose `calibration.json` records collisions. |
| Out of English loses more than into English (cs, is, zh) | **tested** | Calibration reads only the prompt, so en-xx calibration never puts target-language text through the model. `calibration_text: prompt+target` (`slimgpt-20-multi-target`) gains +7.7 BLEU and +0.022 COMET over prompts alone, and beats the token-count control `slimgpt-20-multi-256` by +7.1 BLEU. It helps into English more than out of it, so it does not specifically close an out-of-English gap. On this suite the prompt-only macro loss is not larger out of English. See [docs/pruning.md](docs/pruning.md#calibration-text). |
| Pruning masks for specific directions | **run, confounded** | SlimGPT at 20% calibrated on one pair at a time (`slimgpt-20-{cs,de,is,ru,zh}`), each evaluated on all ten directions and compared with `mnlp-eval overlap`. Each loses to the multi subnetwork on every direction, its own pair included, and loses two to three times as much elsewhere. The 256 against 1,280 calibration segments look like the larger effect. |

## What this branch contains

On top of `main`, the pruning branch adds a fifth stage ahead of the four
evaluation stages. Stages communicate only through files, because COMET and
MetricX pin dependencies that cannot share a process with the generation
stack:

```
prune     (GPU, .venv)        ->  checkpoints/<name>/, subnetworks/<name>.json, configs/models/<name>.yaml
repair    (GPU, .venv)        ->  checkpoints/<name>-lora/, configs/models/<name>-lora.yaml (optional)
generate  (GPU, .venv)        ->  runs/<slug>/hyps/<direction>.{jsonl,txt}
bench     (GPU, .venv)        ->  runs/<slug>/bench.json
score     (GPU, .venv-comet)  ->  runs/<slug>/scores.<group>.json
report    (CPU, any env)      ->  reports/
```

```
src/mnlp_eval/prune/     shared pipeline: groups, collect, budget, compact, spec
src/mnlp_eval/prune/methods/
                         flap.py, llm_pruner.py, slimgpt.py (swappable criteria)
src/mnlp_eval/prune/repair.py
                         LoRA repair of a pruned checkpoint, merged
recipes/pruned.py        loads a compacted checkpoint (dense config + descriptor)
configs/prune/           one YAML per pruning run: method, sparsity, calibration
configs/repair/          one YAML per repair run: the pruned system and the data
configs/suites/          alma10-greedy (sweep), alma10-beam5 (headline),
                         alma10-greedy-300 (pilot, this branch)
slurm/                   job scripts; submit_prune.sh, submit_repair.sh,
                         submit_pilot.sh, submit_sweep.sh
docs/                    experiment, protocol, pruning, subnetworks, environments
tests/                   no GPU, no network, no weights
```

| criterion | statistic | compensation |
| --- | --- | --- |
| FLAP | per-channel activation variance times weight norm | bias on `o_proj`/`down_proj` |
| LLM-Pruner | first-order Taylor `W * dL/dW` | none, LoRA is a separate stage |
| SlimGPT | Hessian `X X^T` per projection | exact weight update on survivors |

A run directory is named `<model>__<suite>__<hash>`, where the hash covers
everything that can move a number, so two systems appear in one table only if
they were measured identically. Details in [docs/pruning.md](docs/pruning.md).

## How to use it

Setup on Snellius, once, on a login node (full notes in
[slurm/README.md](slurm/README.md)):

```bash
export HF_HOME=/scratch-shared/$USER/hf_home
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
pip install --user uv
uv venv --managed-python --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gen,prune,quant,surface,report]"
uv venv --managed-python --python 3.11 .venv-comet
uv pip install --python .venv-comet/bin/python -r envs/comet-requirements.txt
mkdir -p slurm-logs
```

Prune, then evaluate on the pilot suite:

```bash
# 1. Prune. Writes the checkpoint to /scratch-shared/$USER/checkpoints/,
#    plus subnetworks/<name>.json and configs/models/<name>.yaml.
bash slurm/submit_prune.sh configs/prune/flap-20-multi.yaml configs/prune/slimgpt-20-multi.yaml

# 2. When pruning is done (it does not chain), generate, score and report.
BASELINE=alma-7b bash slurm/submit_pilot.sh configs/suites/alma10-greedy-300.yaml \
    configs/models/alma-7b.yaml \
    configs/models/alma-7b-flap20-multi.yaml configs/models/alma-7b-slimgpt20-multi.yaml

# 3. Read results: runs/<model>__alma10-greedy-300__<hash>/scores.surface.json
squeue -u $USER
```

A new pruning run is one YAML in `configs/prune/`, usually extending an
existing one and changing `name` and `sparsity`. The full sweep uses
`slurm/submit_sweep.sh` with `configs/suites/alma10-greedy.yaml`.

Things to know:

- The model YAMLs and descriptors that `prune` writes point at your own
  scratch, which other accounts cannot read and which is purged after 14 days
  untouched. They are generated, not committed.
- Slurm logs land in `slurm-logs/` of the checkout you submitted from.
- Greedy and beam-5 results are not comparable; the report refuses to mix
  them.
- Checks before committing: `ruff format`, `ruff check`, `mypy`,
  `pytest tests/`, `scripts/check_style.py` (no emoji, no em-dashes).
