# small-mnlp-eval

Structured pruning of [ALMA-7B](https://github.com/fe1ixxu/ALMA) across its ten
translation directions (English to and from cs, de, is, ru, zh), and the harness that
evaluates the result: quality, behaviour and efficiency. Study design in
[docs/experiment.md](docs/experiment.md).

## Results

Pilot of 29 September 2026: the first 300 segments of each WMT22 direction, greedy
decoding, no fine-tuning after pruning. Plots, per-direction tables and the overlap
analysis are in [notebooks/pilot_results.ipynb](notebooks/pilot_results.ipynb), the data
in `results/pilot-2026-09-29/`.

| system | removed | calibration | BLEU | chrF++ | COMET |
| --- | --- | --- | --- | --- | --- |
| ALMA-7B | | | 30.35 | 51.14 | 0.8474 |
| SlimGPT | 20% | prompts | 20.65 | 41.71 | 0.8148 |
| SlimGPT | 20% | prompt + reference | 28.33 | 49.33 | 0.8367 |
| SlimGPT, uniform budget | 20% | prompts | 23.01 | 43.81 | 0.8222 |
| SlimGPT, one pair (best: zh) | 20% | prompts, 256 segments | 16.15 | 36.25 | 0.7340 |
| FLAP | 20% | prompts | 19.02 | 38.22 | 0.8031 |
| SlimGPT | 50% | prompts | 4.91 | 19.08 | 0.5567 |

- Calibrating on the prompt with its reference translation appended recovers 79% of
  the BLEU lost at 20%. A control with twice the prompt-only data gains little, so the
  effect comes from the target side.
- Uniform and log-increase per-layer budgets both beat the global one at 20%.
- Subnetworks calibrated on a single language pair are worse everywhere, their own
  pair included; this is confounded with calibration size (256 against 1,280 segments).
- 50% is unusable without repair. Pruning reduces memory (1.23x at 20%, 1.90x at 50%)
  but not generation time with Hugging Face generation.

Open:

- LoRA repair (`mnlp-eval repair`) is implemented and tested on Qwen2.5-0.5B but not
  run on ALMA-7B: the repair set shares two source segments with the WMT22 test set.
- Pair-specific runs at equal calibration size.
- LLM-Pruner runs out of memory; its gradient accumulator needs reducing to
  per-unit sums.

## Layout

Stages communicate only through files, because COMET and MetricX pin dependencies
that cannot share a process with the generation stack:

```
prune     (GPU, .venv)        ->  checkpoints/<name>/, subnetworks/<name>.json, configs/models/<name>.yaml
repair    (GPU, .venv)        ->  checkpoints/<name>-lora/, configs/models/<name>-lora.yaml
generate  (GPU, .venv)        ->  runs/<slug>/hyps/<direction>.{jsonl,txt}
bench     (GPU, .venv)        ->  runs/<slug>/bench.json
score     (GPU, .venv-comet)  ->  runs/<slug>/scores.<group>.json
report    (CPU, any env)      ->  reports/
```

```
src/mnlp_eval/prune/     groups, collect, budget, compact, spec, repair
src/mnlp_eval/prune/methods/
                         flap.py, llm_pruner.py, slimgpt.py
recipes/pruned.py        loads a compacted checkpoint (dense config + descriptor)
configs/prune/           one YAML per pruning run
configs/repair/          one YAML per repair run
configs/suites/          alma10-greedy, alma10-beam5, alma10-greedy-300 (pilot)
slurm/                   job scripts and submit_*.sh
notebooks/, results/     pilot results
docs/                    experiment, protocol, pruning, subnetworks, environments
tests/                   no GPU, no network, no weights
```

A run directory is named `<model>__<suite>__<hash>`, where the hash covers everything
that can move a number, so two systems appear in one table only if they were measured
identically. Pruning details in [docs/pruning.md](docs/pruning.md).

## Running it

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

# 2. Generate, score and report. AFTER=<prune job id> queues it behind pruning.
BASELINE=alma-7b bash slurm/submit_pilot.sh configs/suites/alma10-greedy-300.yaml \
    configs/models/alma-7b.yaml \
    configs/models/alma-7b-flap20-multi.yaml configs/models/alma-7b-slimgpt20-multi.yaml

# 3. Read results: runs/<model>__alma10-greedy-300__<hash>/scores.surface.json
squeue -u $USER
```

Repair queues the same way, behind the prune job that writes its source:

```bash
./.venv/bin/mnlp-eval calibration --spec configs/calibration/repair-multi.yaml   # login node
AFTER=<prune job id> bash slurm/submit_repair.sh configs/repair/slimgpt-20-multi-lora.yaml
```

A new pruning run is one YAML in `configs/prune/`, usually extending an
existing one and changing `name` and `sparsity`. The full sweep uses
`slurm/submit_sweep.sh` with `configs/suites/alma10-greedy.yaml`.

Things to know:

- The model YAMLs and descriptors that `prune` writes are generated, not
  committed. The YAMLs point at your own scratch, which other accounts cannot
  read and which is purged after 14 days untouched.
- Slurm logs land in `slurm-logs/` of the checkout you submitted from.
- Greedy and beam-5 results are not comparable; the report refuses to mix
  them.
- Checks before committing: `ruff format`, `ruff check`, `mypy`,
  `pytest tests/`, `scripts/check_style.py` (no emoji, no em-dashes).
