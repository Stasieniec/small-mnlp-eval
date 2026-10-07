# small-mnlp-eval

Structured pruning of [ALMA-7B](https://github.com/fe1ixxu/ALMA) across its ten
translation directions (English to and from cs, de, is, ru, zh), and the harness that
evaluates the result: quality, behaviour and efficiency. Study design in
[docs/experiment.md](docs/experiment.md).

## Results

Final pilot grid of 6 October 2026: 192 pruned models (SlimGPT and FLAP; prompt +
reference or prompt + dense-generated calibration; 20, 30 and 40 percent removed;
one multi-directional, five pair and ten direction subnetworks, every one calibrated
on 1,280 segments), each evaluated on the first 300 segments of all ten WMT22
directions, greedy, plus LoRA repairs. **Report:
[results/grid-2026-10-06/README.md](results/grid-2026-10-06/README.md)**; decisions and
job log: [docs/overnight-2026-10-06.md](docs/overnight-2026-10-06.md).

COMET-22 (BLEU), macro over the ten directions; pair and dir translate each direction
with the matching specialist. Dense ALMA-7B: 0.8474 (30.35).

| method | calibration | removed | multi | pair | dir | multi + LoRA |
| --- | --- | --- | --- | --- | --- | --- |
| SlimGPT | reference | 20% | 0.8374 (27.72) | 0.8369 (27.97) | 0.8371 (27.95) | 0.8381 (28.41) |
| SlimGPT | generated | 30% | 0.8229 (25.88) | 0.8276 (26.52) | 0.8267 (26.58) | 0.8324 (27.59) |
| SlimGPT | generated | 40% | 0.7905 (22.92) | 0.8092 (24.18) | 0.8081 (24.44) | 0.8179 (26.36) |
| FLAP | reference | 20% | 0.8221 (26.81) | 0.8297 (27.24) | 0.8299 (27.67) | |
| FLAP | reference | 30% | 0.7823 (23.66) | 0.8003 (24.09) | 0.8017 (24.49) | |
| FLAP | reference | 40% | 0.7393 (20.50) | 0.7552 (20.11) | 0.7552 (20.33) | 0.8112 (25.73) |

Rows follow the repaired configurations; the other calibration text is within 0.0015
COMET of each cell, and all twelve method, calibration and sparsity rows are in the write-up.
The five SlimGPT 40 percent pair models, each LoRA-repaired on its own pair, reach
0.8236 (26.08).

- SlimGPT beats FLAP at every setting, by 0.015 COMET at 20 percent and 0.05 at 40.
- Reference and dense-generated calibration are indistinguishable, so calibration
  needs only source sentences.
- At equal calibration size, pair and direction subnetworks beat the
  multi-directional one from 30 percent up (+0.017 COMET at 40), and pair matches
  direction. A second calibration draw reproduces this. They fail on other
  directions that write a different language, and transfer to directions that
  read one.
- Out-of-English Icelandic and Chinese degrade most.
- LoRA repair recovers about half of the 40 percent loss for SlimGPT and two thirds
  for FLAP.

SlimGPT's compensation was fixed on 6 October (see docs/pruning.md); the 29
September pilot in `results/pilot-2026-09-29/` used the earlier code.

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
