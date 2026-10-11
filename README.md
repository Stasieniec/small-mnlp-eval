# small-mnlp-eval

Code and results for our MNLP project: structured pruning of
[ALMA-7B](https://github.com/fe1ixxu/ALMA) with SlimGPT and FLAP at 20, 30 and 40%
sparsity, comparing one multi-directional subnetwork with pair- and direction-specific
ones, plus LoRA repair. Evaluated on WMT22 (WMT21 for Icelandic) with BLEU, chrF++,
COMET-22 and MetricX-24.

## Layout

```
src/mnlp_eval/      the mnlp-eval package: calibration data, pruning, repair, generation, scoring
recipes/pruned.py   loads a pruned (compacted) checkpoint
scripts/            grid driver (grid.py), result aggregation (grid_report.py and friends)
configs/            calibration, prune, repair, model, suite and metric configs
slurm/              Snellius job scripts
results/            scores of every run we report
notebooks/          final_results.ipynb produces all tables and figures in the report
tests/              unit tests (no GPU needed)
```

## Reproducing the tables and figures

Everything in the report comes from `notebooks/final_results.ipynb`, which only reads the
committed files in `results/` (no GPU needed):

```bash
pip install -e ".[report]" jupyter
jupyter nbconvert --to notebook --execute notebooks/final_results.ipynb
```

Tables (CSV and LaTeX) and figures are written to `results/paper/`.

- `results/full-2026-10-08/`: the final grid on the full test sets
- `results/grid-2026-10-06/`: the same grid on the first 300 segments per direction, with
  the transfer matrices, subnetwork masks and the second calibration seed
- `results/pilot-2026-09-29/`: the earlier pilot (first SlimGPT implementation)

## Running the experiments

We ran everything on Snellius. COMET and MetricX need their own environments because
their pinned dependencies conflict with the generation stack.

```bash
export HF_HOME=/scratch-shared/$USER/hf_home
module load 2025 Python/3.13.1-GCCcore-14.2.0
pip install --user uv

uv venv --managed-python --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gen,prune,surface,report]"
uv venv --managed-python --python 3.11 .venv-comet
uv pip install --python .venv-comet/bin/python -r envs/comet-requirements.txt
bash scripts/setup_metricx_env.sh      # MetricX-24, cloned into third_party/

bash slurm/prefetch.sh                 # download models, data and metric checkpoints
sbatch slurm/setup_dense_bf16.sbatch   # bf16 copy of ALMA-7B on scratch
mkdir -p slurm-logs
```

**Calibration data.** Build the calibration and repair sets (on a login node), then
generate the dense model's translations for the "generated" calibration variant:

```bash
for f in configs/calibration/{multi-10dir,pair-*-640,dir-*-1280,repair-multi-clean}.yaml; do
    ./.venv/bin/mnlp-eval calibration --spec "$f"
done
sbatch --export=ALL,CALIBRATION_INPUT=data/calibration/multi-10dir,CALIBRATION_OUT=data/calibration/multi-10dir-generated \
    slurm/calibration_generate.sbatch   # likewise for the pair and dir sets
```

**Pruning grid.** 2 methods x 2 calibration texts x 3 sparsities x 16 subnetworks
(1 multi, 5 pair, 10 direction). Each job prunes one model, then generates and scores it
on the 300-segment suite:

```bash
./.venv/bin/python scripts/grid.py configs     # writes configs/prune/grid/ and configs/grid/manifest.json
./.venv/bin/python scripts/grid.py submit      # --scope multi|pair|dir, --dry-run
./.venv/bin/python scripts/grid.py status
./.venv/bin/python scripts/grid.py repairs --source alma-7b-slimgpt40-gen-multi   # LoRA repair
```

**Full test sets.** Evaluate chosen systems on the full `alma10-greedy` suite (each
specialist on its own directions), then score with MetricX:

```bash
bash slurm/submit_eval.sh alma-7b alma-7b-slimgpt40-gen-multi alma-7b-slimgpt40-gen-pair-{cs,de,is,ru,zh}
sbatch --export=ALL,RUN_LIST=runs.txt slurm/score_metricx.sbatch
```

**Aggregate.**

```bash
./.venv/bin/python scripts/grid_report.py --out results/grid-2026-10-06
./.venv/bin/python scripts/grid_report.py --out results/full-2026-10-08 --suite alma10-greedy
./.venv/bin/python scripts/grid_overlap.py      # subnetwork overlap
./.venv/bin/python scripts/repair_summary.py
```

Single models outside the grid (ablations, the second calibration seed) use
`slurm/single_pipeline.sbatch`. The CLI also has the individual stages:

```bash
./.venv/bin/mnlp-eval prune --spec configs/prune/grid/alma-7b-slimgpt40-gen-multi.yaml
./.venv/bin/mnlp-eval generate --model configs/models/alma-7b-slimgpt40-gen-multi.yaml \
    --suite configs/suites/alma10-greedy.yaml
./.venv/bin/mnlp-eval score --run runs/<run dir> --metrics configs/metrics/default.yaml
```

Model configs under `configs/models/` for pruned systems were written by the prune and
repair stages and point at checkpoints on our scratch space, which is not public.

## Tests

```bash
uv pip install -e ".[surface,dev,report]"
pytest tests/ -m "not gpu and not network and not slow and not torch"
```
