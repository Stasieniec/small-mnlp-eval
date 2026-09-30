# Boundary-layer protection experiment

This is a fixed comparison on ALMA-7B: SlimGPT, prompt plus reference calibration,
20% nominal unit removal, uniform allocation, no repair. Each condition uses
identical calibration records and decoding. It tests whether moving removal away
from the boundaries preserves translation at the same model size.

| Condition | First layers protected | Last layers protected |
| --- | ---: | ---: |
| `none` (primary control) | 0 | 0 |
| `first4` | 4 | 0 |
| `last2` | 0 | 2 |
| `first4-last2` | 4 | 2 |
| `first3-last1` | 3 | 1 |

Dense ALMA and an unprotected, global-allocation, prompt+reference SlimGPT model
are also evaluated by default. The global model is a contextual reference, not
one of the five matched uniform conditions. All are generated and scored on the
same frozen evaluation data. Existing pilot outputs are not mixed into the run.

## Why these masks

[LLM-Pruner, Figure 3 and Appendix B.1](https://proceedings.neurips.cc/paper_files/paper/2023/file/44956951349095f74492a5471128a7e0-Paper-Conference.pdf)
motivates boundary protection. Its prose describes first-three/last-one
protection. The authors' [example script](https://github.com/horseee/LLM-Pruner/blob/main/scripts/llama_prune.sh),
inspected on 2026-09-30, instead supplies the end-exclusive range `[4, 30)`,
protecting first-four/last-two on a 32-layer model. We test both masks; this is an
ALMA/SlimGPT experiment inspired by those settings, not a reproduction of their
complete pruning/recovery procedure. First-only and last-only protect different
numbers of layers, so their direct difference cannot establish sensitivity per
protected layer.

## Exact budget and protection semantics

`protect_first_n` and `protect_last_n` are non-negative integer counts; their
regions must not overlap. Both attention and FFN units are protected. SlimGPT
skips compensation and compaction there, but forwards their activations to later
layers. FLAP and LLM-Pruner also respect these options. Protection applies during
pruning; it does not freeze weights during a later repair stage.

With both counts zero, existing allocation behaviour is unchanged. The legacy
20% uniform control removes 6 of 32 heads and 2,202 of 11,008 FFN channels in every
layer. This means **192 heads and 70,464 channels overall**, not exactly 20% of
heads. All five conditions match these realised counts exactly. Eligible layers
share removal as evenly as whole units permit; deterministic rounding ties favour
earlier eligible layers. The driver prints every width with `plan`.

Global protection reallocates the unprotected global allocator's realised budget
among eligible layers. Protected log-increase keeps the original depth coordinate
and rescales its logarithmic profile over eligible layers. Infeasible budgets are
rejected; each layer must retain at least one head group and one FFN channel.
Mixed head-group steps across eligible layers are rejected rather than silently
missing the budget. Only uniform allocation is varied by this experiment.

Every checkpoint carries `prune.json` (full spec, protected indices, actual removal
counts and parameter counts) and `subnetwork.json` (all retained indices and the
protection mask). Generated model configs include the full pruning spec in their
identity. The final report requires matching measured parameter counts and unit
removals across all five conditions.

## Run on Snellius

Use the generation environment with `.[gen,prune,surface]` installed, and the
separate COMET environment described in `docs/environments.md`. COMET is required;
the launcher fails if it is unavailable. CPU tests for this change use Transformers
4.51.3 and PEFT 0.15.2; CI pins the corresponding compatible ranges.

If the environments do not exist yet, create them first (MetricX is not needed):

```bash
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
python -m venv .venv
.venv/bin/python -m pip install -e ".[gen,prune,surface]" \
  "transformers>=4.48,<4.52" "peft>=0.12,<0.16"
python -m venv .venv-comet
.venv-comet/bin/python -m pip install -r envs/comet-requirements.txt
```

Use the module's Python, not `/usr/bin/python3.11`: Triton compiles a CUDA
extension at runtime and needs the matching `Python.h` development headers.
The launcher and workers load the same Python 3.11 module and check the headers.

First verify both environments on a GPU (run from the repository root):

```bash
mkdir -p slurm-logs
sbatch slurm/layer_protection_smoke.sbatch
```

Check `slurm-logs/boundary-smoke-<jobid>.out` for two `GPU smoke PASS` lines.
Then submit preparation and the experiment with logs in files:

```bash
sbatch slurm/layer_protection_submit.sbatch \
  "/scratch-shared/$USER/layer-protection-seed1234-retry1"
```

This immediately prints the preparation job ID. Preparation runs on a CPU compute
node and submits the experiment DAG; its output goes to
`slurm-logs/boundary-prepare-<jobid>.out`. Stage logs and results go under the
experiment directory. Always use a fresh directory after a failed experiment;
old snapshots are immutable and do not receive fixes made to this checkout.
Cached model downloads are reused when the same `HF_HOME` is used.

Run these commands from the repository on a **login node**. Set `HF_HOME` to the
shared cache used by your existing runs if it is already populated. Choose a new
experiment directory on shared scratch, visible to both login and compute nodes.
The default run creates six pruned checkpoints plus evaluation artifacts, so allow
roughly 100 GB in addition to the shared model/metric cache.

```bash
# One command prepares and submits everything (use a fresh directory).
bash scripts/run_layer_protection.sh run \
  "/scratch-shared/$USER/layer-protection-seed1234"
```

Alternatively, separate preparation and submission for inspection:

```bash
# Optional: inspect all budgets without downloads, GPUs or Slurm submissions.
bash scripts/run_layer_protection.sh plan

EXPERIMENT="/scratch-shared/$USER/layer-protection-seed1234"

# Downloads/cache checks, calibration, frozen test strings, configs and code.
bash scripts/run_layer_protection.sh prepare "$EXPERIMENT"

# Optional: inspect the complete job graph and resource requests.
bash scripts/run_layer_protection.sh submit "$EXPERIMENT" --dry-run

# Queue the whole experiment, including final reports.
bash scripts/run_layer_protection.sh submit "$EXPERIMENT"
```

`PYTHON=/absolute/path/to/python` overrides `.venv/bin/python` for the launcher.
`prepare --comet-python /absolute/path/to/.venv-comet/bin/python` selects another
COMET environment. `submit --partition gpu_h100` changes the GPU partition;
`--cpu-partition` defaults to `rome`, and GPU jobs default to an eight-hour limit
(`--time`). Every benchmark uses an exclusive node and five timed repeats at
batch sizes 1 and 8 with a fixed 128-token output budget. Benchmarks are serialized;
pruning/generation/scoring jobs can run concurrently. Exclusive benchmarks incur
the cluster's exclusive-node allocation cost.

Preparation freezes code and inputs under the experiment directory, including the
exact evaluation strings, calibration manifest/fingerprint, resolved dense Hub
snapshot path, and environment information. Jobs verify input hashes and run the
frozen code offline. Keep the model cache and Python environments available and
unchanged until all jobs complete. No checkpoint is overwritten and the launcher
refuses duplicate submissions. `jobs.json` records every successfully submitted
job immediately, including if submission subsequently fails.

Watch `squeue -u "$USER"` and the experiment's `logs/` directory. Failed jobs cancel
their dependent jobs via `afterok` and `--kill-on-invalid-dep=yes`.

## Reading the results

Everything is under `$EXPERIMENT/reports/`:

- `comparison.md`: the four primary protected-versus-unprotected contrasts,
  macro COMET deltas, paired 95% confidence intervals, raw p-values and Holm-adjusted
  p-values across these four contrasts.
- `comparisons.json`: complete per-direction BLEU/chrF++/COMET bootstrap results
  plus macro COMET results. Macro resampling is paired within each direction and
  weights directions equally, even if their test-set sizes differ.
- `per-direction-deltas.csv`: direct quality deltas and p-values versus the
  unprotected uniform control. Direction-level tests are exploratory, unadjusted.
- `systems.csv`: aggregate/per-direction quality, behavioural failures,
  parameter counts, memory and throughput for every system.
- `budget-audit.json`: confirmation that protected layers remain full-width and
  all five conditions have identical realised unit removals and loaded parameters.
- `against-dense/` and `against-control/`: Markdown/CSV/LaTeX quality, behaviour,
  efficiency and structure tables with ratios to the indicated reference.
  Significance lives in `comparison.md` and `comparisons.json`.

The report refuses incomplete scores/benchmarks, duplicate systems, different
measurement settings, misaligned source/reference records, missing/nonfinite COMET
segment scores and unmatched compression budgets. Raw hypotheses, per-segment
scores, benchmark repeats, descriptors and manifests remain in the same experiment
folder, so the comparisons can be recomputed.

The primary criterion is macro COMET at equal actual compression. Inspect per-language
and behavioural results for regressions hidden by the mean. Confidence intervals
cover test-segment sampling, not variation between calibration seeds. Avoid speed
claims from differences smaller than benchmark variation.

## Pilot versus validation and confirmation

The default is the existing first-300-per-direction WMT pilot, with greedy decoding.
Treat the five predeclared comparisons as a fixed pilot. Do not repeatedly choose
new masks on that test set. For mask tuning, supply a separate validation suite:

```bash
bash scripts/run_layer_protection.sh prepare "$EXPERIMENT" --suite path/to/validation.yaml
```

The suite must include `de-en` for the efficiency benchmark. Calibration is checked
against full WMT22 and against the selected evaluation sources before jobs run.
Use a fresh directory and `--seed 5678` to repeat with a different calibration draw
and pruning seed. Use `--suite configs/suites/alma10-greedy.yaml` for full WMT or
`--suite configs/suites/flores200-10dir-greedy.yaml` for out-of-domain confirmation.
`--no-global-reference` skips the contextual global model. None of these options
changes the five predeclared protection masks.

## Failure recovery and rerunning reports

Preparation refuses an existing directory; a failed preparation has no completed
`experiment.json` and is not submittable. Correct the reported issue and choose a
fresh directory (downloads already cached can be reused).

After a failed submission or GPU job, inspect `jobs.json` and logs before doing
anything else. Do not call `submit` again and duplicate running work. Individual
stages can be resubmitted explicitly using the saved code, for example:

```bash
sbatch --partition gpu_a100 --gpus 1 --cpus-per-task 18 --time 08:00:00 \
  --output "$EXPERIMENT/logs/retry-score-%j.out" \
  "$EXPERIMENT/source/slurm/layer_protection.sbatch" \
  "$PWD/.venv/bin/python" "$EXPERIMENT" score alma-7b-boundary-first4
```

Use the same Python environments and `HF_HOME` recorded in `experiment.json`.
A failed pruning job may leave an incomplete checkpoint directory; the stage
refuses to overwrite it. Preserve the failed artifacts and start a fresh experiment
rather than mixing checkpoints from attempts. Generation and scoring can resume
completed outputs within the unchanged experiment.

After all stages complete, regenerate reports on a CPU allocation:

```bash
sbatch --partition rome --cpus-per-task 16 --time 02:00:00 \
  --output "$EXPERIMENT/logs/retry-report-%j.out" \
  "$EXPERIMENT/source/slurm/layer_protection.sbatch" \
  "$PWD/.venv/bin/python" "$EXPERIMENT" report all
```
