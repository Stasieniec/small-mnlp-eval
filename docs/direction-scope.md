# Direction-specific pruning pilot

Test whether activations from A → B alone preserve A → B better than activations
from both directions of its pair or from all ten ALMA directions. Default:
German and Icelandic, both directions; FLAP; 20% pruning; uniform allocation;
prompt+reference calibration; no repair. Uniform allocation keeps layer widths
fixed across scopes, so calibration cannot change the compression budget.
SlimGPT is available with `--method slimgpt` in a separate experiment directory.

## Design and cost

Eight systems: dense ALMA, one multilingual control, two pair controls, four
direction-specific models. Every pruned model sees **1,280 calibration segments**:
128 per direction for multilingual, 640 per direction for pair-only, 1,280 for
direction-only. Draws use the same calibration seed. Segment budgets match;
token budgets do not, and token-length differences remain a limitation.
Calibration includes references so target-language production is represented.
Contamination checks must pass for every scope and the frozen evaluation set.

Evaluate each model on the same first 200 sentences in each of four directions:
6,400 translations total. Greedy decoding. COMET is primary; BLEU and chrF++
are descriptive. The eight primary contrasts compare direction-only with its
pair control and multilingual control on the target direction. Paired bootstrap
with 2,000 resamples and Holm correction; other directions measure transfer.
One calibration seed is a pilot: repeat with another seed before adopting a
scope. Do not tune on these evaluation segments.

Each system gets one shared-node A100 allocation for pruning, generation and
scoring, with a two-hour limit. Subprocesses release each model before loading
the next. No exclusive-node benchmarks, repair, or full ten-direction evaluation.
Default maximum GPU concurrency is two; `--concurrency 1` serialises the study.
Eight jobs imply a **16 allocated GPU-hour ceiling**, plus a 30-minute 16-CPU
report job. Actual allocation ends when each job finishes. SBU cost depends on
the site's accounting rate; this is a resource ceiling, not a measured runtime
or an SBU quote. One pair reduces the study to five systems / ten GPU-hours.
Retain checkpoints on scratch: seven pruned models require roughly 75–95 GB,
plus source/data/runs. The dense checkpoint is shared through the Hub cache.

## Run on Snellius

Use the Python/COMET environments and scratch cache described in
[layer-protection.md](layer-protection.md). From the repository checkout:

```bash
bash scripts/run_direction_scope.sh plan
bash scripts/run_direction_scope.sh prepare /scratch-shared/$USER/direction-scope-seed1234
bash scripts/run_direction_scope.sh submit /scratch-shared/$USER/direction-scope-seed1234 --dry-run
bash scripts/run_direction_scope.sh submit /scratch-shared/$USER/direction-scope-seed1234
```

Preparation downloads and freezes code, model snapshot, evaluation text,
calibration sets, configs and SHA-256 input hashes. It submits no jobs.
All GPU jobs run offline against that snapshot; modified inputs are rejected.
`jobs.json` records IDs after every submission and prevents accidental duplicate
submission, including after a partial submission failure. `afterok` dependencies
limit concurrency and prevent reporting incomplete runs. A failed job blocks its
lane and the report; inspect the recorded IDs before resubmitting individual
stages. Stages refuse to overwrite existing checkpoints. For a generation or
scoring failure, run the existing CLI stages against the frozen configs rather
than rerunning pruning.

For a cheaper initial check, pass `--pairs de` during preparation. For another
seed or method, use a fresh directory. `plan` accepts the same `--pairs`,
`--budget` and `--method` options without downloads or GPU access. Budgets must
be positive multiples of ten. Small budgets can make SlimGPT reconstruction
unreliable; the default matches the existing multilingual pilot's sample count.

Outputs: `reports/comparison.md`, `comparisons.json`, `by-direction.csv`,
`systems.csv`, and `budget-audit.json`. Reporting rejects incomplete scores,
unaligned segments, different evaluation/metric settings, and unequal realised
pruning budgets. No GPU jobs run merely by creating this implementation.
