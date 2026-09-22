# Running on Snellius

## Prefetching

Everything a batch job touches should be in `HF_HOME` before the job starts. A
run that downloads mid-job has timings that are not comparable, and weights
that may differ from the last run if a Hub repository moved.

Whether compute nodes can reach the network is a separate question and is not
verified here. SURF documents blocking inbound access, not outbound, and SURF's
own AI guide downloads from the Hub inside an `sbatch` job. Check your account
before relying on it either way.

The scripts default to `HF_HUB_OFFLINE=1`, which turns a missing asset into a
fast failure rather than a slow download. Set `HF_HUB_OFFLINE=0` if you would
rather a job fetch what it needs.

SURF also maintains a shared cache at `/projects/2/managed_datasets/hf_cache_dir`,
which may already hold what you need.

## One-time setup, on a login node

```bash
# Scratch, not home: checkpoints run to tens of gigabytes and home quota is
# small. See the expiry warning below.
export HF_HOME=/scratch-shared/$USER/hf_home
mkdir -p "$HF_HOME"
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

module load 2025
module load Python/3.12.3-GCCcore-13.3.0

pip install --user uv
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gen,prune,quant,surface,report]"

uv venv --python 3.11 .venv-comet
uv pip install --python .venv-comet/bin/python -r envs/comet-requirements.txt

# Optional, for the final report. Not a pip install; see docs/environments.md.
bash scripts/setup_metricx_env.sh
```

The `2025` stack is deliberate. `2023` is marked deprecated in SURF's software
documentation and provided as-is without support.

## Calibration and pruning

Both happen before any generation, and the order matters: the calibration set
fixes what every pruning criterion optimises for, so a change to it invalidates
every subnetwork selected on the old one.

`mnlp-eval calibration` downloads ALMA's training data and checks the drawn
segments against the test set, so it runs on a **login node**:

```bash
./.venv/bin/mnlp-eval calibration --spec configs/calibration/multi-10dir.yaml
./.venv/bin/mnlp-eval calibration --spec configs/calibration/pair-de.yaml
```

It hard-fails on any test-set collision. Do not work around it: a subnetwork
selected on contaminated data makes every quality number downstream of it
indefensible.

Then queue the pruning jobs:

```bash
bash slurm/submit_prune.sh              # every config in configs/prune
bash slurm/submit_prune.sh configs/prune/flap-50-multi.yaml
```

One job per config, run in parallel rather than chained: each reads the dense
checkpoint and writes its own subnetwork. Each prints a manifest with the
sparsity it actually achieved, which will differ slightly from the request
because heads and channels are whole units.

Checkpoints go to `/scratch-shared/$USER/checkpoints`, overridable with
`PRUNE_OUT`. They are 13.5 GB each at bfloat16 and the home quota is small.
The descriptors go to `subnetworks/` in the repository, because they are small,
they are the only record of which positions a run chose, and they cannot be
recovered from a checkpoint afterwards.

The pruning stage also writes `configs/models/<name>.yaml`, so a pruned system
is ready to hand to `submit_sweep.sh` with no further editing. It goes beside
the other model configs rather than next to the weights because `extends`
resolves relative to the file that declares it.

Deliberately not chained into the evaluation sweep. That sweep is well over a
hundred GPU-hours, and two things are worth checking first: the achieved
sparsity in each manifest, and

```bash
./.venv/bin/mnlp-eval overlap --subnetwork-dir subnetworks/
```

which should show the three criteria agreeing above chance but not perfectly.
Near-zero excess over chance would mean the descriptors are being written
against different index conventions; near-total overlap would mean the three
criteria are not distinguishable on this data and the comparison has nothing
to say.

**Cost per method.** FLAP is minutes: one forward pass and two vectors per
projection. LLM-Pruner runs a backward pass, so budget under an hour and lower
`batch_size` if it does not fit. SlimGPT is the expensive one at roughly an
hour, with two passes over the calibration set, a Hessian and Cholesky inverse
per projection, and a per-column compensation sweep. `prune.sbatch` asks for
four hours, which covers the slowest with room for the checkpoint write.

One 40 GB A100 is enough but not spacious for SlimGPT: the dense model is
13.5 GB, and `down_proj`'s Hessian alone is 11008 squared in float32. If it
does not fit, lower `segments_per_direction` in the calibration config. Do not
lower `max_length`: it changes what the Hessian measures and the methods stop
being comparable. See [docs/pruning.md](../docs/pruning.md).

## Prefetch, on a login node

Once per environment, because each holds different metric checkpoints.
COMETKiwi and XCOMET are gated, so accept their licences on the Hub and export
`HF_TOKEN` first.

```bash
bash slurm/prefetch.sh
```

It runs with `--strict`, so a failed download is an error here rather than a
surprise hours later inside a GPU job.

## Submit

```bash
BASELINE=alma-7b bash slurm/submit_sweep.sh \
    configs/suites/alma10-greedy.yaml \
    configs/models/alma-7b.yaml \
    configs/models/alma-7b-flap50-multi.yaml \
    configs/models/alma-7b-llm-pruner50-multi.yaml \
    configs/models/alma-7b-slimgpt50-multi.yaml
```

`submit_prune.sh` prints this command with the configs its jobs emitted, so it
does not have to be assembled by hand.

Per model the chain is: a generation array with one task per direction, then
efficiency, then scoring. The report waits on every scoring job. `BASELINE` is
required, because every compression ratio, speedup and p-value is computed
against it and the report silently drops all of them if it cannot find it.

Two things the scripts are careful about:

**Array tasks shard with `--only-direction`, not `--directions`.** A suite's
direction list is part of run identity, so `--directions` would give each task
its own run id and split one sweep into ten one-direction runs the report
refuses to combine. `--only-direction` restricts what a task generates while
leaving the run identity alone, so every task fills in part of the same run.

**Nothing containing a comma is passed through `--export`.** Commas are
Slurm's own delimiter between assignments inside an `--export` value, so
`DIRECTIONS=de-en,en-de` arrives as just `de-en` and the rest are read as
variable names to copy from the submitting environment. Batch scripts resolve
the direction list with `mnlp-eval suite-directions` instead.

## Notes

**Scratch expires after 14 days.** `/scratch-shared` sweeps files whose
contents have not changed in 14 days, with no notification and no guarantee of
even that long. `HF_HOME` holds read-only checkpoints, which is exactly the
access pattern that gets swept, so a long project needs to either re-prefetch
periodically or keep the cache somewhere durable. There is no backup. Copy
finished run directories out:

```bash
rsync -a runs/ $HOME/mnlp-runs/
```

Run directories hold hypotheses, scores and manifests, not weights, so they are
small.

**Runs are idempotent.** A run's identity is a hash of its model, data and
decode settings, so resubmitting an array after a partial failure skips the
directions that already finished. Stage records are one file per direction, so
concurrent tasks cannot overwrite one another. Add `--overwrite` to force
regeneration. `bench` has no skip check and re-runs on resubmission.

**Efficiency runs request an exclusive node.** Latency is the measurement, and
a quarter-node allocation shares the physical GPU node with up to three other
jobs, whose contention moves throughput by more than some compression methods
do. `bench.sbatch` therefore asks for the whole node and pins
`CUDA_VISIBLE_DEVICES=0`. The run records SM clock, temperature and power draw
so a throttled measurement is at least visible.

**Cost.** The ten-direction suite is 17,491 segments. At beam 5 a bf16 7B model
takes roughly 6 to 10 hours on one A100, so the sweep this project needs, a
baseline plus several sparsity levels plus five pair-specific subnetworks plus
their repaired versions, is well over a hundred GPU-hours of generation alone
at beam 5.

Use `configs/suites/alma10-greedy.yaml` for the sweep. Greedy decoding is about
three times cheaper, and the transfer matrix needs every subnetwork evaluated
on all ten directions, which is what makes the sweep expensive rather than the
per-system cost. Reserve `alma10-beam5` for the final systems and the headline
table, and note that the report refuses to mix results from the two.

**Partitions.** `gpu_a100` at `--gpus=1 --cpus-per-task=18` is one quarter of a
72-core, 4-GPU node, which is the correct request shape. The `rome` partition
bills in eighths of a 128-core node, so `report.sbatch` asks for 16 cores
rather than 4 because they cost the same.
