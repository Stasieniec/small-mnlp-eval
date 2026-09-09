# Running on Snellius

## Prefetching, and why

Everything a batch job touches should be in `HF_HOME` before the job starts.

Two reasons, and it is worth being precise about which is which. The first is
reproducibility: a run that downloads mid-job is a run whose timings are not
comparable, and whose weights might differ from the last one if a Hub
repository moved. That reason is solid and applies regardless. The second is
network availability, and here the framework's earlier claim that Snellius
compute nodes have no internet access was not verified. SURF's own AI guide
downloads a Hugging Face dataset from inside an `sbatch` job, and no SURF
documentation states that outbound access is blocked; only inbound is. Check
your own account before relying on it either way.

The scripts default to `HF_HUB_OFFLINE=1`, which turns a missing asset into a
fast, clear failure rather than a slow download. Override it with
`HF_HUB_OFFLINE=0` if you would rather a job fetch what it needs.

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
uv pip install --python .venv/bin/python -e ".[gen,quant,surface,report]"

uv venv --python 3.11 .venv-comet
uv pip install --python .venv-comet/bin/python -r envs/comet-requirements.txt

# Optional, for the final report. Not a pip install; see docs/environments.md.
bash scripts/setup_metricx_env.sh
```

The `2025` stack is deliberate. `2023` is marked deprecated in SURF's software
documentation and provided as-is without support.

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
    configs/models/alma-7b-prune50-multi.yaml \
    configs/models/alma-7b-prune50-multi-lora.yaml
```

Per model the chain is: a generation array with one task per direction, then
efficiency, then scoring. The report waits on every scoring job. `BASELINE` is
required, because every compression ratio, speedup and p-value is computed
against it and the report silently drops all of them if it cannot find it.

Two things the scripts are careful about, both of which used to be wrong:

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
