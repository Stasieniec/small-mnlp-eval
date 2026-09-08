# Running on Snellius

Compute nodes have no internet access. Everything a batch job touches must be
in `HF_HOME` before the job starts, or it fails several minutes in and burns
an allocation.

## One-time setup, on a login node

```bash
# Put the cache on scratch, not in home. Model and metric checkpoints run to
# tens of gigabytes and home quota is small.
export HF_HOME=/scratch-shared/$USER/hf_home
mkdir -p "$HF_HOME"

# Add this to your shell profile so batch scripts inherit it.
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

pip install --user uv
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gen,quant,surface,report]"
uv venv --python 3.11 .venv-comet
uv pip install --python .venv-comet/bin/python -r envs/comet-requirements.txt
```

## Prefetch, on a login node

Once per environment, because each holds different metric checkpoints.
COMETKiwi and XCOMET are gated, so accept their licences on the Hub first and
export `HF_TOKEN`.

```bash
bash slurm/prefetch.sh
```

## Submit

```bash
# One job per (model, direction), then scoring, then the report.
bash slurm/submit_sweep.sh configs/suites/wmt22-6dir-beam5.yaml \
    configs/models/alma-7b-r.yaml \
    configs/models/alma-7b-r-bnb-nf4.yaml
```

`submit_sweep.sh` chains the stages with dependencies, so scoring starts only
after generation for that model succeeds, and the report runs last.

## Notes

**Scratch expires.** Snellius scratch has an expiry policy and no backup. Copy
finished run directories to home or the archive:

```bash
rsync -a runs/ $HOME/mnlp-runs/
```

Run directories are small: hypotheses, scores and manifests, no weights.

**Runs are idempotent.** A run's identity is a hash of its model, data and
decode settings, so resubmitting a job array after a partial failure skips the
directions that already finished. Add `--overwrite` to force regeneration.

**Cost.** The six-direction WMT22 suite is 10,074 segments. At beam 5 a bf16
7B model takes roughly 3 to 6 hours on one A100, so a sweep of five compression
variants is 15 to 30 GPU-hours for generation alone. Use
`configs/suites/wmt22-6dir-greedy.yaml` for exploratory work: greedy decoding
is about three times cheaper, and the report will refuse to mix its results
with beam-5 numbers.

**Efficiency numbers need a quiet node.** `bench` records SM clock and
temperature, but a node shared with another job still produces noisier
latency than an exclusive one. If speedup is a headline claim, request the
whole node.
