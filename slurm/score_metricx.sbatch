#!/usr/bin/env bash
#SBATCH --job-name=mnlp-metricx
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=slurm-logs/eval/%x-%j.out
#
# MetricX-24 over many run directories in one process, so the model loads once.
# Separate from generation because MetricX pins transformers==4.30.2 and needs
# its own environment (scripts/setup_metricx_env.sh); a MetricX failure then
# cannot cost a run its translations or COMET scores.
#
#   sbatch --export=ALL,RUN_LIST=<file with one run directory per line> \
#       slurm/score_metricx.sbatch
#
# Runs are scored with allow_partial, because specialists are deliberately
# evaluated on their own directions only. A run whose scores.metricx.json
# exists is skipped, so resubmitting the same list resumes it.
set -euo pipefail
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=${HF_HOME:-/scratch-shared/${USER}/hf_home}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
: "${RUN_LIST:?file with one run directory per line}"
export PYTHONPATH="${METRICX_SRC:-${PWD}/third_party/metricx}"
METRICS_CONFIG=${METRICS_CONFIG:-configs/metrics/final.yaml}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

./.venv-metricx/bin/python - "${RUN_LIST}" "${METRICS_CONFIG}" <<'PY'
import sys, time
from pathlib import Path

from mnlp_eval.runspec import RunPaths
from mnlp_eval.config import MetricsSpec, load_yaml_config
from mnlp_eval.score import score_run

runs = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
metrics = MetricsSpec.from_dict(load_yaml_config(sys.argv[2]))
failed = []
for index, run in enumerate(runs, 1):
    root = Path(run)
    if (root / "scores.metricx.json").is_file():
        print(f"[{index}/{len(runs)}] {root.name}: already scored", flush=True)
        continue
    start = time.time()
    try:
        score_run(RunPaths(root), metrics, groups=["metricx"], allow_partial=True)
    except Exception as exc:  # one bad run must not stop the rest
        failed.append(root.name)
        print(f"[{index}/{len(runs)}] {root.name}: FAILED {exc!r}", flush=True)
        continue
    print(f"[{index}/{len(runs)}] {root.name}: {time.time() - start:.0f}s", flush=True)
print(f"done; {len(failed)} failed: {failed}", flush=True)
sys.exit(1 if failed else 0)
PY
