#!/usr/bin/env bash
#SBATCH --job-name=mnlp-generate
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:00:00
#SBATCH --output=slurm-logs/generate-%A_%a.out
#
# One array task per direction, all contributing to ONE run.
#
# The direction is resolved here from the suite rather than passed in, because
# commas inside an `sbatch --export` value are Slurm's own delimiter between
# assignments: exporting DIRECTIONS=de-en,en-de,ru-en arrives as just "de-en"
# and the remaining names are read as variables to copy from the submitting
# environment. See SchedMD bug 4213.
#
# Sharding uses --only-direction, not --directions. The suite's direction list
# is part of run identity, so --directions would give every task a different
# run id and split one sweep into six one-direction runs that the report
# refuses to combine.
set -euo pipefail

module load 2025
module load Python/3.12.3-GCCcore-13.3.0

cd "${SLURM_SUBMIT_DIR}"

: "${MODEL_CONFIG:?set by slurm/submit_sweep.sh}"
: "${SUITE_CONFIG:?set by slurm/submit_sweep.sh}"

# Compute nodes may not reach the Hub, and even where they can, a run that
# silently downloads is a run whose timings are not comparable. Everything is
# prefetched by slurm/prefetch.sh from a login node.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

mapfile -t PAIRS < <(./.venv/bin/mnlp-eval suite-directions --suite "${SUITE_CONFIG}")
INDEX=${SLURM_ARRAY_TASK_ID:-0}
if (( INDEX >= ${#PAIRS[@]} )); then
    echo "array task ${INDEX} exceeds the ${#PAIRS[@]} direction(s) in ${SUITE_CONFIG}" >&2
    exit 1
fi
DIRECTION="${PAIRS[INDEX]}"

echo "model=${MODEL_CONFIG} suite=${SUITE_CONFIG} direction=${DIRECTION}"

./.venv/bin/mnlp-eval generate \
    --model "${MODEL_CONFIG}" \
    --suite "${SUITE_CONFIG}" \
    --only-direction "${DIRECTION}" \
    --deterministic-check "${DETERMINISTIC_CHECK:-0}"
