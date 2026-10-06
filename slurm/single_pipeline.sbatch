#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --output=slurm-logs/grid/%x-%j.out
#
# Prune one config, then generate and score it on SUITE_CONFIG, in one
# allocation. For canaries and one-off systems outside the grid manifest; the
# grid itself goes through slurm/grid_pipeline.sbatch. Each stage is skipped
# when its output exists.
set -euo pipefail
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=${HF_HOME:-/scratch-shared/${USER}/hf_home}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
: "${PRUNE_CONFIG:?path to a prune config}"
SUITE_CONFIG=${SUITE_CONFIG:-configs/suites/alma10-greedy-300.yaml}
PRUNE_OUT=${PRUNE_OUT:-/scratch-shared/${USER}/checkpoints}
NAME=$(./.venv/bin/mnlp-eval prune-spec --spec "${PRUNE_CONFIG}" name)
MODEL_CONFIG=configs/models/${NAME}.yaml
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "name=${NAME} start=$(date +%T)"

if [[ ! -f "${MODEL_CONFIG}" || ! -f "${PRUNE_OUT}/${NAME}/subnetwork.json" ]]; then
    ./.venv/bin/mnlp-eval prune --spec "${PRUNE_CONFIG}" --out "${PRUNE_OUT}" \
        --subnetwork-dir subnetworks --model-config-dir configs/models
fi
echo "pruned $(date +%T)"
./.venv/bin/mnlp-eval generate --model "${MODEL_CONFIG}" --suite "${SUITE_CONFIG}"
echo "generated $(date +%T)"
RUN_DIR=$(./.venv/bin/mnlp-eval run-dir --model "${MODEL_CONFIG}" --suite "${SUITE_CONFIG}")
./.venv/bin/mnlp-eval score --run "${RUN_DIR}" --groups surface
./.venv-comet/bin/mnlp-eval score --run "${RUN_DIR}" --groups neural
echo "scored $(date +%T) ${RUN_DIR}"
