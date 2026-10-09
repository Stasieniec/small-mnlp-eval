#!/usr/bin/env bash
#SBATCH --job-name=g-alma-7b-slimgpt20-ref-multi-oldcode
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=slurm-logs/grid/canary-oldcode-%j.out
#
# Prune with the pre-fix code from a worktree of main (OLD_TREE), then generate
# and score with this checkout, so only the pruning code differs from the
# grid's alma-7b-slimgpt20-ref-multi.
set -euo pipefail
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=${HF_HOME:-/scratch-shared/${USER}/hf_home}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
OLD_TREE=${OLD_TREE:-/scratch-shared/${USER}/oldcode-main}
PRUNE_OUT=/scratch-shared/${USER}/checkpoints
SPEC=configs/prune/canary/slimgpt20-ref-multi-oldcode.yaml
NAME=alma-7b-slimgpt20-ref-multi-oldcode
SUITE=configs/suites/alma10-greedy-300.yaml
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -f configs/models/${NAME}.yaml ]]; then
    PYTHONPATH="${OLD_TREE}/src" ./.venv/bin/python -c "import mnlp_eval, sys; assert mnlp_eval.__file__.startswith('${OLD_TREE}'), mnlp_eval.__file__"
    PYTHONPATH="${OLD_TREE}/src" ./.venv/bin/python -m mnlp_eval.cli prune --spec "${SPEC}" \
        --out "${PRUNE_OUT}" --subnetwork-dir subnetworks --model-config-dir configs/models
fi
./.venv/bin/mnlp-eval generate --model "configs/models/${NAME}.yaml" --suite "${SUITE}"
RUN_DIR=$(./.venv/bin/mnlp-eval run-dir --model "configs/models/${NAME}.yaml" --suite "${SUITE}")
./.venv/bin/mnlp-eval score --run "${RUN_DIR}" --groups surface
./.venv-comet/bin/mnlp-eval score --run "${RUN_DIR}" --groups neural
