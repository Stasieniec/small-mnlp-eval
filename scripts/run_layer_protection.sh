#!/usr/bin/env bash
# Run from the repository checkout. See docs/layer-protection.md.
# prepare downloads on a login node; submit queues the complete experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/layer_protection_env.sh
PYTHON=${PYTHON:-./.venv/bin/python}
if [[ ! -x "${PYTHON}" ]]; then
    echo "Generation Python not found: ${PYTHON}. See docs/layer-protection.md for setup." >&2
    exit 1
fi
check_python_headers "${PYTHON}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON}" -m mnlp_eval.experiments.layer_protection "$@"
