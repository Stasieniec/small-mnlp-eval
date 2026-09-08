#!/usr/bin/env bash
# Create the MetricX-24 scoring environment.
#
# MetricX needs its own environment because upstream pins
# transformers==4.30.2, and it needs a clone rather than an install because
# google-research/metricx ships no pyproject.toml and no setup.py. A plain
# `pip install git+...` fails with "does not appear to be a Python project".
set -euo pipefail

cd "$(dirname "$0")/.."

VENV=${VENV:-.venv-metricx}
SRC=${METRICX_SRC:-third_party/metricx}
REVISION=${METRICX_REVISION:-main}

if [[ ! -d "${SRC}/.git" ]]; then
    echo "Cloning google-research/metricx into ${SRC}"
    mkdir -p "$(dirname "${SRC}")"
    git clone --depth 1 --branch "${REVISION}" \
        https://github.com/google-research/metricx.git "${SRC}"
else
    echo "Using existing clone at ${SRC}"
fi

echo "Creating ${VENV}"
uv venv --python 3.11 "${VENV}"
uv pip install --python "${VENV}/bin/python" -r envs/metricx-requirements.txt

# The clone root, not the metricx24 subdirectory: the package is imported as
# `metricx24`, so its parent has to be on the path.
METRICX_PATH=$(cd "${SRC}" && pwd)

echo
echo "Verifying"
if PYTHONPATH="${METRICX_PATH}" "${VENV}/bin/python" -c "
from metricx24 import models  # noqa: F401
from mnlp_eval.metrics.base import group_availability
served, reason = group_availability()['metricx']
print('  metricx group available:', served)
raise SystemExit(0 if served else 1)
"; then
    echo
    echo "Done. Score with:"
    echo "  PYTHONPATH=${METRICX_PATH} ${VENV}/bin/mnlp-eval score --run <dir> --groups metricx"
    echo
    echo "Export it once instead, if you prefer:"
    echo "  export METRICX_SRC=${METRICX_PATH}"
    echo "slurm/score.sbatch reads METRICX_SRC and defaults to \$PWD/third_party/metricx."
else
    echo "Verification failed." >&2
    exit 1
fi
