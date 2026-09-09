#!/usr/bin/env bash
# Reproduce ALMA-7B-R's published WMT22 scores, and record the delta.
#
# This is the framework's own correctness evidence. Until the delta against the
# published table is known, every compression result this repository produces
# rests on an unverified harness. Run it once, keep the output, and cite the
# delta in the report.
#
# Exact agreement is not expected. ALMA generates through the Hugging Face
# Trainer, which pads differently, and beam search is not exactly invariant to
# padding. wmt22-alma-repro disables length bucketing and matches their batch
# size to get as close as the two harnesses can come.
#
# Cost: ten directions, 17,471 segments, beam 5, bf16 7B. Budget 6 to 12 hours
# on one A100. On Snellius prefer:
#   bash slurm/submit_sweep.sh configs/suites/wmt22-alma-repro.yaml \
#        configs/models/alma-7b.yaml
set -euo pipefail

cd "$(dirname "$0")/.."

CLI=${CLI:-./.venv/bin/mnlp-eval}
COMET_CLI=${COMET_CLI:-./.venv-comet/bin/mnlp-eval}
RUNS_ROOT=${RUNS_ROOT:-runs}
SUITE=configs/suites/wmt22-alma-repro.yaml
MODEL=configs/models/alma-7b.yaml

export TOKENIZERS_PARALLELISM=false

echo "=== test set provenance ==="
# If this fails, the reproduction is meaningless: the scores would be computed
# on different data than ALMA used.
"${CLI}" verify-testset \
    --dataset haoranxu/WMT22-Test \
    --directions de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru

echo
echo "=== generation, ALMA settings ==="
"${CLI}" --runs-root "${RUNS_ROOT}" generate \
    --model "${MODEL}" --suite "${SUITE}" --deterministic-check 32

echo
echo "=== surface metrics ==="
"${CLI}" --runs-root "${RUNS_ROOT}" score --groups surface

echo
echo "=== neural metrics ==="
if [[ -x "${COMET_CLI}" ]]; then
    "${COMET_CLI}" --runs-root "${RUNS_ROOT}" score \
        --groups neural --metrics configs/metrics/full.yaml
else
    echo "COMET environment not found. See docs/environments.md." >&2
    exit 1
fi

echo
echo "=== scores ==="
./.venv/bin/python - "${RUNS_ROOT}" <<'PYEOF'
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
run = next(
    (path for path in sorted(runs_root.iterdir()) if path.name.startswith("alma-7b-r__wmt22-alma-repro")),
    None,
)
if run is None:
    raise SystemExit(f"no reproduction run found under {runs_root}")

surface = json.loads((run / "scores.surface.json").read_text())
neural_path = run / "scores.neural.json"
neural = json.loads(neural_path.read_text()) if neural_path.is_file() else {"directions": {}}

print(f"{'direction':>10}  {'BLEU':>7}  {'chrF++':>7}  {'COMET-22':>9}")
for direction in sorted(surface["directions"]):
    metrics = surface["directions"][direction]["metrics"]
    comet = (
        neural["directions"].get(direction, {}).get("metrics", {}).get("wmt22_comet_da", {})
    )
    print(
        f"{direction:>10}  {metrics['bleu']['score']:>7.2f}  "
        f"{metrics['chrf2pp']['score']:>7.2f}  "
        f"{comet.get('score', float('nan')):>9.4f}"
    )
print(
    f"{'average':>10}  {surface['aggregate']['bleu']:>7.2f}  "
    f"{surface['aggregate']['chrf2pp']:>7.2f}  "
    f"{neural.get('aggregate', {}).get('wmt22_comet_da', float('nan')):>9.4f}"
)
print()
print("sacreBLEU signature:")
first = sorted(surface["directions"])[0]
print(f"  {surface['directions'][first]['metrics']['bleu']['signature']}")
print()
print("Compare these against Table 2 of the ALMA-R paper (Xu et al., 2024) and")
print("record the delta in the project report. Paste this output verbatim.")
PYEOF
