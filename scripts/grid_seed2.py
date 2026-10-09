"""A second calibration draw for the SlimGPT reference-calibration slice of the grid.

The grid's bootstrap intervals cover the test segments, not which calibration
segments were drawn. This writes the same 48 SlimGPT models (20/30/40 percent;
multi, five pairs, ten directions; prompt + reference) on calibration sets
drawn with seed 5678 instead of 1234, named ``alma-7b-s2-<grid name>`` so
``scripts/grid_report.py --manifest configs/grid/manifest-seed2.json`` reads
them unchanged.

    ./.venv/bin/python scripts/grid_seed2.py
    for f in configs/calibration/seed2/*.yaml; do mnlp-eval calibration --spec $f; done
    then submit configs/prune/seed2/*.yaml through slurm/single_pipeline.sbatch
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

SEED = 5678
SUFFIX = "s2"
DIRECTIONS = [
    "cs-en",
    "de-en",
    "is-en",
    "ru-en",
    "zh-en",
    "en-cs",
    "en-de",
    "en-is",
    "en-ru",
    "en-zh",
]
LANGUAGES = ["cs", "de", "is", "ru", "zh"]


def main() -> int:
    grid = json.loads(Path("configs/grid/manifest.json").read_text())
    calibration_dir = Path("configs/calibration/seed2")
    prune_dir = Path("configs/prune/seed2")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    prune_dir.mkdir(parents=True, exist_ok=True)

    # The grid's sets, redrawn: same sizes and filters, another seed.
    sets = {"multi-10dir": "../multi-10dir.yaml"}
    sets.update({f"pair-{lang}-640": f"../pair-{lang}-640.yaml" for lang in LANGUAGES})
    sets.update({f"dir-{d}-1280": f"../dir-{d}-1280.yaml" for d in DIRECTIONS})
    for name, parent in sets.items():
        payload = {"extends": parent, "name": f"{name}-{SUFFIX}", "seed": SEED}
        header = f"# {name} redrawn with seed {SEED}. Written by scripts/grid_seed2.py.\n"
        (calibration_dir / f"{name}-{SUFFIX}.yaml").write_text(
            header + yaml.safe_dump(payload, sort_keys=False)
        )

    models = []
    for entry in grid["models"]:
        if entry["method"] != "slimgpt" or entry["tag"] != "ref":
            continue
        spec = yaml.safe_load(Path(entry["prune_config"]).read_text())
        name = entry["name"].replace("alma-7b-", f"alma-7b-{SUFFIX}-", 1)
        spec["name"] = name
        spec["calibration"] = f"{spec['calibration']}-{SUFFIX}"
        spec["seed"] = SEED
        path = prune_dir / f"{name}.yaml"
        header = (
            f"# Seed-{SEED} replicate of {entry['name']}: calibration redrawn, all else equal. "
            "Written by scripts/grid_seed2.py.\n"
        )
        path.write_text(header + yaml.safe_dump(spec, sort_keys=False))
        models.append(
            {
                **entry,
                "name": name,
                "prune_config": str(path),
                "model_config": f"configs/models/{name}.yaml",
            }
        )

    manifest = {**grid, "description": f"seed-{SEED} replicate of SlimGPT ref", "models": models}
    manifest["count"] = len(models)
    Path("configs/grid/manifest-seed2.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(sets)} calibration configs and {len(models)} prune configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
