# Run scores

`manifest.json`, `scores.surface.json` and `scores.neural.json` of every run behind
`results/grid-2026-10-06/` (grid, seed-5678 replicate, FLAP allocation ablation,
canaries, repairs, dense baseline), copied from `runs/` on scur0560. The
per-segment COMET scores in `scores.neural.json` are what the paired bootstraps
resample; segments pair across runs by direction and position. Hypotheses are not
copied (710 MB); they remain under `runs/` on that account.
