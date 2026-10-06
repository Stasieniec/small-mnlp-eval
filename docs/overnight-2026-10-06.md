# Final pilot grid, night of 6 October 2026

Unattended run on scur0560 (Snellius), branch `exp/final-grid`. Times are CEST.
The morning write-up comes first; the chronological log with job ids is below.

## Morning write-up

### Status

- **Everything in the plan ran, and finished at 22:54.** That covers:
  - all 192 grid models: pruned, generated on all ten directions of `alma10-greedy-300` and scored (BLEU, chrF++, COMET-22);
  - five LoRA repairs of multi models.
- **No grid job failed.** Six early jobs failed on two bugs, both fixed: a duplicate prompt in the ru-en calibration data, and a typo in my canary script.
- **Extras, beyond the agreed plan:**
  - a FLAP allocation ablation (6 models);
  - a second calibration draw for SlimGPT with reference calibration (48 models);
  - LoRA on the five SlimGPT 40% pair models, each on its own pair's data.
- **Compute:** 78 GPU-hours, about 11,400 SBU (390k left).
- **Disk:** checkpoints are in `/scratch-shared/scur0560/checkpoints/`, 2.4 TB. They stay until about 20 October (14 days untouched).
- **Two bugs fixed before anything ran:**
  - SlimGPT's compensation was wrong, so every earlier SlimGPT number used weaker pruning than the paper. Details under "What changed in the code".
  - FLAP's uniform budget explains the collapse in David's direction-scope experiment.
- **Branch:** `exp/final-grid`, pushed. main and the teammates' PRs are untouched.

### Results (pilot: first 300 segments of each WMT22 direction, greedy, macro over ten)

COMET-22, with BLEU in brackets. For pair and dir, each direction is translated by its matching specialist: `pair-<l>` for both directions of l, `dir-<d>` for d. All 192 models are evaluated on all ten directions. Dense ALMA-7B scores 0.8474 (30.35).

| method | calib | removed | multi | pair | dir |
| --- | --- | --- | --- | --- | --- |
| SlimGPT | ref | 20% | **0.8374** (27.72) | 0.8369 (27.97) | 0.8371 (27.95) |
| SlimGPT | gen | 20% | 0.8360 (27.52) | **0.8377** (28.13) | 0.8373 (27.93) |
| SlimGPT | ref | 30% | 0.8219 (25.54) | 0.8271 (26.27) | **0.8282** (26.43) |
| SlimGPT | gen | 30% | 0.8229 (25.88) | **0.8276** (26.52) | 0.8267 (26.58) |
| SlimGPT | ref | 40% | 0.7914 (23.13) | **0.8083** (23.99) | 0.8076 (24.13) |
| SlimGPT | gen | 40% | 0.7905 (22.92) | **0.8092** (24.18) | 0.8081 (24.44) |
| FLAP | ref | 20% | 0.8221 (26.81) | 0.8297 (27.24) | **0.8299** (27.67) |
| FLAP | gen | 20% | 0.8209 (26.70) | 0.8291 (27.31) | **0.8306** (27.70) |
| FLAP | ref | 30% | 0.7823 (23.66) | 0.8003 (24.09) | **0.8017** (24.49) |
| FLAP | gen | 30% | 0.7830 (23.97) | 0.8001 (24.39) | **0.8006** (24.59) |
| FLAP | ref | 40% | 0.7393 (20.50) | 0.7552 (20.11) | **0.7552** (20.33) |
| FLAP | gen | 40% | 0.7392 (20.12) | 0.7543 (19.95) | **0.7557** (20.46) |

Removed means the share of attention and MLP weights. Of the whole model including embeddings, that is 19.3%, 28.8% and 38.4%; both methods match their budgets to within 0.1%.

**1. SlimGPT beats FLAP everywhere.**
- The gap is +0.015 COMET at 20% multi, +0.040 at 30% and +0.052 at 40%.
- It is significant in all 18 method contrasts (p < 0.005, paired bootstrap).
- Neither method loops or truncates much: under 1.5% of outputs hit the token budget, against 0% for dense.

**2. At equal calibration size, specialised subnetworks beat the multi-directional one from 30% up.**
- SlimGPT: pair and dir minus multi is +0.004 to +0.006 COMET at 30% and +0.016 to +0.019 at 40% (p < 0.005). At 20% the difference is not significant.
- FLAP gains already at 20% (+0.008), and +0.015 to +0.019 at 30 and 40%.
- Pair and dir models perform the same, so five pair models match ten direction models.
- This reverses the 29 September pilot. That pilot gave pair models 256 calibration segments against multi's 1,280; at an equal 1,280 the specialists win.

**3. The seed-5678 replicate reproduces the headline** (SlimGPT ref: multi, pair and dir at 20/30/40%, `results/grid-2026-10-06/seed2/`).
- Composite COMET moves by 0.000 to 0.004 between calibration draws.
- The specialist gains at 30 and 40% are significant under both draws: +0.011 to +0.014 at 40% in seed 2.

**4. Reference vs generated calibration makes no difference.**
- None of the 18 COMET contrasts is significant; all deltas are within ±0.0015 COMET.
- The dense model's own translations are far from the references (chrF++ 49 to 66, 0 to 3% exact matches; `calibration_diagnostics.md`), so this is not because the two texts are the same.
- Practical upshot: calibration needs only source sentences. ALMA's own greedy output replaces the reference at no cost.

**5. Subnetworks specialise by the language they write.**
- Direction models, 40%, mean COMET change against multi on the same direction:

  | evaluated on | SlimGPT | FLAP |
  | --- | --- | --- |
  | own direction | +0.017 | +0.016 |
  | its reverse | +0.006 to +0.009 | +0.016 to +0.018 |
  | other directions into the same target | -0.006 | -0.006 |
  | en->X directions with a different target language | -0.18 | -0.13 |

- Pair models: the own pair gains +0.017, the other eight directions lose 0.09.
- Into-English directions are robust to any specialist. Generating a language the subnetwork was not calibrated on is what breaks.
- Czech and Russian share units: the en-cs specialist loses only 0.03 on en-ru.
- Heatmaps: `results/grid-2026-10-06/plots/transfer_*`.

**6. Low-resource and distant targets are the most fragile.**
- At 40%, multi models lose most on en-is and en-zh: SlimGPT 0.748 and 0.747, FLAP 0.58 and 0.65, against dense 0.842 and 0.851.
- Into-English directions lose far less.
- The direction specialist helps en-is most: +0.044 for SlimGPT, +0.068 for FLAP.

**7. LoRA repair** (ALMA's recipe on repair-multi-clean, one epoch, 914 steps, about 1 h 45 on an A100):

| repaired model | COMET before -> after (dense 0.8474) | gap recovered | BLEU before -> after |
| --- | --- | --- | --- |
| SlimGPT ref 20% multi | 0.8374 -> 0.8381 (n.s., p = 0.51) | 7% | 27.72 -> 28.41 |
| SlimGPT gen 30% multi | 0.8229 -> 0.8324 | 39% | 25.88 -> 27.59 |
| SlimGPT ref 40% multi | 0.7914 -> 0.8193 | 50% | 23.13 -> 26.13 |
| SlimGPT gen 40% multi | 0.7905 -> 0.8179 | 48% | 22.92 -> 26.36 |
| FLAP ref 40% multi | 0.7393 -> 0.8112 | 67% | 20.50 -> 25.73 |
| SlimGPT gen 40% pair models, own-pair LoRA (composite) | 0.8092 -> 0.8236 | 38% | 24.18 -> 26.08 |

- Held-out loss fell for every model: for example 0.908 to 0.770 at SlimGPT 40%, and 1.10 to 0.80 for FLAP 40%.
- Repair gains grow with sparsity and are largest for the weaker FLAP. It ends 0.008 COMET behind repaired SlimGPT, against 0.052 before repair.
- The best 40% system is the five pair models with own-pair LoRA (0.8236). It beats the repaired multi model (0.8179), and its COMET is close to the unrepaired 30% systems.
- Own-pair repairs recover 21 to 46% of the gap on their own pair, from as few as 28 steps (Icelandic). Full tables: `results/grid-2026-10-06/repair.md`.

**8. FLAP allocation ablation (multi, COMET):**

| allocation | 20% | 30% | 40% |
| --- | --- | --- | --- |
| AL-AM (official code; used in the grid) | 0.822 | 0.782 | 0.739 |
| separate head/channel `global` | 0.832 | 0.428 | 0.568 |

The separate `global` budget is better at 20% but collapses at 30 and 40%: 55% of outputs loop and the length ratio reaches 11. So AL-AM stayed for every scope. FLAP's `uniform` budget is what collapsed in the direction-scope experiment of 5 October.

**9. The SlimGPT fix:** on the same 20% multi ref setting, the old code scores 0.8354 (27.31) and the fixed code 0.8374 (27.72). The old code was never run at 30 or 40%, where the missing compensation matters most.

### Recommendations for tomorrow's full eval

All cheap to run: every checkpoint is on scratch, and the suite is `configs/suites/alma10-greedy.yaml`.

1. **SlimGPT 20% ref multi.** The best single model at 20%; specialists add nothing there.
2. **SlimGPT 40% gen multi + LoRA.** The rescued single model at the highest sparsity.
3. **SlimGPT 40% gen pair models + own-pair LoRA.** The best 40% system; evaluate each pair model on its own two directions only.
4. **Optional:** SlimGPT 30% gen multi + LoRA for the middle of the curve, and FLAP 40% ref + LoRA as the method contrast after repair.

The calibration text doesn't matter, so pick one and say so: ref is the conventional choice, gen needs no references. Full-suite generation of a multi model is about 17.5k segments, roughly 40 min on an A100; `slurm/submit_sweep.sh` (PR #3) loads each model once. Use A100s: H100 nodes ran this workload at about half speed, because it is bound by the host CPU.

### What changed in the code (all on `exp/final-grid`)

- **SlimGPT fix** (`prune/methods/slimgpt.py`).
  - The old compensation, SparseGPT's sweep in column order, only adjusted columns to the right of a removed one. So the last heads of `o_proj` were not compensated at all: on a synthetic layer the error was identical to plain zeroing.
  - The old scores used the trailing-submatrix inverse, so later units looked more important.
  - Removal now applies the exact joint least-squares update in float64, with greedy re-scoring (heads one per step, channels in batches of 128). Attention is pruned before the FFN's Hessian is taken.
  - Tests pin it to the closed-form optimum and check that selection doesn't change when columns are permuted.
- **log-increase schedule:** it now starts at `r_0 = p/4`, which is what the paper's Figure 4 shows, instead of 0.
- **FLAP AL-AM** (`allocation: al-am`), replicated from the official code. The CLI checks it against the parameter sparsity rather than the unit count.
- **Calibration builder:**
  - reference calibration now ends with eos, like generated calibration;
  - generated calibration keeps every record when a prompt repeats (dir-ru-en has one source twice, with two references);
  - `exclude_test_sources` builds `repair-multi-clean` without the two WMT22 collisions.
- **PRs #3 and #4** are merged into this branch.
- **New tooling:**
  - `scripts/grid.py` and `slurm/grid_pipeline.sbatch`: twin A100/H100 jobs, resumable stages, status;
  - `scripts/grid_report.py`: tables, bootstrap contrasts, transfer matrices, plots;
  - `scripts/repair_summary.py`, `scripts/calibration_diagnostics.py`, `scripts/grid_seed2.py`;
  - `slurm/single_pipeline.sbatch`, `slurm/setup_dense_bf16.sbatch`.

### Caveats

- **Pilot suite only:** first 300 segments per direction, ordered by document. One calibration seed for everything except the SlimGPT ref replicate.
- **Mixed hardware:** models ran on mixed A100 and H100 GPUs. Each status file records which (`grid-state/status/<name>.json`).
- **Older numbers aren't comparable:** the 29 September pilot, the layer-protection and the direction-scope numbers came from the old SlimGPT code. Calibration size or allocation also differed.
- **Recovery figures:** the repaired models trained on ALMA's own parallel data, where held-out loss is a sanity check, not a measure of generalisation. Recovery on the test set is what the tables report.
- **Not measured:** throughput. Earlier bench runs found no speedup from pruning under eager Hugging Face generation.

### Where things are

- `results/grid-2026-10-06/`:
  - `summary.md`: every table, CIs and p-values;
  - `long.csv`: one row per system and direction;
  - `results.json`;
  - `plots/`, `transfer/`, `repair.md`, `calibration_diagnostics.md`;
  - `seed2/`.
- `runs/<model>__alma10-greedy-300__<hash>/`: hypotheses and scores.
- `grid-state/`: job status (gitignored).
- `slurm-logs/grid/`: job logs.

## Design as agreed

| factor | levels |
| --- | --- |
| method | SlimGPT (fixed; log-increase, r_0 = p/4), FLAP (AL-AM as in the official code) |
| calibration text | prompt + reference (`ref`), prompt + dense ALMA-7B greedy translation (`gen`), both ending in eos |
| sparsity | 20, 30, 40 percent |
| scope | multi (1 model), pair (5), direction (10) |

192 pruned models. Calibration is 1,280 segments for every model: 128 per
direction for multi, 640 for pair, 1,280 for direction (same seed, same
sources in both calibration arms). Every model is evaluated on all ten
directions of `alma10-greedy-300` (first 300 segments, greedy), BLEU, chrF++,
COMET-22. LoRA repair (ALMA's recipe) on a few multi models chosen from the
results, on `repair-multi-clean`.

Out of scope: beam search, the full test sets, throughput benchmarks,
prompt-only calibration, LLM-Pruner.

## Decisions taken during the night

- Snellius refuses multi-partition requests for shared jobs. Each grid job is
  submitted twice, to `gpu_a100` and `gpu_h100`; a lock lets the first to start
  run and cancel its twin.
- The dense model is read from a bfloat16 safetensors copy
  (`/scratch-shared/scur0560/models/alma-7b-bf16`), verified bit-identical to
  `haoranxu/ALMA-7B` loaded in bfloat16 (291 tensors). Load time 160 s against
  about 500 s for the Hub's float32 shards.
- `dir-ru-en-1280` draws one source twice with two different references. The
  generated-calibration builder refused it; it now translates each distinct
  prompt once and keeps every record, so both arms hold the same 1,280 records.
- Repair data: `repair-multi-clean` drops the two exact source matches with
  WMT22-Test (en-cs "Amazing.", ru-en "Ну и что?"); 117,402 segments.

## Log

- 19:00-19:20 Survey of main, PRs #3 and #4 (merged into this branch), David's
  layer-protection and direction-scope branches, the 29 September overnight
  log. Literature check of SlimGPT and FLAP.
- 19:28 Dense bf16 copy, job 27680978 (4.5 min).
- 19:31 Calibration sets built on the login node, no test collisions:
  pair-{cs,de,is,ru,zh}-640, dir-{10 directions}-1280.
- 19:33 Sixteen generated-calibration jobs 27681075-27681090, about 4 min
  each; dir-ru-en-1280 failed on the duplicate prompt, fixed, resubmitted as
  27681134.
- 19:36 repair-multi-clean built.
- 19:39 Canary 27681152: the pre-fix SlimGPT from a worktree of main
  (`/scratch-shared/scur0560/oldcode-main`), generated and scored with this
  checkout.
- 19:40-20:00 Implementation by four subagents, reviewed and committed:
  exact SlimGPT compensation and greedy selection (83612a2), FLAP al-am
  (f353226), the grid pipeline and driver (ef773fa), the aggregator (2a1a0b5).
- 19:55 Canaries on the final code, 20 percent, multi, `alma10-greedy-300`:

  | system | BLEU | chrF++ | COMET |
  | --- | --- | --- | --- |
  | dense ALMA-7B | 30.35 | 51.14 | 0.8474 |
  | SlimGPT log-increase ref, pre-fix code (27681152) | 27.31 | 48.74 | 0.8354 |
  | SlimGPT log-increase ref, fixed | 27.72 | 49.03 | 0.8374 |
  | SlimGPT log-increase gen, fixed | 27.52 | 48.93 | 0.8360 |
  | FLAP al-am ref | 26.81 | 47.89 | 0.8221 |
  | FLAP al-am gen | 26.70 | 47.78 | 0.8209 |
  | FLAP global (separate head/channel budgets) ref | 27.53 | 48.51 | 0.8321 |
  | old pilot: FLAP global, prompt only | 19.02 | 38.22 | 0.8031 |

  No looping or truncation in any. FLAP al-am removes 20.0 percent of the
  parameters as 2.8 percent of heads and 28.5 percent of FFN channels. al-am is
  0.010 COMET below the separate global budget at 20 percent, so before FLAP's
  pair and direction scopes are queued, the global budget is run on the six
  multi settings as an allocation ablation (27681394-27681398 and the 20
  percent one above) next to al-am's six.
- 20:08 All 96 SlimGPT grid models submitted (twins on gpu_a100 and
  gpu_h100), multi first; the four 20 percent canaries adopted into the grid.
- 20:28 FLAP allocation ablation, multi scope, COMET (BLEU):

  | allocation | calib | 20% | 30% | 40% |
  | --- | --- | --- | --- | --- |
  | al-am | ref | 0.8221 (26.81) | 0.7823 (23.66) | 0.7393 (20.50) |
  | al-am | gen | 0.8209 (26.70) | 0.7830 (23.97) | 0.7392 (20.12) |
  | global, separate budgets | ref | 0.8321 (27.53) | 0.4281 (1.97) | 0.5683 (5.33) |
  | global, separate budgets | gen | 0.8312 (27.81) | 0.4250 (1.94) | 0.5897 (5.60) |

  The separate-budget global allocation is better at 20 percent but collapses
  at 30 and 40 (55 and 26 percent of outputs loop to the token budget, length
  ratios 11 and 5). al-am degrades smoothly. **Decision: FLAP stays on al-am
  for every scope**, as planned and as in the paper; the global runs stay as an
  ablation (`alma-7b-flap*-multi-global`, configs in `configs/prune/canary/`).
- 20:30 Repairs queued (ALMA's LoRA recipe on repair-multi-clean, 914 steps):
  slimgpt40-ref, slimgpt40-gen, slimgpt30-gen, slimgpt20-ref multi; 20:35
  flap40-ref multi (FLAP's best at 40 percent; ref ties gen on COMET and leads
  on BLEU). The SlimGPT 30 percent pick is gen (0.8229 against 0.8219).
- 20:37 The 90 FLAP pair and direction models submitted.
- 20:45 H100 jobs run this workload at about half the A100 speed: repair 320
  to 440 target tokens/s against 711, generation median 712 s against 481 s.
  The training process sits at 100 percent of one CPU core on the shared AMD
  EPYC 9334 H100 nodes, so it is host-bound, not GPU-bound. The three repairs
  that had landed on H100 were cancelled after 15 minutes and resubmitted to
  gpu_a100 only (27682686-27682688); grid jobs keep both partitions, since a
  slower GPU beats a queue.
- 21:16 All 192 grid models done, none failed (FLAP pair and direction
  models 20:37-21:16). Repairs running on A100, about 1 h 45 to 2 h each.
- 21:25 Extra, not in the agreed plan: a second calibration draw (seed 5678)
  for the SlimGPT prompt+reference slice, 48 models (`alma-7b-s2-*`,
  `scripts/grid_seed2.py`, manifest `configs/grid/manifest-seed2.json`). The
  bootstrap intervals cover test segments, not which calibration segments were
  drawn, and the 30 percent specialist gains (+0.004 to +0.006 COMET) are small
  enough that calibration noise could explain them. Cheap with the grid done
  early; submitted through `slurm/single_pipeline.sbatch` on gpu_a100.
- 21:36 Seed-5678 replicate done (48/48, 15 min wall clock). Headline COMET,
  seed 1234 / seed 5678:

  | SlimGPT ref | multi | pair | dir |
  | --- | --- | --- | --- |
  | 20% | 0.8374 / 0.8360 | 0.8369 / 0.8380 | 0.8371 / 0.8363 |
  | 30% | 0.8219 / 0.8223 | 0.8271 / 0.8268 | 0.8282 / 0.8282 |
  | 40% | 0.7914 / 0.7954 | 0.8083 / 0.8067 | 0.8076 / 0.8094 |

  Calibration-draw noise on a composite is 0.0000 to 0.0040 COMET; the
  specialist gains at 30 and 40 percent are significant under both draws.
  Seeds share 4 to 8 percent of their sources except Icelandic, where
  1,280 of 2,009 pairs are drawn and the two draws share 63 percent.
  Report: `results/grid-2026-10-06/seed2/`.
- 21:40 Extra, in the spirit of "LoRA on the most promising": the
  specialists beat multi at 40 percent, so the five SlimGPT gen 40 percent
  pair models are repaired too, each on its own pair's slice of
  repair-multi-clean (`grid.py repairs` now restricts a specialist's repair
  to its pruned directions). Same recipe, so one epoch over far less data:
  for Icelandic a few dozen steps. Jobs 27683649-27683653.
- 22:00-22:25 The five pair repairs done (28 to 237 steps each, 3 to 25
  min of training).
- 22:54 The five multi repairs done (914 steps, 101 to 120 min each on
  A100, peak 11.5 to 14.2 GB). Held-out loss fell for every repair. Final
  reports regenerated (`grid_report.py`, `repair_summary.py`).
