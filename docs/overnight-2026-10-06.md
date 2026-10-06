# Final pilot grid, night of 6 October 2026

Unattended run on scur0560 (Snellius), branch `exp/final-grid`. Times are CEST.
The morning write-up comes first; the chronological log with job ids is below.

## Morning write-up

(Filled in at the end of the run.)

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
