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
