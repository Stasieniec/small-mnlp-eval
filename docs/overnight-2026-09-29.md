# Overnight run, 2026-09-29

The unattended GPU run on scur0560 (Snellius), branch
`claude/busy-bell-pyhq1v`. Times are CEST. The morning write-up comes first;
the chronological log with every job id is below it.

## Morning write-up

### Status

- **Ran:** Phase 0 setup, the smoke test (passed, COMET included), the dense
  baseline, all fourteen prune jobs and their evaluations on
  `alma10-greedy-300`, the subnetwork overlap analysis, and the report. No
  GPU job failed.
- **Did not run: repair.** The repair set `repair-multi` failed the
  test-set contamination check (2 of 117,404 segments), which the brief
  makes a hard stop. None of the three repairs or their evaluations was
  queued. This is the main thing waiting on you; see "What needs a human
  decision".
- **Bench: still queued at the time of writing.** The four jobs (27371094
  alma-7b, 27371086 slimgpt20-multi, 27371087 slimgpt20-multi-uniform,
  27371089 slimgpt50-multi) each need a whole exclusive A100 node and have
  waited since 19:58 with reason `Resources`. The final report 27371166
  runs after them. Until then `reports/` holds the interim report, complete
  except for efficiency.
- Compute used: about 6.3 GPU-hours and 800 SBU before bench, against the
  plan's 40-45 A100-hours. The difference is repair (not run) and SlimGPT
  pruning taking 12-15 min per model instead of the hour the docs estimate.
- `git push` fails on this account (no GitHub credentials: no SSH key, no
  token, no `gh`). Everything is committed locally on
  `claude/busy-bell-pyhq1v`; push it from a machine that has access.

Three results stand out:

1. **Calibrating on prompt plus reference (`slimgpt20-multi-target`)
   recovers most of what 20% pruning costs**: BLEU 28.33 against 20.65 for
   prompt-only and 30.35 dense; COMET 0.8367 against 0.8148 and 0.8474. The
   token-count control (`-256`) gains only +0.57 BLEU, so the gain comes
   from the target side, not from more tokens.
2. **Pair-specific SlimGPT subnetworks collapse outside their pair and lose
   even on it.** They score 10.4-16.2 BLEU against 20.65 for the multi
   subnetwork. The 256 against 1,280 calibration segments confound this; see
   Transfer matrix.
3. **Uniform and log-increase budgets both beat the global one at 20%**
   (+2.4 and +1.3 BLEU, COMET +0.007 each, p=0.001), and tie with each other
   on COMET.

The dense baseline, slimgpt20-multi and both FLAP systems reproduce Jan's
pilot numbers from his account to the second decimal, so the pipeline is
deterministic across accounts.

### Results

Suite `alma10-greedy-300`: the first 300 segments of each WMT22 direction,
greedy, 3,000 segments per system. Macro means over the ten directions.
COMET is `Unbabel/wmt22-comet-da`. Length ratio is hypothesis over reference
tokens; truncated is the share of segments that hit the 256-token budget.
All systems are pruned ALMA-7B, no repair, SlimGPT with a global budget
unless the name says otherwise.

| system | removed | calibration | BLEU | chrF++ | COMET | length ratio | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| alma-7b (dense) | 0 | | 30.35 | 51.14 | 0.8474 | 0.964 | 0.0% |
| slimgpt20-multi | 20% | 1,280 prompts | 20.65 | 41.71 | 0.8148 | 0.909 | 0.1% |
| slimgpt20-multi-uniform | 20% | 1,280 prompts | 23.01 | 43.81 | 0.8222 | 0.890 | 0.0% |
| slimgpt20-multi-log | 20% | 1,280 prompts | 21.92 | 42.92 | 0.8223 | 0.902 | 0.1% |
| slimgpt20-multi-target | 20% | 1,280 prompt+ref | **28.33** | **49.33** | **0.8367** | 0.949 | 0.0% |
| slimgpt20-multi-256 | 20% | 2,560 prompts | 21.22 | 42.28 | 0.8195 | 0.904 | 0.2% |
| slimgpt20-cs | 20% | 256 prompts, cs | 12.28 | 31.73 | 0.6845 | 0.962 | 2.1% |
| slimgpt20-de | 20% | 256 prompts, de | 10.37 | 29.08 | 0.6482 | 0.999 | 1.8% |
| slimgpt20-is | 20% | 256 prompts, is | 13.05 | 33.36 | 0.7105 | 0.968 | 1.3% |
| slimgpt20-ru | 20% | 256 prompts, ru | 10.86 | 30.18 | 0.6523 | 0.949 | 0.6% |
| slimgpt20-zh | 20% | 256 prompts, zh | 16.15 | 36.25 | 0.7340 | 0.955 | 1.6% |
| flap20-multi | 20% | 1,280 prompts | 19.02 | 38.22 | 0.8031 | 0.815 | 0.3% |
| flap20-de | 20% | 256 prompts, de | 19.62 | 39.30 | 0.8043 | 0.864 | 0.4% |
| slimgpt50-multi | 50% | 1,280 prompts | 4.91 | 19.08 | 0.5567 | 0.645 | 0.8% |
| slimgpt50-multi-log | 50% | 1,280 prompts | 3.82 | 17.26 | 0.5645 | 0.612 | 0.0% |

Significance against dense (the report's paired bootstrap, 1,000 resamples,
per direction and metric; 14 systems x 10 directions x 3 metrics = 420
tests): every system is significantly worse than dense in 415 of them, and
no system is better anywhere. The five non-significant cells all belong to
slimgpt20-multi-target: BLEU en-cs -1.35 (p=0.052) and ru-en -0.70
(p=0.153), chrF++ en-zh -0.06 (p=0.382) and ru-en -0.15 (p=0.284), COMET
is-en -0.0055 (p=0.060). Over all 3,000 segments slimgpt20-multi-target is
still below dense on COMET, -0.0107 [95% CI -0.0129, -0.0087], p=0.001.

SlimGPT against FLAP at 20% with multi calibration: +1.63 BLEU, COMET
+0.0116 [+0.0088, +0.0147], p=0.001. FLAP-de against FLAP-multi: COMET
+0.0012, p=0.441, the tie Jan saw. At 50% SlimGPT does not run away as FLAP
did in the pilot (length ratio 3.6-4.2 there); it writes short, fluent, on-topic
English that drops or invents content (length ratio 0.61-0.65, no loops).
Unusable either way without repair.

BLEU per direction, 20% systems (and 50% for reference):

| system | cs-en | de-en | is-en | ru-en | zh-en | en-cs | en-de | en-is | en-ru | en-zh | into en | out of en |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 42.0 | 31.3 | 38.2 | 36.9 | 19.8 | 24.9 | 28.0 | 23.8 | 25.0 | 33.5 | 33.7 | 27.0 |
| slimgpt20-multi | 28.5 | 21.3 | 26.9 | 24.8 | 12.0 | 15.5 | 19.4 | 15.9 | 16.4 | 25.8 | 22.7 | 18.6 |
| slimgpt20-multi-uniform | 31.2 | 24.5 | 28.4 | 27.4 | 14.0 | 18.8 | 21.3 | 17.5 | 20.1 | 26.8 | 25.1 | 20.9 |
| slimgpt20-multi-log | 29.9 | 22.7 | 26.7 | 27.2 | 14.0 | 17.0 | 20.8 | 16.5 | 18.8 | 25.6 | 24.1 | 19.7 |
| slimgpt20-multi-target | 40.0 | 28.4 | 36.5 | 36.2 | 17.8 | 23.6 | 24.7 | 21.9 | 22.3 | 31.8 | 31.8 | 24.9 |
| slimgpt20-multi-256 | 30.0 | 22.1 | 27.7 | 25.2 | 13.8 | 14.7 | 19.3 | 16.3 | 16.8 | 26.4 | 23.7 | 18.7 |
| slimgpt20-cs | 20.7 | 16.0 | 7.9 | 16.3 | 4.4 | 11.9 | 13.9 | 7.1 | 11.7 | 12.9 | 13.1 | 11.5 |
| slimgpt20-de | 13.2 | 16.9 | 6.0 | 16.2 | 4.8 | 8.4 | 14.9 | 1.3 | 11.6 | 10.5 | 11.4 | 9.3 |
| slimgpt20-is | 12.0 | 15.4 | 18.2 | 18.5 | 5.9 | 8.3 | 13.9 | 13.3 | 12.2 | 12.8 | 14.0 | 12.1 |
| slimgpt20-ru | 12.9 | 14.3 | 6.3 | 16.5 | 6.2 | 8.0 | 13.4 | 6.0 | 12.4 | 12.7 | 11.2 | 10.5 |
| slimgpt20-zh | 25.1 | 20.8 | 18.5 | 23.3 | 12.3 | 7.4 | 13.8 | 7.2 | 12.5 | 20.6 | 20.0 | 12.3 |
| flap20-multi | 25.8 | 20.8 | 26.0 | 25.1 | 9.4 | 12.7 | 18.8 | 14.5 | 16.6 | 20.6 | 21.4 | 16.6 |
| flap20-de | 28.3 | 20.9 | 26.7 | 25.4 | 10.0 | 14.0 | 19.5 | 15.0 | 16.2 | 20.2 | 22.3 | 17.0 |
| slimgpt50-multi | 6.3 | 6.7 | 4.6 | 8.6 | 3.1 | 2.8 | 5.1 | 1.6 | 4.9 | 5.4 | 5.9 | 3.9 |
| slimgpt50-multi-log | 5.7 | 6.0 | 5.3 | 5.9 | 3.3 | 1.8 | 2.3 | 1.0 | 3.5 | 3.4 | 5.2 | 2.4 |

COMET per direction:

| system | cs-en | de-en | is-en | ru-en | zh-en | en-cs | en-de | en-is | en-ru | en-zh | into en | out of en |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8394 | 0.8554 |
| slimgpt20-multi | 0.8126 | 0.8086 | 0.8326 | 0.8038 | 0.7485 | 0.8394 | 0.8281 | 0.8151 | 0.8375 | 0.8217 | 0.8012 | 0.8283 |
| slimgpt20-multi-uniform | 0.8287 | 0.8191 | 0.8411 | 0.8142 | 0.7671 | 0.8385 | 0.8304 | 0.8217 | 0.8368 | 0.8242 | 0.8140 | 0.8303 |
| slimgpt20-multi-log | 0.8262 | 0.8208 | 0.8351 | 0.8160 | 0.7672 | 0.8440 | 0.8310 | 0.8178 | 0.8431 | 0.8220 | 0.8131 | 0.8316 |
| slimgpt20-multi-target | 0.8448 | 0.8330 | 0.8572 | 0.8389 | 0.7759 | 0.8564 | 0.8440 | 0.8329 | 0.8451 | 0.8389 | 0.8299 | 0.8435 |
| slimgpt20-multi-256 | 0.8236 | 0.8118 | 0.8391 | 0.8114 | 0.7560 | 0.8369 | 0.8292 | 0.8252 | 0.8379 | 0.8241 | 0.8084 | 0.8307 |
| slimgpt20-cs | 0.7487 | 0.7240 | 0.5955 | 0.7093 | 0.6013 | 0.7668 | 0.7380 | 0.5589 | 0.7496 | 0.6530 | 0.6757 | 0.6933 |
| slimgpt20-de | 0.6516 | 0.7494 | 0.5483 | 0.6859 | 0.5832 | 0.6329 | 0.7607 | 0.5514 | 0.7103 | 0.6082 | 0.6437 | 0.6527 |
| slimgpt20-is | 0.6755 | 0.7398 | 0.7559 | 0.7234 | 0.6235 | 0.6751 | 0.7559 | 0.7661 | 0.7389 | 0.6509 | 0.7036 | 0.7174 |
| slimgpt20-ru | 0.6660 | 0.6991 | 0.5673 | 0.7168 | 0.5930 | 0.6664 | 0.7140 | 0.5015 | 0.7554 | 0.6433 | 0.6484 | 0.6561 |
| slimgpt20-zh | 0.7734 | 0.7902 | 0.7524 | 0.7774 | 0.7285 | 0.6749 | 0.7503 | 0.5705 | 0.7546 | 0.7675 | 0.7644 | 0.7036 |
| flap20-multi | 0.8075 | 0.8054 | 0.8235 | 0.7983 | 0.7372 | 0.8136 | 0.8149 | 0.7975 | 0.8310 | 0.8024 | 0.7944 | 0.8119 |
| flap20-de | 0.8126 | 0.8073 | 0.8218 | 0.7992 | 0.7441 | 0.8245 | 0.8175 | 0.7958 | 0.8325 | 0.7876 | 0.7970 | 0.8116 |
| slimgpt50-multi | 0.5930 | 0.6180 | 0.5626 | 0.5883 | 0.5307 | 0.5455 | 0.5845 | 0.4684 | 0.5735 | 0.5027 | 0.5785 | 0.5349 |
| slimgpt50-multi-log | 0.6120 | 0.6224 | 0.6352 | 0.6007 | 0.5743 | 0.5229 | 0.5541 | 0.4601 | 0.5789 | 0.4842 | 0.6089 | 0.5200 |

Full tables, chrF++ per direction and behavioural rates are in
`reports/report.md` (copied to `$HOME/mnlp-reports/`).

### Repair

**Not run.** `mnlp-eval calibration --spec
configs/calibration/repair-multi.yaml` found two exact source-string
collisions with `haoranxu/WMT22-Test`:

| direction | training record | source | training target | test segment | test reference |
| --- | --- | --- | --- | --- | --- |
| en-cs | 5463 | `Amazing.` | `Nádherný.` | 1356 | `Úžasné.` |
| ru-en | 614 | `Ну и что?` | `So what?` | 311 | `So what?` |

The brief makes a collision a hard failure, so the repair line stopped
there: no held-out losses, training times or recovery ratios exist for
ALMA-7B. The only repair that ran is the smoke test on Qwen2.5-0.5B with
the clean pair-de set: 30 steps, held-out loss 2.671 before and 1.936 after,
peak 2.75 GB, and the repaired model stopped on its own on all 32 segments
where the unrepaired one hit the token budget on all 32. That shows the
repair code path works on a GPU end to end, from training through merging and
reloading to evaluation. It says nothing about ALMA-7B's speed or memory.

### Transfer matrix

Each pair-specific system (SlimGPT, 20%, 128 segments per direction of its
own pair, 256 in all) against slimgpt20-multi (128 per direction of all
ten, 1,280 in all):

| system | own pair COMET | multi on that pair | delta | other eight COMET | multi on those | delta | own-pair BLEU delta | other BLEU delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slimgpt20-cs | 0.7578 | 0.8260 | -0.0682 | 0.6662 | 0.8120 | -0.1458 | -5.8 | -9.0 |
| slimgpt20-de | 0.7550 | 0.8183 | -0.0633 | 0.6215 | 0.8139 | -0.1924 | -4.4 | -11.7 |
| slimgpt20-is | 0.7610 | 0.8239 | -0.0629 | 0.6979 | 0.8125 | -0.1146 | -5.7 | -8.1 |
| slimgpt20-ru | 0.7361 | 0.8206 | -0.0846 | 0.6313 | 0.8133 | -0.1820 | -6.1 | -10.7 |
| slimgpt20-zh | 0.7480 | 0.7851 | -0.0371 | 0.7305 | 0.8222 | -0.0917 | -2.4 | -5.0 |

Every one is significantly below the multi subnetwork overall (COMET -0.08
to -0.17, p=0.001 each) and on its own pair (-0.037 to -0.085, p=0.001
each). Each is specialised: it loses two to three times as much on the
other eight directions as on its own. The worst
cells are Icelandic for the systems that never saw it (en-is BLEU 1.3 for
slimgpt20-de; is-en 6.0-7.9 for cs, de and ru). slimgpt20-de on is-en
hallucinates and loops (8 of 300 segments hit the budget), and 8.8% of its
output overall is in the source language. slimgpt20-zh is the mildest, and
the only pair-specific system to match multi on any direction (zh-en +0.4
BLEU).

The comparison confounds two things, languages and calibration size.
Three observations point at size as the larger part. The pair systems lose
on their own pair, where they had the same 128 segments per direction as
multi. Their FFN selections resemble each other more than they resemble
multi's, although multi's calibration contains every pair's data: FFN
Jaccard 0.89-0.93 among cs, de, is and ru, against 0.84-0.85 between each
of them and multi. And FLAP at 256 segments
(flap20-de) ties FLAP at 1,280, so the size sensitivity is SlimGPT's.
SlimGPT fits a weight update to the calibration Hessian, and 256 segments,
about 16k tokens, barely exceed the 11,008 dimensions of a `down_proj`
Hessian. Adding data at the top end helps much less: slimgpt20-multi-256
(2,560 segments) gains only +0.0047 COMET over 1,280.

Overlap (`reports/overlap/overlap.md`), Jaccard of kept units with the
excess over chance in brackets:

| attention heads | multi | cs | de | is | ru | zh |
| --- | --- | --- | --- | --- | --- | --- |
| multi | - | 0.916 (+0.229) | 0.905 (+0.221) | 0.929 (+0.245) | 0.914 (+0.230) | 0.902 (+0.223) |
| cs | | - | 0.941 (+0.251) | 0.925 (+0.234) | 0.955 (+0.260) | 0.881 (+0.200) |
| de | | | - | 0.909 (+0.223) | 0.932 (+0.245) | 0.876 (+0.196) |
| is | | | | - | 0.911 (+0.224) | 0.874 (+0.196) |
| ru | | | | | - | 0.896 (+0.214) |

| FFN channels | multi | cs | de | is | ru | zh |
| --- | --- | --- | --- | --- | --- | --- |
| multi | - | 0.850 (+0.147) | 0.842 (+0.139) | 0.846 (+0.146) | 0.839 (+0.134) | 0.854 (+0.154) |
| cs | | - | 0.925 (+0.216) | 0.909 (+0.205) | 0.933 (+0.220) | 0.866 (+0.171) |
| de | | | - | 0.901 (+0.199) | 0.933 (+0.220) | 0.861 (+0.168) |
| is | | | | - | 0.893 (+0.189) | 0.879 (+0.181) |
| ru | | | | | - | 0.853 (+0.161) |

zh is the outlier in both components. Across criteria, SlimGPT and FLAP at
20% multi agree above chance but not closely (heads 0.845, +0.168; FFN
0.811, +0.115), which is the range `slurm/README.md` says to expect. The
analysis reported no problems.

### Budget comparison

| comparison | BLEU | COMET delta, all ten | into English | out of English |
| --- | --- | --- | --- | --- |
| 20%: uniform against global | +2.36 | +0.0074 [+0.0048, +0.0098] p=0.001 | +0.0128 p=0.001 | +0.0020 p=0.365 |
| 20%: log-increase against global | +1.27 | +0.0076 [+0.0048, +0.0103] p=0.001 | +0.0119 p=0.001 | +0.0032 p=0.124 |
| 20%: uniform against log-increase | +1.09 | -0.0002 [-0.0025, +0.0021] p=0.879 | +0.0009 p=0.516 | -0.0012 p=0.508 |
| 50%: log-increase against global | -1.09 | +0.0077 [+0.0023, +0.0127] p=0.003 | +0.0304 p=0.001 | -0.0149 p=0.001 |

COMET intervals are paired bootstraps on the stored segment scores (1,000
resamples, seed 12345, the repository's `bootstrap_segment_delta`). At 20%
both per-layer schedules beat the global budget, and the gain is into
English; out of English none of the three differs significantly. Uniform
and log-increase tie on COMET, and uniform is ahead by 1.1 BLEU. At 50% the
comparison is between two unusable systems, and its two halves disagree
(log-increase +0.030 into English, -0.015 out of it), so it does not rank
the schedules. Uniform keeps 26 of 32 heads per layer (0.1875
removed), because whole heads cannot hit 20% per layer.

### Calibration text

| system | calibration | BLEU into / out of en | COMET into / out of en |
| --- | --- | --- | --- |
| dense | | 33.7 / 27.0 | 0.8394 / 0.8554 |
| slimgpt20-multi | 1,280 prompts, 78k tokens | 22.7 / 18.6 | 0.8012 / 0.8283 |
| slimgpt20-multi-256 | 2,560 prompts, 156k tokens | 23.7 / 18.7 | 0.8084 / 0.8307 |
| slimgpt20-multi-target | 1,280 prompt+reference, 134k tokens | 31.8 / 24.9 | 0.8299 / 0.8435 |

Token counts are the ones `docs/pruning.md` gives. Per direction, against
slimgpt20-multi:

| system | cs-en | de-en | is-en | ru-en | zh-en | en-cs | en-de | en-is | en-ru | en-zh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -target, BLEU | +11.5 | +7.1 | +9.6 | +11.4 | +5.9 | +8.0 | +5.4 | +6.0 | +5.8 | +6.0 |
| -256, BLEU | +1.4 | +0.8 | +0.8 | +0.4 | +1.9 | -0.9 | -0.1 | +0.4 | +0.4 | +0.7 |
| -target, COMET | +0.0322 | +0.0244 | +0.0245 | +0.0350 | +0.0274 | +0.0171 | +0.0159 | +0.0178 | +0.0076 | +0.0172 |
| -256, COMET | +0.0110 | +0.0032 | +0.0065 | +0.0075 | +0.0075 | -0.0025 | +0.0012 | +0.0101 | +0.0005 | +0.0024 |

| comparison | COMET delta, all ten | into English | out of English |
| --- | --- | --- | --- |
| -target against multi | +0.0219 [+0.0195, +0.0245] p=0.001 | +0.0287 p=0.001 | +0.0151 p=0.001 |
| -256 against multi | +0.0047 [+0.0027, +0.0070] p=0.001 | +0.0072 p=0.001 | +0.0023 p=0.188 |
| -target against -256 | +0.0172 [+0.0146, +0.0195] p=0.001 | +0.0216 p=0.001 | +0.0128 p=0.001 |

The two questions:

- **Does calibrating on the reference close the out-of-English gap?** It
  improves every direction, but it closes more of the into-English loss
  than of the out-of-English one. Of the pruning loss, -target recovers
  83% of BLEU and 75% of COMET into English, against 75% of BLEU and 56%
  of COMET out of English. What remains afterwards is larger out of English:
  -2.1 BLEU and -0.0119 COMET, against -1.9 and -0.0095 into it. The
  premise also looks weaker on this suite than in the pilot: with prompt
  calibration the macro loss out of English (-8.4 BLEU, -0.027 COMET) is
  not larger than into English (-11.0, -0.038). Per language it is larger
  out of English only in relative BLEU, for cs (-38% against -32%), is
  (-33% against -30%) and by a point for ru; in COMET every language loses
  less out of English than into it.
- **Does -target beat -256?** Yes, by +7.1 BLEU and +0.0172 COMET
  (p=0.001), in both halves. -256 sees more tokens than -target and gains a
  fifth as much COMET, so the gain comes from the target side, not from the
  token count.

A mechanism consistent with this, not tested: generation spends almost all
its tokens writing the translation, and prompt-only calibration never runs
the model in that regime, in either direction. -target also selects
different heads, not just a different compensation: its head overlap with
slimgpt20-multi is 0.853, against 0.988 for -256.

### Efficiency

Not measured yet: the bench jobs are still waiting for exclusive nodes (see
Status). Sizes are known from the checkpoints. Dense ALMA-7B has 6.738B
parameters, 13.5 GB in bfloat16 (the Hub copy is float32, 27 GB). Every 20%
system has 5.443B (0.81 of dense, 11 GB); slimgpt50-multi has 3.500B and
slimgpt50-multi-log 3.503B (0.52, 6.6 GB). The fraction removed is below
the unit sparsity only because the embeddings and output head (0.262B) are
not pruned: 0.8 x 6.476B + 0.262B = 5.443B exactly.

### Failures and fixes

No GPU job failed. What went wrong, and what was done:

| what | cause | action |
| --- | --- | --- |
| `repair-multi` failed its contamination check | 2 exact-source collisions with WMT22-Test (table above) | Hard stop per the brief. Repair not queued. Set renamed to `data/calibration/repair-multi.CONTAMINATED`. |
| A failed calibration build leaves a usable-looking set on disk | `mnlp-eval calibration` writes the set, then exits 1; `submit_repair.sh` checks only that the directory exists | Commit `4fc7193`: `load_calibration_prompts` and `load_repair_data` refuse a set whose `calibration.json` records collisions. Two new tests; checked against the real sets. |
| Bench for alma-7b rejected, "Job dependency problem" | `afterok` on a generation job that had already left the controller (`MinJobAge` 300 s) | Submitted without a dependency, as its generation was complete. |
| `scontrol update` of the report's dependency denied | Not permitted for users on Snellius | Cancelled report 27370301, resubmitted as 27371166 on the jobs still live. |
| `git push` fails | No GitHub credentials on this account | Committed locally throughout. |
| `mypy` reports 2 errors in `src/mnlp_eval/models/hf_causal.py` (lines 90, 95) | Type stubs of the peft and transformers versions installed today; the file is untouched on this branch | Not fixed: not one of the brief's four checks, and not worth editing the generation path mid-run. |

### Deviations from the plan

No experimental setting was changed and no variant system was run. What
differs from the brief:

- The three repairs and their evaluations did not run (contamination stop,
  above). Everything that depends on them (held-out losses, recovery
  ratios) is absent.
- Bench jobs were chained `afterok` on each system's generation job rather
  than submitted by hand once the runs existed. The same thing, sooner.
- An extra interim report (27371383) ran before bench, so the quality
  tables did not wait on whole-node availability. The final report
  27371166 replaces its files.
- The smoke Qwen descriptor was moved from `subnetworks/` to
  `runs-smoke/subnetworks/` before the overlap analysis, so it is not
  compared with the ALMA ones and is not committed.

### What needs a human decision

1. **The repair data.** The options as I see them: accept the set as is;
   rebuild it without the two segments (a change to the repair data); or
   exclude single-sentence utterances some other way. Facts that bear on
   it: both segments are one- to three-word stock phrases; both lie outside
   the 300 segments per direction that this suite evaluates (test segments
   1356 and 311); ALMA-7B was itself fine-tuned on this corpus, so the
   dense baseline has already trained on both; the check is exact string
   match on the source. Once decided, the chain is in `slurm/README.md`
   ("Chaining prune, repair and evaluation"). The pruned parents'
   checkpoints are on scratch, which purges files untouched for 14 days,
   so they last until about 13 October unless re-pruned.
2. **Which system repair should start from.** The repair configs point at
   slimgpt20-multi, but slimgpt20-multi-target is 7.7 BLEU better before
   any repair. Repairing both would show whether repair and target
   calibration add up. A config for the target parent does not exist yet.
3. **The transfer matrix design.** As designed, pair-specific means 256
   segments against 1,280, and with SlimGPT that size difference looks
   like the larger effect. Options: report it with the confound stated;
   add pair-specific runs at equal total size (640 per direction, which
   `configs/calibration/pair-*.yaml` currently argues against); or build
   the matrix with FLAP, which is not size-sensitive here.
4. **Whether the other comparisons should move to prompt+target
   calibration**, given its size, and whether its gain holds with the
   uniform or log-increase budget.
5. `docs/pruning.md` and `slurm/README.md` estimate an hour per SlimGPT
   prune; it takes 12-15 minutes. Not changed tonight.

## Log

### Phase 0: setup (login node int4)

- 19:08 Checked out `claude/busy-bell-pyhq1v` at `a2e4532`. The account had
  nothing set up: no `.venv`, no `.venv-comet`, no `HF_HOME`, no
  `/scratch-shared/scur0560`.
- 19:10 `HF_HOME=/scratch-shared/$USER/hf_home` created and exported in
  `~/.bashrc`. Batch scripts do not read `~/.bashrc`; they inherit the
  submitting shell through `--export ALL`, so every submission below was made
  from a shell with `HF_HOME` exported.
- 19:12 Both venvs built with `uv venv --managed-python --python 3.11`
  (CPython 3.11.16 under `~/.local/share/uv`, `Python.h` present). `.venv`:
  torch 2.14.0+cu130, transformers 4.55.4, peft 0.21.1. `.venv-comet`:
  unbabel-comet 2.2.7, torch 2.14.0+cu130.
- 19:17 `pytest tests/ -q`: all pass.
- 19:18 Budget (`accinfo`, `budget-overview`): shared course account
  `gpuuva086`, 406,439 SBU remaining, 388,359 SBU left for dispatch after
  other users' running jobs. Billing weights from `scontrol show partition`:
  `gpu_a100` bills `gres/gpu=128, cpu=7.11` per hour, max-of-TRES (a whole
  node is 512 either way), so one A100 with 18 cores is 128 SBU/h, as the
  brief assumed. `gpu_h100` is 192 per GPU and 12 per core. The plan's
  40-45 A100-hours is about 5,800 SBU, 1.5% of what is left, so nothing is
  dropped for budget. No per-user job or QoS limits
  (`sacctmgr show assoc`, `show qos`).
- 19:19-19:21 Calibration sets built, no collisions: multi-10dir (1,280
  segments, fingerprint 44f1fd1f413fbce1), multi-10dir-256 (2,560,
  2be4440baf50dce0), pair-cs (83ae5b5b5f939a31), pair-de (ce4605a66bf86226),
  pair-is (0cac80da404e13b2), pair-ru (c24f4e603a1db741), pair-zh
  (75876fd8936b8368), 256 segments each.
- 19:21 **repair-multi FAILED the contamination check: 2 of 117,404 segments
  collide with haoranxu/WMT22-Test.** Per the brief this is a hard failure:
  no repair runs tonight on this data, and nothing works around it. The two
  collisions (exact source-string matches):
  - en-cs, training record 5463, source `Amazing.` (train target `Nádherný.`),
    equal to WMT22 en-cs test segment 1356 (reference `Úžasné.`);
  - ru-en, training record 614, source `Ну и что?` (train target `So what?`),
    equal to WMT22 ru-en test segment 311 (reference `So what?`).
  Both lie outside the first 300 segments that `alma10-greedy-300` evaluates.
  `mnlp-eval calibration` exits 1 but still writes the set to disk under its
  usable name, and `submit_repair.sh` only checks that the directory exists,
  so the directory was renamed to `data/calibration/repair-multi.CONTAMINATED`
  so that nothing can train on it by accident.
- 19:22 Prefetch done: ALMA-7B, ALMA-7B-R, NLLB-600M, opus-mt-de-en,
  Qwen2.5-0.5B-Instruct, WMT22 and FLORES-200 for all ten directions,
  `Unbabel/wmt22-comet-da` and the `xlm-roberta-large` encoder files.
  COMETKiwi and XCOMET-XL failed as expected (gated, no `HF_TOKEN`); the
  default metric set does not use them.
- 19:23 COMET offline check on the login node passes:
  `HF_HUB_OFFLINE=1 .venv-comet/bin/python -c "... load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))"`
  prints `ok`.
- 19:24 Probe job 27370124 (1 A100, 50 s, COMPLETED): driver 595.91.07 on
  A100-SXM4-40GB, so the cu130 torch wheels work; `torch.cuda.is_available()`
  true, a bf16 matmul and a `torch.compile` (triton JIT, the step that needed
  `Python.h`) succeed in both venvs; `HF_HOME` reaches the job. Its
  `AllocTRES` shows `billing=128`, confirming the rate. The probe was worth
  it because `default_device()` falls back to CPU silently, so a driver
  mismatch would have had every prune job grind on CPU for four hours.

### Queued at 19:27

The smoke chain, the dense baseline, and all fourteen prune jobs with an
evaluation (generate, then score) behind each. The three real repair jobs
and their evaluations were NOT queued (repair-multi is contaminated, above).
The report is queued once the last score job id is known.

| system | prune | repair | generate | score |
| --- | --- | --- | --- | --- |
| smoke-qwen-slimgpt20 | 27370221 | | 27370224 | 27370225 |
| smoke-qwen-slimgpt20-lora | | 27370222 | 27370226 | 27370227 |
| alma-7b (dense) | | | 27370229 | 27370230 |
| alma-7b-slimgpt20-multi | 27370233 | | 27370234 | 27370235 |
| alma-7b-slimgpt50-multi | 27370236 | | 27370238 | 27370239 |
| alma-7b-slimgpt20-cs | 27370240 | | 27370242 | 27370243 |
| alma-7b-slimgpt20-de | 27370245 | | 27370246 | 27370247 |
| alma-7b-slimgpt20-is | 27370248 | | 27370249 | 27370250 |
| alma-7b-slimgpt20-ru | 27370251 | | 27370252 | 27370253 |
| alma-7b-slimgpt20-zh | 27370254 | | 27370256 | 27370257 |
| alma-7b-slimgpt20-multi-uniform | 27370262 | | 27370263 | 27370264 |
| alma-7b-slimgpt20-multi-log | 27370267 | | 27370268 | 27370269 |
| alma-7b-slimgpt50-multi-log | 27370270 | | 27370271 | 27370273 |
| alma-7b-slimgpt20-multi-target | 27370275 | | 27370276 | 27370278 |
| alma-7b-slimgpt20-multi-256 | 27370279 | | 27370280 | 27370281 |
| alma-7b-flap20-multi | 27370282 | | 27370283 | 27370284 |
| alma-7b-flap20-de | 27370285 | | 27370286 | 27370287 |

The final report was queued at 19:28 as job 27370301, `afterany` on the
fifteen real score jobs, so it runs even if one system fails.

### Phase 1: smoke test, passed (19:28-19:46)

- Prune 27370221 (1 min): achieved unit sparsity 0.1999 against 0.2
  requested; FFN 0.19999, heads 0.179 (Qwen2.5-0.5B has 14 query heads in 2
  key/value groups, so heads go in pairs).
- Repair 27370222 (50 s): 30 optimizer steps, loss lines every step, about
  450 target tokens/s (a 0.5B model on micro-batches of 4, so not a guide to
  ALMA-7B), held-out loss 2.671 before and 1.936 after, peak 2.75 GB.
  `repair.json` is beside the merged checkpoint.
- Generate 27370224 and 27370226: 32 de-en hypotheses each, English, on
  topic, no loops. The unrepaired model ran to the 128-token budget on every
  segment (budget hit rate 1.0); after repair it stopped on its own every
  time (0.0).
- Score 27370225 and 27370227: both wrote `scores.surface.json` and
  `scores.neural.json`. **COMET works on a GPU node.** Smoke numbers, for the
  record only: BLEU 11.5 to 14.6, COMET 0.676 to 0.696.
- Smoke run directories moved to `runs-smoke/`.

### Phase 2 (from 19:30)

- Dense baseline 27370229/27370230: BLEU 30.3511, chrF++ 51.139, identical
  to Jan's pilot on his account (30.35, 51.14), so generation reproduces
  across accounts. COMET 0.8474.
- SlimGPT prune jobs take 12-13 min each, not the hour the docs estimate.
  Every finished one hit its sparsity exactly (0.2000 or 0.5000, heads and
  channels separately).
- 19:55 Commit 4fc7193: `load_calibration_prompts` and `load_repair_data`
  now refuse a set whose own `calibration.json` records test collisions.
  Before it, the calibration CLI's non-zero exit was the only guard, and the
  set was already on disk under the name the repair configs read. Checked
  against the real sets: the seven clean ones load, the quarantined
  repair-multi is refused. Committed after every prune job had started, so
  no running job imported a changed module.
- 19:58 Bench jobs queued, each `afterok` on its system's generation job:
  27371094 (alma-7b), 27371086 (slimgpt20-multi), 27371087
  (slimgpt20-multi-uniform), 27371089 (slimgpt50-multi). The dense one went
  without a dependency: its generation job had already left the controller
  (`MinJobAge` is 300 s), and a dependency on a job the controller no longer
  knows is rejected with "Job dependency problem".
- 19:58 `scontrol update` of the report's dependency is not permitted for
  users here ("Access/permission denied"), so report 27370301 was cancelled
  and replaced by 27371166, `afterany` on the score and bench jobs still
  live at that moment (those already finished had their output on disk).
- 20:00 `mnlp-eval overlap --subnetwork-dir subnetworks/ --out
  reports/overlap` on the login node, after moving the smoke Qwen descriptor
  to `runs-smoke/subnetworks/` so it stays out of the ALMA comparison.
- 20:08 Every generation and score job finished, none failed: all fifteen
  systems have `scores.surface.json` and `scores.neural.json`. Generation
  took 10-15 min per system and scoring 2-3 min.
- 20:09 Interim report 27371383 queued with no dependency, because the
  exclusive-node bench jobs are waiting for whole free nodes (reason
  `Resources`) and the final report waits on them.
- 20:10 Sanity checks on the two surprising results. (1) The pair-specific
  systems lose 7-11 BLEU against slimgpt20-multi on every direction, their
  own pair included. Not a pipeline fault: slimgpt20-de on is-en
  hallucinates and loops (8 of 300 segments hit the token budget) where
  slimgpt20-multi translates the same segments correctly. (2)
  slimgpt20-multi-target reaches 28.33 BLEU. Its checkpoint has 5.443B
  parameters like the other 20% systems, its run manifest points at its own
  checkpoint, and 24 of its 300 en-cs hypotheses are identical to
  slimgpt20-multi's and 28 to dense's, so it is a genuinely different pruned
  model.
- 20:12 Interim report 27371383 COMPLETED (1 min 41 s): 15 files in
  `reports/`, efficiency columns empty pending bench.
- 20:24 The pre-commit checks now take 14 min on the login node (load
  average 77 with 51 users; pytest gets about a third of a core). Slow, not
  failing.
- 20:27 Morning write-up drafted from the interim report, the overlap
  analysis and paired bootstraps on the stored COMET segment scores
  (`bootstrap_segment_delta`, the repository's own implementation). Compute
  so far from `sacct`: 6.26 GPU-hours, 801 SBU.
