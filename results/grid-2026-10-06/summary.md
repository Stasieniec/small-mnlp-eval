# Pruning grid report

Suite `alma10-greedy-300`, generated 2026-10-08 19:45 UTC from `runs` and `configs/grid/manifest.json`. Paired bootstrap: 1000 resamples, seed 12345. Missing cells are `-`.

## Coverage

Grid models complete: **192/192** (BLEU, chrF++ and COMET on every direction the model is evaluated on: all 10 for multi models, the model's own pruned directions for pair and dir models; a `-lora` system follows its source); 0 partially scored; 0 without a run. LoRA-repaired systems: 27/27 complete.

Specialists scored on all 10 directions, which feed the transfer section: 60 pair and 120 dir models, of 180 specialists with a run.

MetricX-24 (`scores.metricx.json`): 0/192 grid models (0 on every in-scope direction), 0/27 LoRA systems, dense baseline no.

Dense baseline `alma-7b` (run 63d55369fea1, 10/10 directions): COMET 0.8474, BLEU 30.35, chrF++ 51.14.

grid-state status files: done 192.

| Config | multi | pair | dir |
| :--- | ---: | ---: | ---: |
| SlimGPT ref 20% | 1/1 | 5/5 | 10/10 |
| SlimGPT ref 30% | 1/1 | 5/5 | 10/10 |
| SlimGPT ref 40% | 1/1 | 5/5 | 10/10 |
| SlimGPT gen 20% | 1/1 | 5/5 | 10/10 |
| SlimGPT gen 30% | 1/1 | 5/5 | 10/10 |
| SlimGPT gen 40% | 1/1 | 5/5 | 10/10 |
| FLAP ref 20% | 1/1 | 5/5 | 10/10 |
| FLAP ref 30% | 1/1 | 5/5 | 10/10 |
| FLAP ref 40% | 1/1 | 5/5 | 10/10 |
| FLAP gen 20% | 1/1 | 5/5 | 10/10 |
| FLAP gen 30% | 1/1 | 5/5 | 10/10 |
| FLAP gen 40% | 1/1 | 5/5 | 10/10 |

## Headline: one multi model vs matched specialists

Macro over the 10 directions. For pair and dir, each direction is translated by the matching specialised model: `pair-<l>` for both directions of language l, `dir-<d>` for direction d. Bold marks the best complete scope per metric. The delta columns are specialist minus multi on the macro over the shared directions, so they equal the difference of the macro columns, with a stratified paired bootstrap 95% CI and two-sided p: each resample redraws segment positions within each direction, the same positions for both models, and averages the per-direction mean differences with equal weight. `(n/10)` means only n directions exist yet.

| Method | Calib | Sparsity | multi COMET | multi BLEU | multi chrF++ | pair COMET | pair BLEU | pair chrF++ | dir COMET | dir BLEU | dir chrF++ | pair - multi COMET | dir - multi COMET |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | - | 0% | 0.8474 | 30.35 | 51.14 | 0.8474 | 30.35 | 51.14 | 0.8474 | 30.35 | 51.14 | - | - |
| SlimGPT | ref | 20% | **0.8374** | 27.72 | 49.03 | 0.8369 | **27.97** | 49.22 | 0.8371 | 27.95 | **49.38** | -0.0005 [-0.0026, +0.0015] p=0.640 | -0.0003 [-0.0023, +0.0018] p=0.796 |
| SlimGPT | ref | 30% | 0.8219 | 25.54 | 47.08 | 0.8271 | 26.27 | 47.85 | **0.8282** | **26.43** | **47.91** | +0.0052 [+0.0028, +0.0076] p<0.001 | +0.0063 [+0.0037, +0.0087] p<0.001 |
| SlimGPT | ref | 40% | 0.7914 | 23.13 | 44.56 | **0.8083** | 23.99 | 45.66 | 0.8076 | **24.13** | **45.80** | +0.0169 [+0.0136, +0.0198] p<0.001 | +0.0162 [+0.0129, +0.0193] p<0.001 |
| SlimGPT | gen | 20% | 0.8360 | 27.52 | 48.93 | **0.8377** | **28.13** | **49.37** | 0.8373 | 27.93 | 49.36 | +0.0017 [-0.0004, +0.0037] p=0.105 | +0.0012 [-0.0009, +0.0033] p=0.242 |
| SlimGPT | gen | 30% | 0.8229 | 25.88 | 47.58 | **0.8276** | 26.52 | 48.06 | 0.8267 | **26.58** | **48.12** | +0.0047 [+0.0022, +0.0070] p<0.001 | +0.0038 [+0.0013, +0.0062] p=0.003 |
| SlimGPT | gen | 40% | 0.7905 | 22.92 | 44.44 | **0.8092** | 24.18 | 45.95 | 0.8081 | **24.44** | **45.98** | +0.0187 [+0.0156, +0.0219] p<0.001 | +0.0175 [+0.0142, +0.0209] p<0.001 |
| FLAP | ref | 20% | 0.8221 | 26.81 | 47.89 | 0.8297 | 27.24 | 48.29 | **0.8299** | **27.67** | **48.66** | +0.0076 [+0.0052, +0.0101] p<0.001 | +0.0078 [+0.0053, +0.0104] p<0.001 |
| FLAP | ref | 30% | 0.7823 | 23.66 | 44.56 | 0.8003 | 24.09 | 45.36 | **0.8017** | **24.49** | **45.76** | +0.0180 [+0.0141, +0.0219] p<0.001 | +0.0194 [+0.0158, +0.0232] p<0.001 |
| FLAP | ref | 40% | 0.7393 | **20.50** | 40.94 | **0.7552** | 20.11 | 40.88 | **0.7552** | 20.33 | **41.34** | +0.0159 [+0.0115, +0.0198] p<0.001 | +0.0159 [+0.0114, +0.0200] p<0.001 |
| FLAP | gen | 20% | 0.8209 | 26.70 | 47.78 | 0.8291 | 27.31 | 48.30 | **0.8306** | **27.70** | **48.75** | +0.0083 [+0.0059, +0.0108] p<0.001 | +0.0097 [+0.0073, +0.0126] p<0.001 |
| FLAP | gen | 30% | 0.7830 | 23.97 | 44.79 | 0.8001 | 24.39 | 45.37 | **0.8006** | **24.59** | **45.59** | +0.0172 [+0.0134, +0.0208] p<0.001 | +0.0176 [+0.0141, +0.0213] p<0.001 |
| FLAP | gen | 40% | 0.7392 | 20.12 | 40.79 | 0.7543 | 19.95 | 40.86 | **0.7557** | **20.46** | **41.54** | +0.0151 [+0.0106, +0.0194] p<0.001 | +0.0165 [+0.0118, +0.0213] p<0.001 |

## Headline, MetricX-24 (lower is better)

No MetricX-24 scores (`scores.metricx.json`) for suite `alma10-greedy-300` yet.

## Ref vs gen calibration text

Calibrating on prompt + dense-generated continuation (gen) instead of prompt + reference (ref). Scores are the scope composites of the headline, over the directions both sides have, each direction weighted equally. COMET and MetricX-24 use a stratified paired bootstrap of the macro (segment positions resampled within each direction, the same for both sides); BLEU does the same, recomputing corpus BLEU per direction from sufficient statistics. MetricX-24 is lower-is-better; its delta is also gen minus ref, so a negative MetricX delta means gen is better.

| Method | Sparsity | Scope | COMET ref | COMET gen | gen - ref [95% CI] | p | BLEU ref | BLEU gen | gen - ref [95% CI] | p | MetricX ref | MetricX gen | MetricX gen - ref [95% CI] | p |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlimGPT | 20% | multi | 0.8374 | 0.8360 | -0.0013 [-0.0029, +0.0004] | 0.104 | 27.72 | 27.52 | -0.20 [-0.49, +0.13] | 0.223 | - | - | - | - |
| SlimGPT | 20% | pair | 0.8369 | 0.8377 | +0.0008 [-0.0005, +0.0023] | 0.239 | 27.96 | 28.13 | +0.16 [-0.10, +0.44] | 0.253 | - | - | - | - |
| SlimGPT | 20% | dir | 0.8371 | 0.8373 | +0.0002 [-0.0014, +0.0018] | 0.822 | 27.95 | 27.93 | -0.02 [-0.38, +0.30] | 0.911 | - | - | - | - |
| SlimGPT | 30% | multi | 0.8219 | 0.8229 | +0.0010 [-0.0009, +0.0030] | 0.311 | 25.54 | 25.88 | +0.34 [-0.03, +0.68] | 0.059 | - | - | - | - |
| SlimGPT | 30% | pair | 0.8271 | 0.8276 | +0.0006 [-0.0011, +0.0021] | 0.504 | 26.27 | 26.52 | +0.25 [-0.08, +0.60] | 0.140 | - | - | - | - |
| SlimGPT | 30% | dir | 0.8282 | 0.8267 | -0.0015 [-0.0033, +0.0003] | 0.115 | 26.43 | 26.58 | +0.16 [-0.16, +0.49] | 0.344 | - | - | - | - |
| SlimGPT | 40% | multi | 0.7914 | 0.7905 | -0.0009 [-0.0036, +0.0018] | 0.522 | 23.13 | 22.92 | -0.22 [-0.64, +0.21] | 0.323 | - | - | - | - |
| SlimGPT | 40% | pair | 0.8083 | 0.8092 | +0.0010 [-0.0011, +0.0032] | 0.377 | 23.99 | 24.18 | +0.19 [-0.13, +0.65] | 0.322 | - | - | - | - |
| SlimGPT | 40% | dir | 0.8076 | 0.8081 | +0.0005 [-0.0018, +0.0029] | 0.688 | 24.13 | 24.44 | +0.32 [+0.02, +0.69] | 0.065 | - | - | - | - |
| FLAP | 20% | multi | 0.8221 | 0.8209 | -0.0012 [-0.0032, +0.0008] | 0.216 | 26.81 | 26.70 | -0.11 [-0.41, +0.21] | 0.480 | - | - | - | - |
| FLAP | 20% | pair | 0.8297 | 0.8291 | -0.0006 [-0.0022, +0.0010] | 0.480 | 27.24 | 27.31 | +0.07 [-0.21, +0.40] | 0.675 | - | - | - | - |
| FLAP | 20% | dir | 0.8299 | 0.8306 | +0.0007 [-0.0011, +0.0024] | 0.458 | 27.67 | 27.70 | +0.03 [-0.26, +0.28] | 0.838 | - | - | - | - |
| FLAP | 30% | multi | 0.7823 | 0.7830 | +0.0007 [-0.0020, +0.0035] | 0.607 | 23.66 | 23.97 | +0.31 [-0.09, +0.71] | 0.138 | - | - | - | - |
| FLAP | 30% | pair | 0.8003 | 0.8001 | -0.0001 [-0.0026, +0.0023] | 0.912 | 24.09 | 24.39 | +0.30 [-0.10, +0.73] | 0.149 | - | - | - | - |
| FLAP | 30% | dir | 0.8017 | 0.8006 | -0.0011 [-0.0040, +0.0014] | 0.430 | 24.49 | 24.59 | +0.10 [-0.31, +0.42] | 0.616 | - | - | - | - |
| FLAP | 40% | multi | 0.7393 | 0.7392 | -0.0001 [-0.0035, +0.0032] | 0.949 | 20.50 | 20.12 | -0.38 [-0.73, +0.02] | 0.055 | - | - | - | - |
| FLAP | 40% | pair | 0.7552 | 0.7543 | -0.0009 [-0.0044, +0.0025] | 0.598 | 20.11 | 19.95 | -0.15 [-0.73, +0.36] | 0.577 | - | - | - | - |
| FLAP | 40% | dir | 0.7552 | 0.7557 | +0.0005 [-0.0027, +0.0036] | 0.742 | 20.33 | 20.46 | +0.13 [-0.27, +0.53] | 0.522 | - | - | - | - |

## SlimGPT vs FLAP

Deltas are SlimGPT minus FLAP at the same calibration text, sparsity and scope; positive means SlimGPT is better on COMET and BLEU. Same composites and tests as above. MetricX-24 is lower-is-better; its delta is also SlimGPT minus FLAP, so a negative MetricX delta means SlimGPT is better.

| Calib | Sparsity | Scope | COMET FLAP | COMET SlimGPT | SlimGPT - FLAP [95% CI] | p | BLEU FLAP | BLEU SlimGPT | SlimGPT - FLAP [95% CI] | p | MetricX FLAP | MetricX SlimGPT | MetricX SlimGPT - FLAP [95% CI] | p |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ref | 20% | multi | 0.8221 | 0.8374 | +0.0152 [+0.0126, +0.0179] | <0.001 | 26.81 | 27.72 | +0.91 [+0.30, +1.44] | 0.004 | - | - | - | - |
| ref | 20% | pair | 0.8297 | 0.8369 | +0.0072 [+0.0047, +0.0094] | <0.001 | 27.24 | 27.96 | +0.72 [+0.20, +1.25] | 0.007 | - | - | - | - |
| ref | 20% | dir | 0.8299 | 0.8371 | +0.0072 [+0.0046, +0.0097] | <0.001 | 27.67 | 27.95 | +0.28 [-0.24, +0.72] | 0.256 | - | - | - | - |
| ref | 30% | multi | 0.7823 | 0.8219 | +0.0396 [+0.0358, +0.0435] | <0.001 | 23.66 | 25.54 | +1.88 [+1.22, +2.63] | <0.001 | - | - | - | - |
| ref | 30% | pair | 0.8003 | 0.8271 | +0.0268 [+0.0235, +0.0302] | <0.001 | 24.09 | 26.27 | +2.18 [+1.51, +2.89] | <0.001 | - | - | - | - |
| ref | 30% | dir | 0.8017 | 0.8282 | +0.0265 [+0.0233, +0.0298] | <0.001 | 24.49 | 26.43 | +1.93 [+1.35, +2.54] | <0.001 | - | - | - | - |
| ref | 40% | multi | 0.7393 | 0.7914 | +0.0521 [+0.0479, +0.0561] | <0.001 | 20.50 | 23.13 | +2.64 [+1.96, +3.28] | <0.001 | - | - | - | - |
| ref | 40% | pair | 0.7552 | 0.8083 | +0.0531 [+0.0488, +0.0574] | <0.001 | 20.11 | 23.99 | +3.88 [+3.07, +4.53] | <0.001 | - | - | - | - |
| ref | 40% | dir | 0.7552 | 0.8076 | +0.0524 [+0.0480, +0.0562] | <0.001 | 20.33 | 24.13 | +3.79 [+3.19, +4.37] | <0.001 | - | - | - | - |
| gen | 20% | multi | 0.8209 | 0.8360 | +0.0151 [+0.0124, +0.0179] | <0.001 | 26.70 | 27.52 | +0.81 [+0.21, +1.37] | 0.005 | - | - | - | - |
| gen | 20% | pair | 0.8291 | 0.8377 | +0.0086 [+0.0059, +0.0109] | <0.001 | 27.31 | 28.13 | +0.82 [+0.31, +1.33] | 0.005 | - | - | - | - |
| gen | 20% | dir | 0.8306 | 0.8373 | +0.0066 [+0.0042, +0.0089] | <0.001 | 27.70 | 27.93 | +0.23 [-0.32, +0.75] | 0.366 | - | - | - | - |
| gen | 30% | multi | 0.7830 | 0.8229 | +0.0399 [+0.0361, +0.0440] | <0.001 | 23.97 | 25.88 | +1.91 [+1.27, +2.69] | <0.001 | - | - | - | - |
| gen | 30% | pair | 0.8001 | 0.8276 | +0.0275 [+0.0241, +0.0308] | <0.001 | 24.39 | 26.52 | +2.13 [+1.47, +2.84] | <0.001 | - | - | - | - |
| gen | 30% | dir | 0.8006 | 0.8267 | +0.0261 [+0.0226, +0.0299] | <0.001 | 24.59 | 26.58 | +1.99 [+1.44, +2.67] | <0.001 | - | - | - | - |
| gen | 40% | multi | 0.7392 | 0.7905 | +0.0514 [+0.0471, +0.0558] | <0.001 | 20.12 | 22.92 | +2.80 [+2.10, +3.49] | <0.001 | - | - | - | - |
| gen | 40% | pair | 0.7543 | 0.8092 | +0.0550 [+0.0508, +0.0592] | <0.001 | 19.95 | 24.18 | +4.22 [+3.54, +4.95] | <0.001 | - | - | - | - |
| gen | 40% | dir | 0.7557 | 0.8081 | +0.0524 [+0.0482, +0.0566] | <0.001 | 20.46 | 24.44 | +3.98 [+3.39, +4.59] | <0.001 | - | - | - | - |

## Per-direction scores, multi models

One row per multi model (plus dense). into-EN and out-of-EN are macros over the five directions on each side.

### COMET

| System | cs-en | de-en | is-en | ru-en | zh-en | into-EN | en-cs | en-de | en-is | en-ru | en-zh | out-of-EN | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8394 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8554 | 0.8474 |
| SlimGPT ref 20% | 0.8423 | 0.8336 | 0.8590 | 0.8360 | 0.7891 | 0.8320 | 0.8519 | 0.8445 | 0.8278 | 0.8493 | 0.8399 | 0.8427 | 0.8374 |
| SlimGPT ref 30% | 0.8367 | 0.8237 | 0.8504 | 0.8294 | 0.7767 | 0.8234 | 0.8418 | 0.8247 | 0.8048 | 0.8230 | 0.8079 | 0.8204 | 0.8219 |
| SlimGPT ref 40% | 0.8223 | 0.8152 | 0.8337 | 0.8167 | 0.7510 | 0.8078 | 0.7899 | 0.7970 | 0.7479 | 0.7933 | 0.7469 | 0.7750 | 0.7914 |
| SlimGPT gen 20% | 0.8436 | 0.8331 | 0.8571 | 0.8370 | 0.7895 | 0.8321 | 0.8432 | 0.8460 | 0.8282 | 0.8451 | 0.8373 | 0.8400 | 0.8360 |
| SlimGPT gen 30% | 0.8378 | 0.8262 | 0.8496 | 0.8283 | 0.7817 | 0.8247 | 0.8312 | 0.8332 | 0.8016 | 0.8268 | 0.8126 | 0.8211 | 0.8229 |
| SlimGPT gen 40% | 0.8205 | 0.8168 | 0.8341 | 0.8122 | 0.7503 | 0.8068 | 0.7893 | 0.7971 | 0.7410 | 0.7905 | 0.7537 | 0.7743 | 0.7905 |
| FLAP ref 20% | 0.8398 | 0.8339 | 0.8560 | 0.8294 | 0.7766 | 0.8272 | 0.8345 | 0.8231 | 0.7817 | 0.8344 | 0.8117 | 0.8171 | 0.8221 |
| FLAP ref 30% | 0.8235 | 0.8207 | 0.8351 | 0.8131 | 0.7520 | 0.8089 | 0.7833 | 0.7987 | 0.6805 | 0.7938 | 0.7218 | 0.7556 | 0.7823 |
| FLAP ref 40% | 0.8138 | 0.8063 | 0.8032 | 0.7899 | 0.7204 | 0.7867 | 0.7172 | 0.7625 | 0.5826 | 0.7455 | 0.6513 | 0.6918 | 0.7393 |
| FLAP gen 20% | 0.8396 | 0.8337 | 0.8559 | 0.8319 | 0.7747 | 0.8272 | 0.8310 | 0.8268 | 0.7750 | 0.8332 | 0.8069 | 0.8146 | 0.8209 |
| FLAP gen 30% | 0.8278 | 0.8184 | 0.8340 | 0.8149 | 0.7576 | 0.8105 | 0.7859 | 0.7949 | 0.6802 | 0.7973 | 0.7187 | 0.7554 | 0.7830 |
| FLAP gen 40% | 0.8088 | 0.8068 | 0.8061 | 0.7902 | 0.7227 | 0.7869 | 0.7156 | 0.7551 | 0.5718 | 0.7557 | 0.6588 | 0.6914 | 0.7392 |

### BLEU

| System | cs-en | de-en | is-en | ru-en | zh-en | into-EN | en-cs | en-de | en-is | en-ru | en-zh | out-of-EN | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (alma-7b) | 41.99 | 31.32 | 38.25 | 36.94 | 19.84 | 33.67 | 24.93 | 27.99 | 23.75 | 25.01 | 33.50 | 27.04 | 30.35 |
| SlimGPT ref 20% | 37.27 | 27.91 | 35.99 | 34.00 | 18.54 | 30.74 | 22.39 | 25.63 | 21.53 | 22.97 | 30.92 | 24.69 | 27.72 |
| SlimGPT ref 30% | 35.56 | 27.08 | 33.90 | 33.61 | 16.68 | 29.37 | 19.73 | 22.61 | 18.70 | 20.09 | 27.43 | 21.71 | 25.54 |
| SlimGPT ref 40% | 32.85 | 25.66 | 31.46 | 31.75 | 15.41 | 27.43 | 17.57 | 21.14 | 15.61 | 17.82 | 22.08 | 18.84 | 23.13 |
| SlimGPT gen 20% | 37.21 | 28.31 | 35.76 | 35.05 | 18.40 | 30.95 | 22.18 | 24.34 | 21.11 | 22.38 | 30.45 | 24.09 | 27.52 |
| SlimGPT gen 30% | 36.92 | 27.36 | 33.49 | 34.02 | 17.09 | 29.77 | 20.10 | 23.11 | 18.66 | 19.82 | 28.20 | 21.98 | 25.88 |
| SlimGPT gen 40% | 32.43 | 25.68 | 31.58 | 30.53 | 15.17 | 27.08 | 17.45 | 21.35 | 15.40 | 17.61 | 22.00 | 18.76 | 22.92 |
| FLAP ref 20% | 37.82 | 29.38 | 35.51 | 35.19 | 17.52 | 31.08 | 21.64 | 24.33 | 17.37 | 21.72 | 27.62 | 22.54 | 26.81 |
| FLAP ref 30% | 36.06 | 28.33 | 32.43 | 31.35 | 14.98 | 28.63 | 19.93 | 21.84 | 13.93 | 18.25 | 19.47 | 18.68 | 23.66 |
| FLAP ref 40% | 31.59 | 26.03 | 28.75 | 29.74 | 12.88 | 25.80 | 15.67 | 18.23 | 10.62 | 16.07 | 15.39 | 15.20 | 20.50 |
| FLAP gen 20% | 37.89 | 29.45 | 35.41 | 34.84 | 17.06 | 30.93 | 20.62 | 24.76 | 17.76 | 21.38 | 27.87 | 22.48 | 26.70 |
| FLAP gen 30% | 36.05 | 28.71 | 33.03 | 32.35 | 15.60 | 29.15 | 20.15 | 22.23 | 14.15 | 17.80 | 19.63 | 18.79 | 23.97 |
| FLAP gen 40% | 30.51 | 25.70 | 28.26 | 29.69 | 12.99 | 25.43 | 15.52 | 18.33 | 9.99 | 15.28 | 14.95 | 14.81 | 20.12 |

### MetricX-24 (lower is better)

No MetricX-24 scores yet.

## Sparsity curve (macro COMET)

| Method | Calib | Scope | 0% (dense) | 20% | 30% | 40% |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| SlimGPT | ref | multi | 0.8474 | 0.8374 | 0.8219 | 0.7914 |
| SlimGPT | ref | pair | 0.8474 | 0.8369 | 0.8271 | 0.8083 |
| SlimGPT | ref | dir | 0.8474 | 0.8371 | 0.8282 | 0.8076 |
| SlimGPT | gen | multi | 0.8474 | 0.8360 | 0.8229 | 0.7905 |
| SlimGPT | gen | pair | 0.8474 | 0.8377 | 0.8276 | 0.8092 |
| SlimGPT | gen | dir | 0.8474 | 0.8373 | 0.8267 | 0.8081 |
| FLAP | ref | multi | 0.8474 | 0.8221 | 0.7823 | 0.7393 |
| FLAP | ref | pair | 0.8474 | 0.8297 | 0.8003 | 0.7552 |
| FLAP | ref | dir | 0.8474 | 0.8299 | 0.8017 | 0.7552 |
| FLAP | gen | multi | 0.8474 | 0.8209 | 0.7830 | 0.7392 |
| FLAP | gen | pair | 0.8474 | 0.8291 | 0.8001 | 0.7543 |
| FLAP | gen | dir | 0.8474 | 0.8306 | 0.8006 | 0.7557 |

## Sparsity curve (macro MetricX-24, lower is better)

No MetricX-24 scores yet.

## Transfer

Only specialised models scored on all 10 directions enter: 60 pair and 120 dir models here, of 180 specialists with a run (the Models columns count them per config). Specialists evaluated only on their own directions, as on the full suite, are left out. Each cell is the mean COMET over the matrix cells in that category, with the mean of (specialist minus the multi model of the same config, on the same direction) in parentheses. Pair: own = the two directions of the model's language, other = the remaining eight. Dir: own = the calibration direction, reverse = its reverse, same target / same source = other directions sharing the target / source language (into-English models have same-target neighbours, out-of-English models same-source ones), other = the rest. Full matrices are in `transfer/<config>.md` and `.csv`.

### Pair models (5 x 10)

| Config | Models | Own pair | Other 8 | Own - other |
| :--- | ---: | ---: | ---: | ---: |
| SlimGPT ref 20% | 5/5 | 0.8369 (-0.0005) | 0.8054 (-0.0319) | +0.0315 |
| SlimGPT ref 30% | 5/5 | 0.8271 (+0.0052) | 0.7608 (-0.0611) | +0.0663 |
| SlimGPT ref 40% | 5/5 | 0.8083 (+0.0169) | 0.6994 (-0.0920) | +0.1088 |
| SlimGPT gen 20% | 5/5 | 0.8377 (+0.0017) | 0.8051 (-0.0309) | +0.0326 |
| SlimGPT gen 30% | 5/5 | 0.8276 (+0.0047) | 0.7615 (-0.0615) | +0.0662 |
| SlimGPT gen 40% | 5/5 | 0.8092 (+0.0187) | 0.6978 (-0.0927) | +0.1114 |
| FLAP ref 20% | 5/5 | 0.8297 (+0.0076) | 0.8089 (-0.0132) | +0.0208 |
| FLAP ref 30% | 5/5 | 0.8003 (+0.0180) | 0.7456 (-0.0366) | +0.0546 |
| FLAP ref 40% | 5/5 | 0.7552 (+0.0159) | 0.6723 (-0.0670) | +0.0829 |
| FLAP gen 20% | 5/5 | 0.8291 (+0.0083) | 0.8089 (-0.0120) | +0.0202 |
| FLAP gen 30% | 5/5 | 0.8001 (+0.0172) | 0.7448 (-0.0381) | +0.0553 |
| FLAP gen 40% | 5/5 | 0.7543 (+0.0151) | 0.6712 (-0.0680) | +0.0831 |

### Direction models (10 x 10)

| Config | Models | Own | Reverse | Same target | Same source | Other | Own - reverse |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlimGPT ref 20% | 10/10 | 0.8371 (-0.0003) | 0.8364 (-0.0010) | 0.8276 (-0.0044) | 0.7809 (-0.0618) | 0.7996 (-0.0378) | +0.0007 |
| SlimGPT ref 30% | 10/10 | 0.8282 (+0.0063) | 0.8232 (+0.0013) | 0.8185 (-0.0049) | 0.7050 (-0.1154) | 0.7546 (-0.0673) | +0.0050 |
| SlimGPT ref 40% | 10/10 | 0.8076 (+0.0162) | 0.8002 (+0.0088) | 0.8008 (-0.0070) | 0.5935 (-0.1815) | 0.6884 (-0.1030) | +0.0074 |
| SlimGPT gen 20% | 10/10 | 0.8373 (+0.0012) | 0.8368 (+0.0008) | 0.8284 (-0.0037) | 0.7809 (-0.0591) | 0.7980 (-0.0381) | +0.0004 |
| SlimGPT gen 30% | 10/10 | 0.8267 (+0.0038) | 0.8231 (+0.0002) | 0.8189 (-0.0058) | 0.7032 (-0.1179) | 0.7532 (-0.0697) | +0.0036 |
| SlimGPT gen 40% | 10/10 | 0.8081 (+0.0175) | 0.7965 (+0.0060) | 0.8010 (-0.0057) | 0.5959 (-0.1784) | 0.6878 (-0.1027) | +0.0115 |
| FLAP ref 20% | 10/10 | 0.8299 (+0.0078) | 0.8287 (+0.0066) | 0.8264 (-0.0008) | 0.7946 (-0.0225) | 0.8057 (-0.0164) | +0.0012 |
| FLAP ref 30% | 10/10 | 0.8017 (+0.0194) | 0.8008 (+0.0186) | 0.8075 (-0.0013) | 0.6811 (-0.0745) | 0.7459 (-0.0363) | +0.0009 |
| FLAP ref 40% | 10/10 | 0.7552 (+0.0159) | 0.7572 (+0.0179) | 0.7806 (-0.0062) | 0.5588 (-0.1330) | 0.6730 (-0.0662) | -0.0019 |
| FLAP gen 20% | 10/10 | 0.8306 (+0.0097) | 0.8286 (+0.0077) | 0.8266 (-0.0006) | 0.7935 (-0.0211) | 0.8052 (-0.0157) | +0.0021 |
| FLAP gen 30% | 10/10 | 0.8006 (+0.0176) | 0.8010 (+0.0180) | 0.8085 (-0.0020) | 0.6810 (-0.0744) | 0.7460 (-0.0370) | -0.0004 |
| FLAP gen 40% | 10/10 | 0.7557 (+0.0165) | 0.7556 (+0.0164) | 0.7802 (-0.0067) | 0.5524 (-0.1390) | 0.6729 (-0.0663) | +0.0001 |

## Behaviour, multi models

Macro over directions of the per-direction rates (share of all segments). Hit budget = generation ran out of tokens without EOS; truncated = the hypothesis itself was cut off; repetition = looping flagged by post-processing; length ratio = hypothesis / reference characters. All systems are in long.csv.

| System | Off-target | Repetition | Hit budget | Truncated | Empty | Source copy | Length ratio |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (alma-7b) | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% | 0.964 |
| SlimGPT ref 20% | 0.4% | 0.0% | 0.1% | 0.1% | 0.0% | 0.4% | 0.965 |
| SlimGPT ref 30% | 0.3% | 0.0% | 0.1% | 0.1% | 0.0% | 0.2% | 0.957 |
| SlimGPT ref 40% | 0.5% | 0.0% | 0.2% | 0.2% | 0.0% | 0.3% | 0.948 |
| SlimGPT gen 20% | 0.3% | 0.0% | 0.1% | 0.1% | 0.0% | 0.4% | 0.966 |
| SlimGPT gen 30% | 0.3% | 0.0% | 0.1% | 0.1% | 0.0% | 0.3% | 0.959 |
| SlimGPT gen 40% | 0.5% | 0.1% | 0.3% | 0.3% | 0.0% | 0.3% | 0.951 |
| FLAP ref 20% | 0.3% | 0.0% | 0.1% | 0.1% | 0.0% | 0.1% | 0.954 |
| FLAP ref 30% | 0.4% | 0.5% | 0.9% | 0.9% | 0.0% | 0.2% | 0.978 |
| FLAP ref 40% | 0.3% | 0.4% | 1.3% | 1.3% | 0.0% | 0.2% | 0.960 |
| FLAP gen 20% | 0.2% | 0.0% | 0.2% | 0.2% | 0.0% | 0.1% | 0.960 |
| FLAP gen 30% | 0.4% | 0.3% | 1.0% | 1.0% | 0.0% | 0.2% | 0.979 |
| FLAP gen 40% | 0.3% | 0.4% | 1.1% | 1.1% | 0.0% | 0.3% | 0.949 |

## Repair (LoRA)

Macro over the directions all three systems have (a specialist's repair is evaluated on its own directions on the full suite). Recovered = share of the pruning loss won back, (repaired - pruned) / (dense - pruned): 1 means the repair closed the whole gap to dense, 0 means nothing; the ratio is the same for a lower-is-better metric. The deltas are repaired minus pruned on the macro, stratified paired bootstrap; on MetricX-24 (lower is better) a negative delta means the repair helped.

| System | Metric | Dense | Pruned | Repaired | Repaired - pruned | Recovered |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| alma-7b-flap30-gen-multi-lora | COMET | 0.8474 | 0.7830 | 0.8242 | +0.0412 | 64.0% |
| alma-7b-flap30-gen-multi-lora | BLEU | 30.35 | 23.97 | 27.09 | +3.12 | 49.0% |
| alma-7b-flap30-gen-multi-lora | chrF++ | 51.14 | 44.79 | 48.10 | +3.32 | 52.2% |
| alma-7b-flap30-gen-pair-cs-lora | COMET | 0.8474 | 0.7612 | 0.7925 | +0.0313 | 36.3% |
| alma-7b-flap30-gen-pair-cs-lora | BLEU | 30.35 | 22.39 | 24.89 | +2.50 | 31.4% |
| alma-7b-flap30-gen-pair-cs-lora | chrF++ | 51.14 | 43.37 | 45.43 | +2.06 | 26.5% |
| alma-7b-flap30-gen-pair-de-lora | COMET | 0.8474 | 0.7506 | 0.7834 | +0.0328 | 33.9% |
| alma-7b-flap30-gen-pair-de-lora | BLEU | 30.35 | 21.43 | 24.65 | +3.22 | 36.1% |
| alma-7b-flap30-gen-pair-de-lora | chrF++ | 51.14 | 42.65 | 45.41 | +2.77 | 32.6% |
| alma-7b-flap30-gen-pair-is-lora | COMET | 0.8474 | 0.7610 | 0.7931 | +0.0320 | 37.1% |
| alma-7b-flap30-gen-pair-is-lora | BLEU | 30.35 | 20.84 | 22.70 | +1.86 | 19.5% |
| alma-7b-flap30-gen-pair-is-lora | chrF++ | 51.14 | 42.09 | 43.97 | +1.88 | 20.7% |
| alma-7b-flap30-gen-pair-ru-lora | COMET | 0.8474 | 0.7543 | 0.7619 | +0.0075 | 8.1% |
| alma-7b-flap30-gen-pair-ru-lora | BLEU | 30.35 | 21.89 | 20.57 | -1.32 | -15.7% |
| alma-7b-flap30-gen-pair-ru-lora | chrF++ | 51.14 | 42.87 | 35.96 | -6.92 | -83.7% |
| alma-7b-flap30-gen-pair-zh-lora | COMET | 0.8474 | 0.7524 | 0.7761 | +0.0237 | 25.0% |
| alma-7b-flap30-gen-pair-zh-lora | BLEU | 30.35 | 21.92 | 23.91 | +1.99 | 23.7% |
| alma-7b-flap30-gen-pair-zh-lora | chrF++ | 51.14 | 42.74 | 44.88 | +2.14 | 25.5% |
| alma-7b-flap40-gen-multi-lora | COMET | 0.8474 | 0.7392 | 0.8110 | +0.0719 | 66.4% |
| alma-7b-flap40-gen-multi-lora | BLEU | 30.35 | 20.12 | 25.85 | +5.73 | 56.0% |
| alma-7b-flap40-gen-multi-lora | chrF++ | 51.14 | 40.79 | 47.05 | +6.26 | 60.5% |
| alma-7b-flap40-gen-pair-cs-lora | COMET | 0.8474 | 0.6883 | 0.7500 | +0.0618 | 38.8% |
| alma-7b-flap40-gen-pair-cs-lora | BLEU | 30.35 | 17.00 | 21.51 | +4.51 | 33.8% |
| alma-7b-flap40-gen-pair-cs-lora | chrF++ | 51.14 | 37.39 | 42.47 | +5.08 | 37.0% |
| alma-7b-flap40-gen-pair-de-lora | COMET | 0.8474 | 0.6735 | 0.7341 | +0.0607 | 34.9% |
| alma-7b-flap40-gen-pair-de-lora | BLEU | 30.35 | 16.31 | 21.07 | +4.76 | 33.9% |
| alma-7b-flap40-gen-pair-de-lora | chrF++ | 51.14 | 36.64 | 41.75 | +5.11 | 35.2% |
| alma-7b-flap40-gen-pair-is-lora | COMET | 0.8474 | 0.6792 | 0.7426 | +0.0634 | 37.7% |
| alma-7b-flap40-gen-pair-is-lora | BLEU | 30.35 | 14.96 | 18.61 | +3.64 | 23.7% |
| alma-7b-flap40-gen-pair-is-lora | chrF++ | 51.14 | 35.26 | 39.75 | +4.50 | 28.3% |
| alma-7b-flap40-gen-pair-ru-lora | COMET | 0.8474 | 0.7019 | 0.7485 | +0.0466 | 32.0% |
| alma-7b-flap40-gen-pair-ru-lora | BLEU | 30.35 | 17.90 | 21.68 | +3.78 | 30.4% |
| alma-7b-flap40-gen-pair-ru-lora | chrF++ | 51.14 | 38.41 | 42.04 | +3.64 | 28.6% |
| alma-7b-flap40-gen-pair-zh-lora | COMET | 0.8474 | 0.6962 | 0.7388 | +0.0426 | 28.2% |
| alma-7b-flap40-gen-pair-zh-lora | BLEU | 30.35 | 17.65 | 21.38 | +3.74 | 29.4% |
| alma-7b-flap40-gen-pair-zh-lora | chrF++ | 51.14 | 37.97 | 42.12 | +4.15 | 31.5% |
| alma-7b-flap40-ref-multi-lora | COMET | 0.8474 | 0.7393 | 0.8113 | +0.0720 | 66.6% |
| alma-7b-flap40-ref-multi-lora | BLEU | 30.35 | 20.50 | 25.73 | +5.23 | 53.1% |
| alma-7b-flap40-ref-multi-lora | chrF++ | 51.14 | 40.94 | 46.84 | +5.90 | 57.8% |
| alma-7b-slimgpt20-ref-multi-lora | COMET | 0.8474 | 0.8374 | 0.8381 | +0.0007 | 7.1% |
| alma-7b-slimgpt20-ref-multi-lora | BLEU | 30.35 | 27.72 | 28.41 | +0.69 | 26.3% |
| alma-7b-slimgpt20-ref-multi-lora | chrF++ | 51.14 | 49.03 | 49.43 | +0.40 | 19.0% |
| alma-7b-slimgpt30-gen-multi-lora | COMET | 0.8474 | 0.8229 | 0.8324 | +0.0095 | 38.9% |
| alma-7b-slimgpt30-gen-multi-lora | BLEU | 30.35 | 25.88 | 27.59 | +1.72 | 38.4% |
| alma-7b-slimgpt30-gen-multi-lora | chrF++ | 51.14 | 47.58 | 48.65 | +1.07 | 30.0% |
| alma-7b-slimgpt30-gen-pair-cs-lora | COMET | 0.8474 | 0.7858 | 0.7916 | +0.0057 | 9.3% |
| alma-7b-slimgpt30-gen-pair-cs-lora | BLEU | 30.35 | 23.18 | 23.92 | +0.74 | 10.3% |
| alma-7b-slimgpt30-gen-pair-cs-lora | chrF++ | 51.14 | 44.66 | 45.23 | +0.57 | 8.8% |
| alma-7b-slimgpt30-gen-pair-de-lora | COMET | 0.8474 | 0.7757 | 0.7833 | +0.0077 | 10.7% |
| alma-7b-slimgpt30-gen-pair-de-lora | BLEU | 30.35 | 22.68 | 23.98 | +1.30 | 17.0% |
| alma-7b-slimgpt30-gen-pair-de-lora | chrF++ | 51.14 | 44.31 | 45.17 | +0.86 | 12.6% |
| alma-7b-slimgpt30-gen-pair-is-lora | COMET | 0.8474 | 0.7824 | 0.7884 | +0.0060 | 9.2% |
| alma-7b-slimgpt30-gen-pair-is-lora | BLEU | 30.35 | 22.22 | 22.02 | -0.20 | -2.4% |
| alma-7b-slimgpt30-gen-pair-is-lora | chrF++ | 51.14 | 44.21 | 44.00 | -0.21 | -3.0% |
| alma-7b-slimgpt30-gen-pair-ru-lora | COMET | 0.8474 | 0.7841 | 0.7844 | +0.0003 | 0.5% |
| alma-7b-slimgpt30-gen-pair-ru-lora | BLEU | 30.35 | 23.52 | 23.10 | -0.43 | -6.3% |
| alma-7b-slimgpt30-gen-pair-ru-lora | chrF++ | 51.14 | 44.67 | 41.75 | -2.92 | -45.0% |
| alma-7b-slimgpt30-gen-pair-zh-lora | COMET | 0.8474 | 0.7454 | 0.7545 | +0.0090 | 8.9% |
| alma-7b-slimgpt30-gen-pair-zh-lora | BLEU | 30.35 | 21.03 | 21.63 | +0.60 | 6.5% |
| alma-7b-slimgpt30-gen-pair-zh-lora | chrF++ | 51.14 | 42.42 | 41.02 | -1.40 | -16.0% |
| alma-7b-slimgpt40-gen-multi-lora | COMET | 0.8474 | 0.7905 | 0.8179 | +0.0274 | 48.2% |
| alma-7b-slimgpt40-gen-multi-lora | BLEU | 30.35 | 22.92 | 26.36 | +3.44 | 46.3% |
| alma-7b-slimgpt40-gen-multi-lora | chrF++ | 51.14 | 44.44 | 47.40 | +2.96 | 44.2% |
| alma-7b-slimgpt40-gen-pair-cs-lora | COMET | 0.8474 | 0.7381 | 0.7522 | +0.0141 | 12.9% |
| alma-7b-slimgpt40-gen-pair-cs-lora | BLEU | 30.35 | 20.16 | 20.95 | +0.78 | 7.7% |
| alma-7b-slimgpt40-gen-pair-cs-lora | chrF++ | 51.14 | 41.58 | 42.61 | +1.03 | 10.8% |
| alma-7b-slimgpt40-gen-pair-de-lora | COMET | 0.8474 | 0.7166 | 0.7324 | +0.0158 | 12.1% |
| alma-7b-slimgpt40-gen-pair-de-lora | BLEU | 30.35 | 19.49 | 19.09 | -0.40 | -3.6% |
| alma-7b-slimgpt40-gen-pair-de-lora | chrF++ | 51.14 | 40.37 | 37.83 | -2.54 | -23.6% |
| alma-7b-slimgpt40-gen-pair-is-lora | COMET | 0.8474 | 0.7265 | 0.7465 | +0.0201 | 16.6% |
| alma-7b-slimgpt40-gen-pair-is-lora | BLEU | 30.35 | 17.77 | 18.64 | +0.87 | 6.9% |
| alma-7b-slimgpt40-gen-pair-is-lora | chrF++ | 51.14 | 39.58 | 40.55 | +0.97 | 8.4% |
| alma-7b-slimgpt40-gen-pair-ru-lora | COMET | 0.8474 | 0.7217 | 0.7462 | +0.0246 | 19.5% |
| alma-7b-slimgpt40-gen-pair-ru-lora | BLEU | 30.35 | 19.90 | 21.66 | +1.76 | 16.8% |
| alma-7b-slimgpt40-gen-pair-ru-lora | chrF++ | 51.14 | 40.51 | 42.00 | +1.49 | 14.0% |
| alma-7b-slimgpt40-gen-pair-zh-lora | COMET | 0.8474 | 0.6977 | 0.7116 | +0.0139 | 9.3% |
| alma-7b-slimgpt40-gen-pair-zh-lora | BLEU | 30.35 | 17.48 | 17.79 | +0.31 | 2.4% |
| alma-7b-slimgpt40-gen-pair-zh-lora | chrF++ | 51.14 | 37.23 | 33.80 | -3.43 | -24.6% |
| alma-7b-slimgpt40-ref-multi-lora | COMET | 0.8474 | 0.7914 | 0.8193 | +0.0279 | 49.9% |
| alma-7b-slimgpt40-ref-multi-lora | BLEU | 30.35 | 23.13 | 26.13 | +2.99 | 41.5% |
| alma-7b-slimgpt40-ref-multi-lora | chrF++ | 51.14 | 44.56 | 47.40 | +2.84 | 43.2% |

Repaired - pruned COMET with 95% CI:

- alma-7b-flap30-gen-multi-lora: +0.0412 [+0.0375, +0.0450] p<0.001
- alma-7b-flap30-gen-pair-cs-lora: +0.0313 [+0.0273, +0.0355] p<0.001
- alma-7b-flap30-gen-pair-de-lora: +0.0328 [+0.0289, +0.0368] p<0.001
- alma-7b-flap30-gen-pair-is-lora: +0.0320 [+0.0280, +0.0359] p<0.001
- alma-7b-flap30-gen-pair-ru-lora: +0.0075 [+0.0033, +0.0115] p<0.001
- alma-7b-flap30-gen-pair-zh-lora: +0.0237 [+0.0200, +0.0273] p<0.001
- alma-7b-flap40-gen-multi-lora: +0.0719 [+0.0677, +0.0761] p<0.001
- alma-7b-flap40-gen-pair-cs-lora: +0.0618 [+0.0572, +0.0664] p<0.001
- alma-7b-flap40-gen-pair-de-lora: +0.0607 [+0.0558, +0.0655] p<0.001
- alma-7b-flap40-gen-pair-is-lora: +0.0634 [+0.0584, +0.0686] p<0.001
- alma-7b-flap40-gen-pair-ru-lora: +0.0466 [+0.0420, +0.0509] p<0.001
- alma-7b-flap40-gen-pair-zh-lora: +0.0426 [+0.0385, +0.0470] p<0.001
- alma-7b-flap40-ref-multi-lora: +0.0720 [+0.0677, +0.0755] p<0.001
- alma-7b-slimgpt20-ref-multi-lora: +0.0007 [-0.0014, +0.0030] p=0.530
- alma-7b-slimgpt30-gen-multi-lora: +0.0095 [+0.0070, +0.0122] p<0.001
- alma-7b-slimgpt30-gen-pair-cs-lora: +0.0057 [+0.0025, +0.0089] p=0.004
- alma-7b-slimgpt30-gen-pair-de-lora: +0.0077 [+0.0042, +0.0108] p<0.001
- alma-7b-slimgpt30-gen-pair-is-lora: +0.0060 [+0.0030, +0.0091] p<0.001
- alma-7b-slimgpt30-gen-pair-ru-lora: +0.0003 [-0.0031, +0.0037] p=0.868
- alma-7b-slimgpt30-gen-pair-zh-lora: +0.0090 [+0.0056, +0.0126] p<0.001
- alma-7b-slimgpt40-gen-multi-lora: +0.0274 [+0.0238, +0.0308] p<0.001
- alma-7b-slimgpt40-gen-pair-cs-lora: +0.0141 [+0.0107, +0.0176] p<0.001
- alma-7b-slimgpt40-gen-pair-de-lora: +0.0158 [+0.0113, +0.0203] p<0.001
- alma-7b-slimgpt40-gen-pair-is-lora: +0.0201 [+0.0162, +0.0240] p<0.001
- alma-7b-slimgpt40-gen-pair-ru-lora: +0.0246 [+0.0203, +0.0284] p<0.001
- alma-7b-slimgpt40-gen-pair-zh-lora: +0.0139 [+0.0102, +0.0180] p<0.001
- alma-7b-slimgpt40-ref-multi-lora: +0.0279 [+0.0249, +0.0309] p<0.001

### Per-direction COMET

The macro column averages the directions all three systems have (each system's own while they share none yet).

| System | cs-en | de-en | is-en | ru-en | zh-en | en-cs | en-de | en-is | en-ru | en-zh | macro |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap30-gen-multi | 0.8278 | 0.8184 | 0.8340 | 0.8149 | 0.7576 | 0.7859 | 0.7949 | 0.6802 | 0.7973 | 0.7187 | 0.7830 |
| alma-7b-flap30-gen-multi-lora | 0.8393 | 0.8291 | 0.8435 | 0.8288 | 0.7733 | 0.8326 | 0.8318 | 0.7944 | 0.8414 | 0.8275 | 0.8242 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap30-gen-pair-cs | 0.8280 | 0.8190 | 0.8356 | 0.8191 | 0.7335 | 0.8132 | 0.7762 | 0.5644 | 0.7951 | 0.6272 | 0.7612 |
| alma-7b-flap30-gen-pair-cs-lora | 0.8360 | 0.8273 | 0.8500 | 0.8209 | 0.7598 | 0.8404 | 0.8021 | 0.6470 | 0.8196 | 0.7217 | 0.7925 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap30-gen-pair-de | 0.8250 | 0.8187 | 0.8366 | 0.8177 | 0.7387 | 0.7160 | 0.8046 | 0.5812 | 0.7804 | 0.5869 | 0.7506 |
| alma-7b-flap30-gen-pair-de-lora | 0.8338 | 0.8274 | 0.8473 | 0.8264 | 0.7622 | 0.7756 | 0.8289 | 0.6365 | 0.8050 | 0.6906 | 0.7834 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap30-gen-pair-is | 0.8262 | 0.8217 | 0.8335 | 0.8127 | 0.7457 | 0.6998 | 0.7626 | 0.7615 | 0.7520 | 0.5945 | 0.7610 |
| alma-7b-flap30-gen-pair-is-lora | 0.8271 | 0.8262 | 0.8447 | 0.8216 | 0.7617 | 0.7644 | 0.7878 | 0.7990 | 0.7942 | 0.7038 | 0.7931 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap30-gen-pair-ru | 0.8225 | 0.8197 | 0.8297 | 0.8117 | 0.7422 | 0.7435 | 0.7685 | 0.5162 | 0.8004 | 0.6890 | 0.7543 |
| alma-7b-flap30-gen-pair-ru-lora | 0.8362 | 0.8242 | 0.8380 | 0.8227 | 0.7645 | 0.7587 | 0.6285 | 0.5689 | 0.8330 | 0.7443 | 0.7619 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap30-gen-pair-zh | 0.8254 | 0.8219 | 0.8365 | 0.8175 | 0.7597 | 0.6849 | 0.7436 | 0.5177 | 0.7464 | 0.7699 | 0.7524 |
| alma-7b-flap30-gen-pair-zh-lora | 0.8361 | 0.8239 | 0.8446 | 0.8220 | 0.7760 | 0.6994 | 0.7818 | 0.5737 | 0.7776 | 0.8258 | 0.7761 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap40-gen-multi | 0.8088 | 0.8068 | 0.8061 | 0.7902 | 0.7227 | 0.7156 | 0.7551 | 0.5718 | 0.7557 | 0.6588 | 0.7392 |
| alma-7b-flap40-gen-multi-lora | 0.8341 | 0.8197 | 0.8343 | 0.8230 | 0.7595 | 0.8255 | 0.8272 | 0.7663 | 0.8208 | 0.7997 | 0.8110 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap40-gen-pair-cs | 0.8015 | 0.8046 | 0.8004 | 0.7892 | 0.6998 | 0.7503 | 0.6833 | 0.4370 | 0.6850 | 0.4315 | 0.6883 |
| alma-7b-flap40-gen-pair-cs-lora | 0.8219 | 0.8176 | 0.8272 | 0.8102 | 0.7507 | 0.8321 | 0.7519 | 0.5136 | 0.7829 | 0.5922 | 0.7500 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap40-gen-pair-de | 0.8036 | 0.8041 | 0.7954 | 0.7784 | 0.6995 | 0.5844 | 0.7650 | 0.4388 | 0.6353 | 0.4301 | 0.6735 |
| alma-7b-flap40-gen-pair-de-lora | 0.8196 | 0.8189 | 0.8277 | 0.8066 | 0.7414 | 0.6793 | 0.8125 | 0.5206 | 0.7538 | 0.5607 | 0.7341 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap40-gen-pair-is | 0.7885 | 0.7997 | 0.7965 | 0.7823 | 0.7089 | 0.5665 | 0.6765 | 0.6569 | 0.5859 | 0.4301 | 0.6792 |
| alma-7b-flap40-gen-pair-is-lora | 0.8058 | 0.8120 | 0.8290 | 0.8014 | 0.7361 | 0.6521 | 0.7418 | 0.7800 | 0.7246 | 0.5435 | 0.7426 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap40-gen-pair-ru | 0.8010 | 0.7976 | 0.7860 | 0.7761 | 0.7072 | 0.6561 | 0.7178 | 0.4188 | 0.7606 | 0.5978 | 0.7019 |
| alma-7b-flap40-gen-pair-ru-lora | 0.8242 | 0.8129 | 0.8216 | 0.8092 | 0.7490 | 0.7358 | 0.7459 | 0.4779 | 0.8256 | 0.6827 | 0.7485 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap40-gen-pair-zh | 0.8036 | 0.8067 | 0.8095 | 0.7867 | 0.7370 | 0.5704 | 0.6634 | 0.4048 | 0.6854 | 0.6947 | 0.6962 |
| alma-7b-flap40-gen-pair-zh-lora | 0.8215 | 0.8204 | 0.8292 | 0.8118 | 0.7668 | 0.6357 | 0.7042 | 0.4564 | 0.7284 | 0.8139 | 0.7388 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-flap40-ref-multi | 0.8138 | 0.8063 | 0.8032 | 0.7899 | 0.7204 | 0.7172 | 0.7625 | 0.5826 | 0.7455 | 0.6513 | 0.7393 |
| alma-7b-flap40-ref-multi-lora | 0.8335 | 0.8195 | 0.8320 | 0.8153 | 0.7626 | 0.8192 | 0.8137 | 0.7826 | 0.8292 | 0.8048 | 0.8113 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt20-ref-multi | 0.8423 | 0.8336 | 0.8590 | 0.8360 | 0.7891 | 0.8519 | 0.8445 | 0.8278 | 0.8493 | 0.8399 | 0.8374 |
| alma-7b-slimgpt20-ref-multi-lora | 0.8443 | 0.8372 | 0.8508 | 0.8335 | 0.7884 | 0.8544 | 0.8456 | 0.8320 | 0.8538 | 0.8406 | 0.8381 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt30-gen-multi | 0.8378 | 0.8262 | 0.8496 | 0.8283 | 0.7817 | 0.8312 | 0.8332 | 0.8016 | 0.8268 | 0.8126 | 0.8229 |
| alma-7b-slimgpt30-gen-multi-lora | 0.8402 | 0.8345 | 0.8474 | 0.8341 | 0.7900 | 0.8478 | 0.8368 | 0.8200 | 0.8405 | 0.8328 | 0.8324 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt30-gen-pair-cs | 0.8391 | 0.8269 | 0.8500 | 0.8262 | 0.7673 | 0.8475 | 0.7986 | 0.7022 | 0.8091 | 0.5915 | 0.7858 |
| alma-7b-slimgpt30-gen-pair-cs-lora | 0.8437 | 0.8315 | 0.8506 | 0.8270 | 0.7774 | 0.8514 | 0.8039 | 0.7018 | 0.8135 | 0.6149 | 0.7916 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt30-gen-pair-de | 0.8358 | 0.8288 | 0.8485 | 0.8281 | 0.7657 | 0.7667 | 0.8391 | 0.6843 | 0.8040 | 0.5556 | 0.7757 |
| alma-7b-slimgpt30-gen-pair-de-lora | 0.8423 | 0.8308 | 0.8556 | 0.8301 | 0.7640 | 0.7661 | 0.8428 | 0.6830 | 0.8004 | 0.6184 | 0.7833 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt30-gen-pair-is | 0.8322 | 0.8301 | 0.8481 | 0.8193 | 0.7644 | 0.7693 | 0.8028 | 0.8150 | 0.7796 | 0.5636 | 0.7824 |
| alma-7b-slimgpt30-gen-pair-is-lora | 0.8360 | 0.8318 | 0.8489 | 0.8216 | 0.7691 | 0.7783 | 0.8020 | 0.8240 | 0.7968 | 0.5756 | 0.7884 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt30-gen-pair-ru | 0.8397 | 0.8244 | 0.8448 | 0.8254 | 0.7660 | 0.7824 | 0.7820 | 0.6345 | 0.8326 | 0.7089 | 0.7841 |
| alma-7b-slimgpt30-gen-pair-ru-lora | 0.8406 | 0.8302 | 0.8514 | 0.8344 | 0.7712 | 0.7864 | 0.7728 | 0.5764 | 0.8420 | 0.7382 | 0.7844 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt30-gen-pair-zh | 0.8248 | 0.8212 | 0.8348 | 0.8210 | 0.7763 | 0.6404 | 0.6978 | 0.4658 | 0.7476 | 0.8245 | 0.7454 |
| alma-7b-slimgpt30-gen-pair-zh-lora | 0.8320 | 0.8228 | 0.8363 | 0.8276 | 0.7836 | 0.6575 | 0.6844 | 0.5214 | 0.7445 | 0.8345 | 0.7545 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt40-gen-multi | 0.8205 | 0.8168 | 0.8341 | 0.8122 | 0.7503 | 0.7893 | 0.7971 | 0.7410 | 0.7905 | 0.7537 | 0.7905 |
| alma-7b-slimgpt40-gen-multi-lora | 0.8371 | 0.8254 | 0.8392 | 0.8236 | 0.7726 | 0.8185 | 0.8190 | 0.7938 | 0.8285 | 0.8216 | 0.8179 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt40-gen-pair-cs | 0.8304 | 0.8172 | 0.8290 | 0.8127 | 0.7510 | 0.8246 | 0.7374 | 0.5403 | 0.7778 | 0.4603 | 0.7381 |
| alma-7b-slimgpt40-gen-pair-cs-lora | 0.8338 | 0.8259 | 0.8403 | 0.8143 | 0.7586 | 0.8357 | 0.7653 | 0.5526 | 0.7902 | 0.5049 | 0.7522 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt40-gen-pair-de | 0.8226 | 0.8177 | 0.8339 | 0.8124 | 0.7337 | 0.6329 | 0.8174 | 0.5027 | 0.7037 | 0.4889 | 0.7166 |
| alma-7b-slimgpt40-gen-pair-de-lora | 0.8315 | 0.8254 | 0.8422 | 0.8178 | 0.7450 | 0.6384 | 0.8359 | 0.5982 | 0.6010 | 0.5886 | 0.7324 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt40-gen-pair-is | 0.8111 | 0.8173 | 0.8309 | 0.8001 | 0.7294 | 0.6293 | 0.7232 | 0.7931 | 0.6642 | 0.4660 | 0.7265 |
| alma-7b-slimgpt40-gen-pair-is-lora | 0.8210 | 0.8215 | 0.8392 | 0.8092 | 0.7462 | 0.6721 | 0.7371 | 0.8156 | 0.7175 | 0.4859 | 0.7465 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt40-gen-pair-ru | 0.8320 | 0.8170 | 0.8305 | 0.8194 | 0.7348 | 0.6573 | 0.6982 | 0.4459 | 0.8108 | 0.5706 | 0.7217 |
| alma-7b-slimgpt40-gen-pair-ru-lora | 0.8393 | 0.8211 | 0.8388 | 0.8272 | 0.7579 | 0.7055 | 0.7104 | 0.4696 | 0.8295 | 0.6628 | 0.7462 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt40-gen-pair-zh | 0.8055 | 0.8130 | 0.8061 | 0.7932 | 0.7600 | 0.5112 | 0.5975 | 0.4307 | 0.6718 | 0.7882 | 0.6977 |
| alma-7b-slimgpt40-gen-pair-zh-lora | 0.8130 | 0.8157 | 0.8165 | 0.8131 | 0.7734 | 0.5439 | 0.5806 | 0.5194 | 0.6196 | 0.8206 | 0.7116 |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8474 |
| alma-7b-slimgpt40-ref-multi | 0.8223 | 0.8152 | 0.8337 | 0.8167 | 0.7510 | 0.7899 | 0.7970 | 0.7479 | 0.7933 | 0.7469 | 0.7914 |
| alma-7b-slimgpt40-ref-multi-lora | 0.8345 | 0.8267 | 0.8429 | 0.8259 | 0.7733 | 0.8253 | 0.8293 | 0.7994 | 0.8235 | 0.8125 | 0.8193 |

## Structure

Removed parameters as a share of the whole model (embeddings included), from kept head and FFN channel counts. Found for 192/192 models; pair and dir cells show the mean and range over their models. Per-model figures are in structure.csv.

| Config | multi | pair | dir |
| :--- | ---: | ---: | ---: |
| SlimGPT ref 20% | 19.3% | 19.3% (19.3% to 19.3%, n=5) | 19.3% (19.3% to 19.3%, n=10) |
| SlimGPT ref 30% | 28.8% | 28.8% (28.8% to 28.8%, n=5) | 28.8% (28.8% to 28.8%, n=10) |
| SlimGPT ref 40% | 38.4% | 38.4% (38.4% to 38.4%, n=5) | 38.4% (38.4% to 38.4%, n=10) |
| SlimGPT gen 20% | 19.3% | 19.3% (19.3% to 19.3%, n=5) | 19.3% (19.3% to 19.3%, n=10) |
| SlimGPT gen 30% | 28.8% | 28.8% (28.8% to 28.8%, n=5) | 28.8% (28.8% to 28.8%, n=10) |
| SlimGPT gen 40% | 38.4% | 38.4% (38.4% to 38.4%, n=5) | 38.4% (38.4% to 38.4%, n=10) |
| FLAP ref 20% | 19.2% | 19.2% (19.2% to 19.2%, n=5) | 19.2% (19.2% to 19.2%, n=10) |
| FLAP ref 30% | 28.8% | 28.8% (28.8% to 28.8%, n=5) | 28.8% (28.8% to 28.8%, n=10) |
| FLAP ref 40% | 38.4% | 38.4% (38.4% to 38.4%, n=5) | 38.4% (38.4% to 38.5%, n=10) |
| FLAP gen 20% | 19.2% | 19.2% (19.2% to 19.2%, n=5) | 19.2% (19.2% to 19.2%, n=10) |
| FLAP gen 30% | 28.8% | 28.8% (28.8% to 28.8%, n=5) | 28.8% (28.8% to 28.8%, n=10) |
| FLAP gen 40% | 38.4% | 38.4% (38.4% to 38.4%, n=5) | 38.4% (38.4% to 38.5%, n=10) |

### Per-layer kept units, multi models

Heads are counts per layer (of 32); FFN is the kept share of 11008 channels. min / mean / max over layers, then mean of the first four / last four layers.

| System | Params removed | Heads min/mean/max | Heads first4/last4 | FFN min/mean/max | FFN first4/last4 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SlimGPT ref 20% | 19.3% | 24 / 25.5 / 30 | 28.8 / 24.0 | 75% / 80% / 95% | 90% / 75% |
| SlimGPT ref 30% | 28.8% | 20 / 22.4 / 30 | 27.5 / 20.0 | 62% / 70% / 92% | 85% / 62% |
| SlimGPT ref 40% | 38.4% | 16 / 19.2 / 29 | 26.0 / 16.0 | 49% / 60% / 90% | 81% / 50% |
| SlimGPT gen 20% | 19.3% | 24 / 25.5 / 30 | 28.8 / 24.0 | 75% / 80% / 95% | 90% / 75% |
| SlimGPT gen 30% | 28.8% | 20 / 22.4 / 30 | 27.5 / 20.0 | 62% / 70% / 92% | 85% / 62% |
| SlimGPT gen 40% | 38.4% | 16 / 19.2 / 29 | 26.0 / 16.0 | 49% / 60% / 90% | 81% / 50% |
| FLAP ref 20% | 19.2% | 14 / 31.1 / 32 | 32.0 / 32.0 | 43% / 71% / 100% | 99% / 87% |
| FLAP ref 30% | 28.8% | 9 / 28.3 / 32 | 32.0 / 22.8 | 36% / 61% / 100% | 91% / 85% |
| FLAP ref 40% | 38.4% | 7 / 23.2 / 32 | 30.5 / 12.2 | 31% / 54% / 100% | 87% / 80% |
| FLAP gen 20% | 19.2% | 13 / 31.1 / 32 | 32.0 / 32.0 | 42% / 71% / 100% | 99% / 87% |
| FLAP gen 30% | 28.8% | 9 / 28.3 / 32 | 32.0 / 23.2 | 36% / 61% / 100% | 91% / 85% |
| FLAP gen 40% | 38.4% | 7 / 23.2 / 32 | 30.5 / 13.0 | 31% / 54% / 100% | 87% / 80% |

## Warnings

None.
