# Pruning grid report

Suite `alma10-greedy`, generated 2026-10-08 20:23 UTC from `runs` and `configs/grid/manifest.json`. Paired bootstrap: 1000 resamples, seed 12345. Missing cells are `-`.

## Coverage

Grid models complete: **192/192** (BLEU, chrF++ and COMET on every direction the model is evaluated on: all 10 for multi models, the model's own pruned directions for pair and dir models; a `-lora` system follows its source); 0 partially scored; 0 without a run. LoRA-repaired systems: 27/27 complete.

Specialists scored on all 10 directions, which feed the transfer section: 0 pair and 10 dir models, of 180 specialists with a run.

MetricX-24 (`scores.metricx.json`): 192/192 grid models (192 on every in-scope direction), 27/27 LoRA systems, dense baseline yes.

Dense baseline `alma-7b` (run a3469aefba68, 10/10 directions): COMET 0.8475, BLEU 30.32, chrF++ 51.13, MetricX-24 2.937.

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
| dense | - | 0% | 0.8475 | 30.32 | 51.13 | 0.8475 | 30.32 | 51.13 | 0.8475 | 30.32 | 51.13 | - | - |
| SlimGPT | ref | 20% | 0.8361 | 27.76 | 49.10 | **0.8383** | **28.21** | **49.45** | 0.8382 | 28.10 | 49.38 | +0.0022 [+0.0014, +0.0031] p<0.001 | +0.0021 [+0.0013, +0.0029] p<0.001 |
| SlimGPT | ref | 30% | 0.8223 | 25.78 | 47.27 | 0.8285 | **26.60** | **48.07** | **0.8289** | 26.57 | 48.01 | +0.0062 [+0.0052, +0.0073] p<0.001 | +0.0066 [+0.0056, +0.0076] p<0.001 |
| SlimGPT | ref | 40% | 0.7931 | 23.24 | 44.77 | 0.8088 | 24.04 | 45.71 | **0.8097** | **24.23** | **45.81** | +0.0156 [+0.0143, +0.0171] p<0.001 | +0.0166 [+0.0151, +0.0180] p<0.001 |
| SlimGPT | gen | 20% | 0.8359 | 27.91 | 49.19 | **0.8387** | **28.27** | **49.59** | 0.8382 | 28.19 | 49.44 | +0.0028 [+0.0019, +0.0037] p<0.001 | +0.0023 [+0.0013, +0.0031] p<0.001 |
| SlimGPT | gen | 30% | 0.8227 | 25.94 | 47.53 | **0.8287** | 26.78 | **48.26** | 0.8282 | **26.81** | 48.17 | +0.0060 [+0.0050, +0.0070] p<0.001 | +0.0054 [+0.0044, +0.0065] p<0.001 |
| SlimGPT | gen | 40% | 0.7937 | 23.27 | 44.79 | 0.8089 | 24.33 | 45.83 | **0.8096** | **24.44** | **46.05** | +0.0152 [+0.0138, +0.0166] p<0.001 | +0.0160 [+0.0146, +0.0173] p<0.001 |
| FLAP | ref | 20% | 0.8224 | 26.74 | 47.82 | **0.8300** | 27.49 | 48.43 | **0.8300** | **27.70** | **48.71** | +0.0076 [+0.0065, +0.0087] p<0.001 | +0.0076 [+0.0065, +0.0086] p<0.001 |
| FLAP | ref | 30% | 0.7832 | 23.61 | 44.59 | 0.8017 | **24.50** | 45.53 | **0.8024** | 24.48 | **45.75** | +0.0186 [+0.0169, +0.0202] p<0.001 | +0.0193 [+0.0175, +0.0209] p<0.001 |
| FLAP | ref | 40% | 0.7414 | 20.31 | 41.00 | **0.7596** | 20.31 | 41.24 | 0.7592 | **20.45** | **41.45** | +0.0182 [+0.0164, +0.0202] p<0.001 | +0.0178 [+0.0160, +0.0198] p<0.001 |
| FLAP | gen | 20% | 0.8217 | 26.72 | 47.81 | 0.8292 | 27.44 | 48.41 | **0.8305** | **27.73** | **48.80** | +0.0075 [+0.0064, +0.0086] p<0.001 | +0.0089 [+0.0078, +0.0100] p<0.001 |
| FLAP | gen | 30% | 0.7823 | 23.55 | 44.56 | 0.8011 | 24.46 | 45.49 | **0.8014** | **24.52** | **45.70** | +0.0188 [+0.0171, +0.0205] p<0.001 | +0.0191 [+0.0174, +0.0206] p<0.001 |
| FLAP | gen | 40% | 0.7417 | 20.17 | 40.85 | **0.7580** | 20.34 | 41.31 | 0.7577 | **20.54** | **41.53** | +0.0163 [+0.0144, +0.0182] p<0.001 | +0.0160 [+0.0140, +0.0178] p<0.001 |

## Headline, MetricX-24 (lower is better)

MetricX-24 is an error score in [0, 25]: lower is better. Same scope composites and macro over directions as the COMET headline; bold marks the lowest (best) complete scope. The delta columns are specialist minus multi on macro MetricX, with the same stratified paired bootstrap 95% CI and two-sided p: **a negative delta means the specialist is better**. `-` means no MetricX scores yet; `(n/10)` means only n directions have them.

| Method | Calib | Sparsity | multi MetricX | pair MetricX | dir MetricX | pair - multi MetricX | dir - multi MetricX |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| dense | - | 0% | 2.937 | 2.937 | 2.937 | - | - |
| SlimGPT | ref | 20% | 3.356 | 3.287 | **3.265** | -0.069 [-0.105, -0.032] p<0.001 | -0.091 [-0.126, -0.052] p<0.001 |
| SlimGPT | ref | 30% | 3.809 | 3.608 | **3.562** | -0.202 [-0.241, -0.164] p<0.001 | -0.248 [-0.289, -0.206] p<0.001 |
| SlimGPT | ref | 40% | 4.857 | 4.324 | **4.246** | -0.533 [-0.587, -0.478] p<0.001 | -0.611 [-0.666, -0.554] p<0.001 |
| SlimGPT | gen | 20% | 3.345 | **3.258** | 3.274 | -0.086 [-0.118, -0.050] p<0.001 | -0.071 [-0.106, -0.034] p<0.001 |
| SlimGPT | gen | 30% | 3.815 | 3.614 | **3.578** | -0.201 [-0.243, -0.161] p<0.001 | -0.237 [-0.275, -0.195] p<0.001 |
| SlimGPT | gen | 40% | 4.866 | 4.303 | **4.278** | -0.563 [-0.618, -0.508] p<0.001 | -0.587 [-0.639, -0.533] p<0.001 |
| FLAP | ref | 20% | 3.851 | **3.604** | 3.605 | -0.247 [-0.289, -0.202] p<0.001 | -0.246 [-0.289, -0.200] p<0.001 |
| FLAP | ref | 30% | 5.222 | 4.612 | **4.591** | -0.610 [-0.668, -0.553] p<0.001 | -0.631 [-0.687, -0.570] p<0.001 |
| FLAP | ref | 40% | 6.697 | **6.154** | **6.154** | -0.543 [-0.608, -0.481] p<0.001 | -0.543 [-0.606, -0.479] p<0.001 |
| FLAP | gen | 20% | 3.884 | 3.640 | **3.593** | -0.244 [-0.284, -0.198] p<0.001 | -0.291 [-0.333, -0.244] p<0.001 |
| FLAP | gen | 30% | 5.231 | 4.641 | **4.604** | -0.590 [-0.648, -0.534] p<0.001 | -0.627 [-0.682, -0.568] p<0.001 |
| FLAP | gen | 40% | 6.665 | **6.175** | 6.179 | -0.490 [-0.556, -0.426] p<0.001 | -0.486 [-0.545, -0.421] p<0.001 |

## Ref vs gen calibration text

Calibrating on prompt + dense-generated continuation (gen) instead of prompt + reference (ref). Scores are the scope composites of the headline, over the directions both sides have, each direction weighted equally. COMET and MetricX-24 use a stratified paired bootstrap of the macro (segment positions resampled within each direction, the same for both sides); BLEU does the same, recomputing corpus BLEU per direction from sufficient statistics. MetricX-24 is lower-is-better; its delta is also gen minus ref, so a negative MetricX delta means gen is better.

| Method | Sparsity | Scope | COMET ref | COMET gen | gen - ref [95% CI] | p | BLEU ref | BLEU gen | gen - ref [95% CI] | p | MetricX ref | MetricX gen | MetricX gen - ref [95% CI] | p |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlimGPT | 20% | multi | 0.8361 | 0.8359 | -0.0002 [-0.0008, +0.0005] | 0.603 | 27.76 | 27.91 | +0.15 [-0.01, +0.28] | 0.039 | 3.356 | 3.345 | -0.011 [-0.040, +0.015] | 0.421 |
| SlimGPT | 20% | pair | 0.8383 | 0.8387 | +0.0004 [-0.0002, +0.0010] | 0.193 | 28.21 | 28.27 | +0.06 [-0.07, +0.19] | 0.407 | 3.287 | 3.258 | -0.028 [-0.055, -0.002] | 0.037 |
| SlimGPT | 20% | dir | 0.8382 | 0.8382 | +0.0000 [-0.0006, +0.0006] | 0.943 | 28.10 | 28.19 | +0.09 [-0.03, +0.22] | 0.176 | 3.265 | 3.274 | +0.009 [-0.018, +0.035] | 0.483 |
| SlimGPT | 30% | multi | 0.8223 | 0.8227 | +0.0004 [-0.0004, +0.0012] | 0.335 | 25.78 | 25.94 | +0.16 [+0.01, +0.31] | 0.037 | 3.809 | 3.815 | +0.006 [-0.027, +0.038] | 0.747 |
| SlimGPT | 30% | pair | 0.8285 | 0.8287 | +0.0002 [-0.0005, +0.0009] | 0.519 | 26.60 | 26.78 | +0.18 [+0.05, +0.31] | 0.011 | 3.608 | 3.614 | +0.006 [-0.024, +0.038] | 0.690 |
| SlimGPT | 30% | dir | 0.8289 | 0.8282 | -0.0007 [-0.0014, +0.0000] | 0.047 | 26.57 | 26.81 | +0.24 [+0.10, +0.38] | 0.002 | 3.562 | 3.578 | +0.017 [-0.014, +0.047] | 0.260 |
| SlimGPT | 40% | multi | 0.7931 | 0.7937 | +0.0005 [-0.0006, +0.0017] | 0.358 | 23.24 | 23.27 | +0.03 [-0.13, +0.18] | 0.684 | 4.857 | 4.866 | +0.009 [-0.034, +0.051] | 0.716 |
| SlimGPT | 40% | pair | 0.8088 | 0.8089 | +0.0001 [-0.0010, +0.0011] | 0.867 | 24.04 | 24.33 | +0.29 [+0.14, +0.43] | <0.001 | 4.324 | 4.303 | -0.021 [-0.059, +0.017] | 0.279 |
| SlimGPT | 40% | dir | 0.8097 | 0.8096 | -0.0001 [-0.0011, +0.0008] | 0.930 | 24.23 | 24.44 | +0.21 [+0.07, +0.37] | 0.008 | 4.246 | 4.278 | +0.032 [-0.008, +0.070] | 0.115 |
| FLAP | 20% | multi | 0.8224 | 0.8217 | -0.0008 [-0.0015, +0.0001] | 0.073 | 26.74 | 26.72 | -0.02 [-0.15, +0.11] | 0.724 | 3.851 | 3.884 | +0.033 [+0.001, +0.065] | 0.049 |
| FLAP | 20% | pair | 0.8300 | 0.8292 | -0.0008 [-0.0016, -0.0001] | 0.034 | 27.49 | 27.44 | -0.05 [-0.17, +0.08] | 0.439 | 3.604 | 3.640 | +0.035 [+0.008, +0.065] | 0.012 |
| FLAP | 20% | dir | 0.8300 | 0.8305 | +0.0005 [-0.0002, +0.0012] | 0.145 | 27.70 | 27.73 | +0.03 [-0.09, +0.16] | 0.600 | 3.605 | 3.593 | -0.012 [-0.038, +0.017] | 0.402 |
| FLAP | 30% | multi | 0.7832 | 0.7823 | -0.0009 [-0.0022, +0.0004] | 0.183 | 23.61 | 23.55 | -0.07 [-0.22, +0.10] | 0.445 | 5.222 | 5.231 | +0.009 [-0.031, +0.054] | 0.695 |
| FLAP | 30% | pair | 0.8017 | 0.8011 | -0.0007 [-0.0017, +0.0004] | 0.233 | 24.50 | 24.46 | -0.05 [-0.21, +0.12] | 0.560 | 4.612 | 4.641 | +0.029 [-0.008, +0.067] | 0.131 |
| FLAP | 30% | dir | 0.8024 | 0.8014 | -0.0011 [-0.0022, +0.0001] | 0.069 | 24.48 | 24.52 | +0.04 [-0.11, +0.18] | 0.591 | 4.591 | 4.604 | +0.014 [-0.027, +0.055] | 0.516 |
| FLAP | 40% | multi | 0.7414 | 0.7417 | +0.0003 [-0.0011, +0.0020] | 0.635 | 20.31 | 20.17 | -0.14 [-0.30, +0.05] | 0.131 | 6.697 | 6.665 | -0.032 [-0.082, +0.017] | 0.197 |
| FLAP | 40% | pair | 0.7596 | 0.7580 | -0.0015 [-0.0030, -0.0001] | 0.042 | 20.30 | 20.34 | +0.04 [-0.17, +0.21] | 0.740 | 6.154 | 6.175 | +0.021 [-0.028, +0.070] | 0.374 |
| FLAP | 40% | dir | 0.7592 | 0.7577 | -0.0015 [-0.0030, +0.0000] | 0.052 | 20.45 | 20.54 | +0.09 [-0.07, +0.26] | 0.291 | 6.154 | 6.179 | +0.025 [-0.019, +0.073] | 0.303 |

## SlimGPT vs FLAP

Deltas are SlimGPT minus FLAP at the same calibration text, sparsity and scope; positive means SlimGPT is better on COMET and BLEU. Same composites and tests as above. MetricX-24 is lower-is-better; its delta is also SlimGPT minus FLAP, so a negative MetricX delta means SlimGPT is better.

| Calib | Sparsity | Scope | COMET FLAP | COMET SlimGPT | SlimGPT - FLAP [95% CI] | p | BLEU FLAP | BLEU SlimGPT | SlimGPT - FLAP [95% CI] | p | MetricX FLAP | MetricX SlimGPT | MetricX SlimGPT - FLAP [95% CI] | p |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ref | 20% | multi | 0.8224 | 0.8361 | +0.0137 [+0.0125, +0.0148] | <0.001 | 26.74 | 27.76 | +1.02 [+0.81, +1.24] | <0.001 | 3.851 | 3.356 | -0.495 [-0.537, -0.452] | <0.001 |
| ref | 20% | pair | 0.8300 | 0.8383 | +0.0083 [+0.0072, +0.0093] | <0.001 | 27.49 | 28.21 | +0.73 [+0.53, +0.98] | <0.001 | 3.604 | 3.287 | -0.318 [-0.359, -0.277] | <0.001 |
| ref | 20% | dir | 0.8300 | 0.8382 | +0.0082 [+0.0072, +0.0092] | <0.001 | 27.70 | 28.10 | +0.40 [+0.17, +0.60] | <0.001 | 3.605 | 3.265 | -0.340 [-0.382, -0.300] | <0.001 |
| ref | 30% | multi | 0.7832 | 0.8223 | +0.0391 [+0.0375, +0.0407] | <0.001 | 23.61 | 25.78 | +2.17 [+1.91, +2.46] | <0.001 | 5.222 | 3.809 | -1.412 [-1.472, -1.351] | <0.001 |
| ref | 30% | pair | 0.8017 | 0.8285 | +0.0267 [+0.0252, +0.0282] | <0.001 | 24.50 | 26.60 | +2.10 [+1.86, +2.39] | <0.001 | 4.612 | 3.608 | -1.004 [-1.057, -0.954] | <0.001 |
| ref | 30% | dir | 0.8024 | 0.8289 | +0.0265 [+0.0252, +0.0279] | <0.001 | 24.48 | 26.57 | +2.09 [+1.85, +2.33] | <0.001 | 4.591 | 3.562 | -1.029 [-1.083, -0.976] | <0.001 |
| ref | 40% | multi | 0.7414 | 0.7931 | +0.0518 [+0.0498, +0.0538] | <0.001 | 20.31 | 23.24 | +2.93 [+2.69, +3.21] | <0.001 | 6.697 | 4.857 | -1.840 [-1.906, -1.777] | <0.001 |
| ref | 40% | pair | 0.7596 | 0.8088 | +0.0492 [+0.0476, +0.0511] | <0.001 | 20.30 | 24.04 | +3.74 [+3.49, +3.99] | <0.001 | 6.154 | 4.324 | -1.830 [-1.895, -1.770] | <0.001 |
| ref | 40% | dir | 0.7592 | 0.8097 | +0.0505 [+0.0488, +0.0523] | <0.001 | 20.45 | 24.23 | +3.79 [+3.53, +4.05] | <0.001 | 6.154 | 4.246 | -1.908 [-1.970, -1.846] | <0.001 |
| gen | 20% | multi | 0.8217 | 0.8359 | +0.0142 [+0.0131, +0.0153] | <0.001 | 26.72 | 27.91 | +1.20 [+0.97, +1.42] | <0.001 | 3.884 | 3.345 | -0.539 [-0.583, -0.496] | <0.001 |
| gen | 20% | pair | 0.8292 | 0.8387 | +0.0095 [+0.0085, +0.0106] | <0.001 | 27.44 | 28.27 | +0.83 [+0.62, +1.08] | <0.001 | 3.640 | 3.258 | -0.381 [-0.423, -0.340] | <0.001 |
| gen | 20% | dir | 0.8305 | 0.8382 | +0.0077 [+0.0066, +0.0086] | <0.001 | 27.73 | 28.19 | +0.46 [+0.23, +0.67] | <0.001 | 3.593 | 3.274 | -0.319 [-0.363, -0.279] | <0.001 |
| gen | 30% | multi | 0.7823 | 0.8227 | +0.0404 [+0.0388, +0.0421] | <0.001 | 23.55 | 25.94 | +2.39 [+2.13, +2.67] | <0.001 | 5.231 | 3.815 | -1.416 [-1.477, -1.362] | <0.001 |
| gen | 30% | pair | 0.8011 | 0.8287 | +0.0276 [+0.0262, +0.0290] | <0.001 | 24.46 | 26.78 | +2.33 [+2.11, +2.60] | <0.001 | 4.641 | 3.614 | -1.027 [-1.080, -0.978] | <0.001 |
| gen | 30% | dir | 0.8014 | 0.8282 | +0.0268 [+0.0255, +0.0283] | <0.001 | 24.52 | 26.81 | +2.28 [+2.05, +2.54] | <0.001 | 4.604 | 3.578 | -1.026 [-1.078, -0.976] | <0.001 |
| gen | 40% | multi | 0.7417 | 0.7937 | +0.0519 [+0.0501, +0.0539] | <0.001 | 20.17 | 23.27 | +3.10 [+2.81, +3.39] | <0.001 | 6.665 | 4.866 | -1.799 [-1.865, -1.733] | <0.001 |
| gen | 40% | pair | 0.7580 | 0.8089 | +0.0508 [+0.0492, +0.0526] | <0.001 | 20.34 | 24.33 | +3.99 [+3.76, +4.28] | <0.001 | 6.175 | 4.303 | -1.873 [-1.937, -1.807] | <0.001 |
| gen | 40% | dir | 0.7577 | 0.8096 | +0.0519 [+0.0501, +0.0538] | <0.001 | 20.54 | 24.44 | +3.91 [+3.66, +4.16] | <0.001 | 6.179 | 4.278 | -1.901 [-1.966, -1.840] | <0.001 |

## Per-direction scores, multi models

One row per multi model (plus dense). into-EN and out-of-EN are macros over the five directions on each side.

### COMET

| System | cs-en | de-en | is-en | ru-en | zh-en | into-EN | en-cs | en-de | en-is | en-ru | en-zh | out-of-EN | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8383 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8567 | 0.8475 |
| SlimGPT ref 20% | 0.8496 | 0.8331 | 0.8532 | 0.8363 | 0.7868 | 0.8318 | 0.8598 | 0.8333 | 0.8309 | 0.8481 | 0.8295 | 0.8403 | 0.8361 |
| SlimGPT ref 30% | 0.8427 | 0.8267 | 0.8470 | 0.8293 | 0.7766 | 0.8245 | 0.8431 | 0.8171 | 0.8080 | 0.8327 | 0.7999 | 0.8202 | 0.8223 |
| SlimGPT ref 40% | 0.8325 | 0.8134 | 0.8349 | 0.8155 | 0.7561 | 0.8105 | 0.7967 | 0.7889 | 0.7471 | 0.8015 | 0.7446 | 0.7758 | 0.7931 |
| SlimGPT gen 20% | 0.8493 | 0.8324 | 0.8545 | 0.8368 | 0.7870 | 0.8320 | 0.8611 | 0.8341 | 0.8287 | 0.8469 | 0.8283 | 0.8398 | 0.8359 |
| SlimGPT gen 30% | 0.8441 | 0.8265 | 0.8486 | 0.8295 | 0.7787 | 0.8255 | 0.8416 | 0.8186 | 0.8074 | 0.8318 | 0.8005 | 0.8200 | 0.8227 |
| SlimGPT gen 40% | 0.8311 | 0.8153 | 0.8353 | 0.8141 | 0.7541 | 0.8100 | 0.7974 | 0.7890 | 0.7509 | 0.8007 | 0.7488 | 0.7773 | 0.7937 |
| FLAP ref 20% | 0.8456 | 0.8307 | 0.8487 | 0.8327 | 0.7810 | 0.8277 | 0.8428 | 0.8160 | 0.7886 | 0.8344 | 0.8036 | 0.8171 | 0.8224 |
| FLAP ref 30% | 0.8325 | 0.8198 | 0.8294 | 0.8179 | 0.7546 | 0.8109 | 0.7908 | 0.7916 | 0.6790 | 0.7972 | 0.7188 | 0.7555 | 0.7832 |
| FLAP ref 40% | 0.8120 | 0.8016 | 0.8064 | 0.7978 | 0.7319 | 0.7899 | 0.7223 | 0.7585 | 0.5807 | 0.7520 | 0.6505 | 0.6928 | 0.7414 |
| FLAP gen 20% | 0.8452 | 0.8303 | 0.8497 | 0.8338 | 0.7787 | 0.8275 | 0.8383 | 0.8182 | 0.7841 | 0.8352 | 0.8031 | 0.8158 | 0.8217 |
| FLAP gen 30% | 0.8327 | 0.8192 | 0.8291 | 0.8185 | 0.7563 | 0.8112 | 0.7882 | 0.7914 | 0.6817 | 0.7937 | 0.7122 | 0.7534 | 0.7823 |
| FLAP gen 40% | 0.8116 | 0.8014 | 0.8053 | 0.7965 | 0.7288 | 0.7887 | 0.7263 | 0.7588 | 0.5822 | 0.7510 | 0.6554 | 0.6947 | 0.7417 |

### BLEU

| System | cs-en | de-en | is-en | ru-en | zh-en | into-EN | en-cs | en-de | en-is | en-ru | en-zh | out-of-EN | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (alma-7b) | 41.32 | 29.07 | 34.60 | 37.00 | 22.42 | 32.88 | 26.41 | 28.27 | 23.91 | 25.15 | 35.04 | 27.76 | 30.32 |
| SlimGPT ref 20% | 38.24 | 26.62 | 33.01 | 34.59 | 20.41 | 30.57 | 23.81 | 25.69 | 21.36 | 22.67 | 31.21 | 24.95 | 27.76 |
| SlimGPT ref 30% | 36.18 | 25.75 | 31.83 | 33.30 | 18.89 | 29.19 | 20.81 | 23.94 | 18.73 | 20.63 | 27.74 | 22.37 | 25.78 |
| SlimGPT ref 40% | 34.33 | 24.09 | 29.80 | 31.15 | 17.01 | 27.28 | 17.63 | 21.78 | 15.91 | 17.82 | 22.85 | 19.20 | 23.24 |
| SlimGPT gen 20% | 38.13 | 26.89 | 33.39 | 34.82 | 20.70 | 30.79 | 23.73 | 25.68 | 21.52 | 22.90 | 31.38 | 25.04 | 27.91 |
| SlimGPT gen 30% | 36.23 | 25.73 | 31.68 | 32.95 | 19.38 | 29.19 | 21.23 | 24.26 | 19.00 | 20.75 | 28.17 | 22.68 | 25.94 |
| SlimGPT gen 40% | 34.37 | 24.40 | 30.04 | 30.79 | 16.99 | 27.32 | 17.63 | 22.04 | 15.82 | 17.99 | 22.63 | 19.22 | 23.27 |
| FLAP ref 20% | 37.26 | 27.65 | 32.72 | 35.08 | 19.89 | 30.52 | 22.17 | 24.81 | 18.06 | 21.49 | 28.28 | 22.96 | 26.74 |
| FLAP ref 30% | 35.06 | 27.06 | 30.33 | 31.86 | 17.56 | 28.38 | 19.23 | 22.50 | 13.66 | 18.58 | 20.26 | 18.85 | 23.61 |
| FLAP ref 40% | 31.46 | 24.16 | 28.08 | 28.86 | 15.40 | 25.59 | 14.60 | 19.56 | 9.31 | 15.74 | 15.93 | 15.03 | 20.31 |
| FLAP gen 20% | 37.92 | 27.74 | 32.52 | 34.71 | 19.64 | 30.51 | 22.12 | 24.76 | 18.13 | 21.58 | 28.04 | 22.92 | 26.72 |
| FLAP gen 30% | 34.88 | 27.09 | 30.39 | 32.12 | 17.42 | 28.38 | 18.75 | 22.53 | 13.45 | 18.60 | 20.25 | 18.71 | 23.55 |
| FLAP gen 40% | 30.87 | 23.89 | 27.87 | 29.12 | 14.89 | 25.32 | 14.99 | 19.36 | 9.67 | 15.39 | 15.64 | 15.01 | 20.17 |

### MetricX-24 (lower is better)

| System | cs-en | de-en | is-en | ru-en | zh-en | into-EN | en-cs | en-de | en-is | en-ru | en-zh | out-of-EN | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (alma-7b) | 3.912 | 3.420 | 3.442 | 3.113 | 2.703 | 3.318 | 3.484 | 1.094 | 4.072 | 2.270 | 1.859 | 2.556 | 2.937 |
| SlimGPT ref 20% | 4.129 | 3.594 | 3.562 | 3.379 | 3.055 | 3.544 | 4.142 | 1.425 | 5.232 | 2.748 | 2.294 | 3.168 | 3.356 |
| SlimGPT ref 30% | 4.466 | 3.836 | 3.916 | 3.691 | 3.381 | 3.858 | 4.877 | 1.608 | 6.269 | 3.164 | 2.886 | 3.761 | 3.809 |
| SlimGPT ref 40% | 4.939 | 4.373 | 4.445 | 4.327 | 4.170 | 4.451 | 6.825 | 2.150 | 9.087 | 4.182 | 4.074 | 5.264 | 4.857 |
| SlimGPT gen 20% | 4.105 | 3.619 | 3.539 | 3.356 | 3.088 | 3.541 | 4.138 | 1.392 | 5.203 | 2.718 | 2.289 | 3.148 | 3.345 |
| SlimGPT gen 30% | 4.439 | 3.839 | 3.809 | 3.661 | 3.442 | 3.838 | 4.922 | 1.632 | 6.333 | 3.214 | 2.858 | 3.792 | 3.815 |
| SlimGPT gen 40% | 4.970 | 4.322 | 4.451 | 4.317 | 4.230 | 4.458 | 6.899 | 2.132 | 9.161 | 4.143 | 4.033 | 5.274 | 4.866 |
| FLAP ref 20% | 4.333 | 3.741 | 3.740 | 3.518 | 3.383 | 3.743 | 4.985 | 1.638 | 7.240 | 3.163 | 2.769 | 3.959 | 3.851 |
| FLAP ref 30% | 4.899 | 4.172 | 4.613 | 4.194 | 4.331 | 4.442 | 7.003 | 2.247 | 11.734 | 4.329 | 4.698 | 6.002 | 5.222 |
| FLAP ref 40% | 5.940 | 4.928 | 5.637 | 5.102 | 5.214 | 5.364 | 9.584 | 2.762 | 15.706 | 5.793 | 6.302 | 8.029 | 6.697 |
| FLAP gen 20% | 4.350 | 3.737 | 3.724 | 3.483 | 3.405 | 3.740 | 5.088 | 1.622 | 7.438 | 3.183 | 2.807 | 4.027 | 3.884 |
| FLAP gen 30% | 4.938 | 4.189 | 4.590 | 4.196 | 4.340 | 4.451 | 7.032 | 2.247 | 11.532 | 4.423 | 4.821 | 6.011 | 5.231 |
| FLAP gen 40% | 5.872 | 4.948 | 5.664 | 5.115 | 5.256 | 5.371 | 9.317 | 2.748 | 15.713 | 5.792 | 6.224 | 7.959 | 6.665 |

## Sparsity curve (macro COMET)

| Method | Calib | Scope | 0% (dense) | 20% | 30% | 40% |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| SlimGPT | ref | multi | 0.8475 | 0.8361 | 0.8223 | 0.7931 |
| SlimGPT | ref | pair | 0.8475 | 0.8383 | 0.8285 | 0.8088 |
| SlimGPT | ref | dir | 0.8475 | 0.8382 | 0.8289 | 0.8097 |
| SlimGPT | gen | multi | 0.8475 | 0.8359 | 0.8227 | 0.7937 |
| SlimGPT | gen | pair | 0.8475 | 0.8387 | 0.8287 | 0.8089 |
| SlimGPT | gen | dir | 0.8475 | 0.8382 | 0.8282 | 0.8096 |
| FLAP | ref | multi | 0.8475 | 0.8224 | 0.7832 | 0.7414 |
| FLAP | ref | pair | 0.8475 | 0.8300 | 0.8017 | 0.7596 |
| FLAP | ref | dir | 0.8475 | 0.8300 | 0.8024 | 0.7592 |
| FLAP | gen | multi | 0.8475 | 0.8217 | 0.7823 | 0.7417 |
| FLAP | gen | pair | 0.8475 | 0.8292 | 0.8011 | 0.7580 |
| FLAP | gen | dir | 0.8475 | 0.8305 | 0.8014 | 0.7577 |

## Sparsity curve (macro MetricX-24, lower is better)

| Method | Calib | Scope | 0% (dense) | 20% | 30% | 40% |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| SlimGPT | ref | multi | 2.937 | 3.356 | 3.809 | 4.857 |
| SlimGPT | ref | pair | 2.937 | 3.287 | 3.608 | 4.324 |
| SlimGPT | ref | dir | 2.937 | 3.265 | 3.562 | 4.246 |
| SlimGPT | gen | multi | 2.937 | 3.345 | 3.815 | 4.866 |
| SlimGPT | gen | pair | 2.937 | 3.258 | 3.614 | 4.303 |
| SlimGPT | gen | dir | 2.937 | 3.274 | 3.578 | 4.278 |
| FLAP | ref | multi | 2.937 | 3.851 | 5.222 | 6.697 |
| FLAP | ref | pair | 2.937 | 3.604 | 4.612 | 6.154 |
| FLAP | ref | dir | 2.937 | 3.605 | 4.591 | 6.154 |
| FLAP | gen | multi | 2.937 | 3.884 | 5.231 | 6.665 |
| FLAP | gen | pair | 2.937 | 3.640 | 4.641 | 6.175 |
| FLAP | gen | dir | 2.937 | 3.593 | 4.604 | 6.179 |

## Transfer

Only specialised models scored on all 10 directions enter: 0 pair and 10 dir models here, of 180 specialists with a run (the Models columns count them per config). Specialists evaluated only on their own directions, as on the full suite, are left out. Each cell is the mean COMET over the matrix cells in that category, with the mean of (specialist minus the multi model of the same config, on the same direction) in parentheses. Pair: own = the two directions of the model's language, other = the remaining eight. Dir: own = the calibration direction, reverse = its reverse, same target / same source = other directions sharing the target / source language (into-English models have same-target neighbours, out-of-English models same-source ones), other = the rest. Full matrices are in `transfer/<config>.md` and `.csv`.

### Pair models (5 x 10)

| Config | Models | Own pair | Other 8 | Own - other |
| :--- | ---: | ---: | ---: | ---: |
| SlimGPT ref 20% | 0/5 | - | - | - |
| SlimGPT ref 30% | 0/5 | - | - | - |
| SlimGPT ref 40% | 0/5 | - | - | - |
| SlimGPT gen 20% | 0/5 | - | - | - |
| SlimGPT gen 30% | 0/5 | - | - | - |
| SlimGPT gen 40% | 0/5 | - | - | - |
| FLAP ref 20% | 0/5 | - | - | - |
| FLAP ref 30% | 0/5 | - | - | - |
| FLAP ref 40% | 0/5 | - | - | - |
| FLAP gen 20% | 0/5 | - | - | - |
| FLAP gen 30% | 0/5 | - | - | - |
| FLAP gen 40% | 0/5 | - | - | - |

### Direction models (10 x 10)

| Config | Models | Own | Reverse | Same target | Same source | Other | Own - reverse |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlimGPT ref 20% | 0/10 | - | - | - | - | - | - |
| SlimGPT ref 30% | 0/10 | - | - | - | - | - | - |
| SlimGPT ref 40% | 0/10 | - | - | - | - | - | - |
| SlimGPT gen 20% | 0/10 | - | - | - | - | - | - |
| SlimGPT gen 30% | 0/10 | - | - | - | - | - | - |
| SlimGPT gen 40% | 10/10 | 0.8096 (+0.0160) | 0.7990 (+0.0053) | 0.8023 (-0.0077) | 0.5951 (-0.1823) | 0.6908 (-0.1029) | +0.0107 |
| FLAP ref 20% | 0/10 | - | - | - | - | - | - |
| FLAP ref 30% | 0/10 | - | - | - | - | - | - |
| FLAP ref 40% | 0/10 | - | - | - | - | - | - |
| FLAP gen 20% | 0/10 | - | - | - | - | - | - |
| FLAP gen 30% | 0/10 | - | - | - | - | - | - |
| FLAP gen 40% | 0/10 | - | - | - | - | - | - |

## Behaviour, multi models

Macro over directions of the per-direction rates (share of all segments). Hit budget = generation ran out of tokens without EOS; truncated = the hypothesis itself was cut off; repetition = looping flagged by post-processing; length ratio = hypothesis / reference characters. All systems are in long.csv.

| System | Off-target | Repetition | Hit budget | Truncated | Empty | Source copy | Length ratio |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (alma-7b) | 0.2% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.966 |
| SlimGPT ref 20% | 0.3% | 0.0% | 0.1% | 0.1% | 0.0% | 0.2% | 0.963 |
| SlimGPT ref 30% | 0.2% | 0.0% | 0.1% | 0.1% | 0.0% | 0.1% | 0.951 |
| SlimGPT ref 40% | 0.4% | 0.1% | 0.2% | 0.2% | 0.0% | 0.2% | 0.945 |
| SlimGPT gen 20% | 0.3% | 0.0% | 0.1% | 0.1% | 0.0% | 0.2% | 0.964 |
| SlimGPT gen 30% | 0.3% | 0.0% | 0.1% | 0.1% | 0.0% | 0.2% | 0.957 |
| SlimGPT gen 40% | 0.3% | 0.1% | 0.2% | 0.2% | 0.0% | 0.2% | 0.948 |
| FLAP ref 20% | 0.3% | 0.0% | 0.1% | 0.1% | 0.0% | 0.1% | 0.954 |
| FLAP ref 30% | 0.4% | 0.3% | 0.9% | 0.9% | 0.0% | 0.2% | 0.967 |
| FLAP ref 40% | 0.3% | 0.5% | 1.3% | 1.3% | 0.0% | 0.1% | 0.958 |
| FLAP gen 20% | 0.3% | 0.0% | 0.2% | 0.2% | 0.0% | 0.1% | 0.958 |
| FLAP gen 30% | 0.3% | 0.3% | 1.1% | 1.1% | 0.0% | 0.2% | 0.979 |
| FLAP gen 40% | 0.3% | 0.5% | 1.2% | 1.2% | 0.0% | 0.1% | 0.953 |

## Repair (LoRA)

Macro over the directions all three systems have (a specialist's repair is evaluated on its own directions on the full suite). Recovered = share of the pruning loss won back, (repaired - pruned) / (dense - pruned): 1 means the repair closed the whole gap to dense, 0 means nothing; the ratio is the same for a lower-is-better metric. The deltas are repaired minus pruned on the macro, stratified paired bootstrap; on MetricX-24 (lower is better) a negative delta means the repair helped.

| System | Metric | Dense | Pruned | Repaired | Repaired - pruned | Recovered |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| alma-7b-flap30-gen-multi-lora | COMET | 0.8475 | 0.7823 | 0.8239 | +0.0416 | 63.8% |
| alma-7b-flap30-gen-multi-lora | BLEU | 30.32 | 23.55 | 27.05 | +3.50 | 51.7% |
| alma-7b-flap30-gen-multi-lora | chrF++ | 51.13 | 44.56 | 48.09 | +3.53 | 53.8% |
| alma-7b-flap30-gen-multi-lora | MetricX-24 | 2.937 | 5.231 | 3.831 | -1.400 | 61.0% |
| alma-7b-flap30-gen-pair-cs-lora | COMET | 0.8675 | 0.8269 | 0.8462 | +0.0193 | 47.4% |
| alma-7b-flap30-gen-pair-cs-lora | BLEU | 33.87 | 27.81 | 31.01 | +3.19 | 52.7% |
| alma-7b-flap30-gen-pair-cs-lora | chrF++ | 56.30 | 50.56 | 53.57 | +3.01 | 52.3% |
| alma-7b-flap30-gen-pair-cs-lora | MetricX-24 | 3.698 | 5.428 | 4.543 | -0.886 | 51.2% |
| alma-7b-flap30-gen-pair-de-lora | COMET | 0.8431 | 0.8139 | 0.8293 | +0.0154 | 52.8% |
| alma-7b-flap30-gen-pair-de-lora | BLEU | 28.67 | 25.27 | 27.27 | +2.00 | 58.8% |
| alma-7b-flap30-gen-pair-de-lora | chrF++ | 53.39 | 49.82 | 51.77 | +1.95 | 54.5% |
| alma-7b-flap30-gen-pair-de-lora | MetricX-24 | 2.257 | 3.046 | 2.620 | -0.426 | 53.9% |
| alma-7b-flap30-gen-pair-is-lora | COMET | 0.8508 | 0.7908 | 0.8216 | +0.0308 | 51.4% |
| alma-7b-flap30-gen-pair-is-lora | BLEU | 29.26 | 22.75 | 23.80 | +1.04 | 16.1% |
| alma-7b-flap30-gen-pair-is-lora | chrF++ | 52.60 | 45.78 | 47.61 | +1.82 | 26.7% |
| alma-7b-flap30-gen-pair-is-lora | MetricX-24 | 3.757 | 6.697 | 5.364 | -1.334 | 45.4% |
| alma-7b-flap30-gen-pair-ru-lora | COMET | 0.8538 | 0.8079 | 0.8312 | +0.0233 | 50.8% |
| alma-7b-flap30-gen-pair-ru-lora | BLEU | 31.08 | 25.16 | 27.83 | +2.67 | 45.2% |
| alma-7b-flap30-gen-pair-ru-lora | chrF++ | 54.76 | 49.35 | 51.82 | +2.47 | 45.5% |
| alma-7b-flap30-gen-pair-ru-lora | MetricX-24 | 2.692 | 4.263 | 3.455 | -0.808 | 51.4% |
| alma-7b-flap30-gen-pair-zh-lora | COMET | 0.8222 | 0.7660 | 0.8021 | +0.0361 | 64.3% |
| alma-7b-flap30-gen-pair-zh-lora | BLEU | 28.73 | 21.30 | 26.20 | +4.91 | 66.0% |
| alma-7b-flap30-gen-pair-zh-lora | chrF++ | 38.57 | 31.93 | 36.90 | +4.98 | 74.9% |
| alma-7b-flap30-gen-pair-zh-lora | MetricX-24 | 2.281 | 3.771 | 2.809 | -0.962 | 64.6% |
| alma-7b-flap40-gen-multi-lora | COMET | 0.8475 | 0.7417 | 0.8113 | +0.0696 | 65.8% |
| alma-7b-flap40-gen-multi-lora | BLEU | 30.32 | 20.17 | 25.91 | +5.74 | 56.6% |
| alma-7b-flap40-gen-multi-lora | chrF++ | 51.13 | 40.85 | 47.03 | +6.18 | 60.1% |
| alma-7b-flap40-gen-multi-lora | MetricX-24 | 2.937 | 6.665 | 4.252 | -2.413 | 64.7% |
| alma-7b-flap40-gen-pair-cs-lora | COMET | 0.8675 | 0.7797 | 0.8312 | +0.0515 | 58.7% |
| alma-7b-flap40-gen-pair-cs-lora | BLEU | 33.87 | 23.20 | 28.68 | +5.48 | 51.4% |
| alma-7b-flap40-gen-pair-cs-lora | chrF++ | 56.30 | 45.78 | 51.78 | +6.00 | 57.0% |
| alma-7b-flap40-gen-pair-cs-lora | MetricX-24 | 3.698 | 7.246 | 5.156 | -2.091 | 58.9% |
| alma-7b-flap40-gen-pair-de-lora | COMET | 0.8431 | 0.7825 | 0.8176 | +0.0351 | 57.9% |
| alma-7b-flap40-gen-pair-de-lora | BLEU | 28.67 | 21.67 | 25.46 | +3.79 | 54.2% |
| alma-7b-flap40-gen-pair-de-lora | chrF++ | 53.39 | 46.24 | 50.28 | +4.03 | 56.4% |
| alma-7b-flap40-gen-pair-de-lora | MetricX-24 | 2.257 | 3.859 | 2.976 | -0.883 | 55.1% |
| alma-7b-flap40-gen-pair-is-lora | COMET | 0.8508 | 0.7268 | 0.8018 | +0.0751 | 60.5% |
| alma-7b-flap40-gen-pair-is-lora | BLEU | 29.26 | 18.10 | 21.56 | +3.45 | 31.0% |
| alma-7b-flap40-gen-pair-is-lora | chrF++ | 52.60 | 40.12 | 45.16 | +5.04 | 40.4% |
| alma-7b-flap40-gen-pair-is-lora | MetricX-24 | 3.757 | 9.505 | 6.246 | -3.259 | 56.7% |
| alma-7b-flap40-gen-pair-ru-lora | COMET | 0.8538 | 0.7773 | 0.8181 | +0.0408 | 53.4% |
| alma-7b-flap40-gen-pair-ru-lora | BLEU | 31.08 | 21.60 | 26.22 | +4.62 | 48.7% |
| alma-7b-flap40-gen-pair-ru-lora | chrF++ | 54.76 | 46.03 | 50.44 | +4.41 | 50.5% |
| alma-7b-flap40-gen-pair-ru-lora | MetricX-24 | 2.692 | 5.345 | 3.963 | -1.382 | 52.1% |
| alma-7b-flap40-gen-pair-zh-lora | COMET | 0.8222 | 0.7239 | 0.7904 | +0.0664 | 67.6% |
| alma-7b-flap40-gen-pair-zh-lora | BLEU | 28.73 | 17.13 | 24.97 | +7.84 | 67.6% |
| alma-7b-flap40-gen-pair-zh-lora | chrF++ | 38.57 | 28.35 | 35.77 | +7.42 | 72.6% |
| alma-7b-flap40-gen-pair-zh-lora | MetricX-24 | 2.281 | 4.923 | 3.101 | -1.822 | 69.0% |
| alma-7b-flap40-ref-multi-lora | COMET | 0.8475 | 0.7414 | 0.8111 | +0.0697 | 65.7% |
| alma-7b-flap40-ref-multi-lora | BLEU | 30.32 | 20.31 | 25.72 | +5.41 | 54.1% |
| alma-7b-flap40-ref-multi-lora | chrF++ | 51.13 | 41.00 | 46.90 | +5.90 | 58.3% |
| alma-7b-flap40-ref-multi-lora | MetricX-24 | 2.937 | 6.697 | 4.216 | -2.480 | 66.0% |
| alma-7b-slimgpt20-ref-multi-lora | COMET | 0.8475 | 0.8361 | 0.8391 | +0.0030 | 26.2% |
| alma-7b-slimgpt20-ref-multi-lora | BLEU | 30.32 | 27.76 | 28.51 | +0.75 | 29.2% |
| alma-7b-slimgpt20-ref-multi-lora | chrF++ | 51.13 | 49.10 | 49.55 | +0.45 | 22.2% |
| alma-7b-slimgpt20-ref-multi-lora | MetricX-24 | 2.937 | 3.356 | 3.234 | -0.122 | 29.2% |
| alma-7b-slimgpt30-gen-multi-lora | COMET | 0.8475 | 0.8227 | 0.8312 | +0.0085 | 34.3% |
| alma-7b-slimgpt30-gen-multi-lora | BLEU | 30.32 | 25.94 | 27.48 | +1.54 | 35.2% |
| alma-7b-slimgpt30-gen-multi-lora | chrF++ | 51.13 | 47.53 | 48.54 | +1.02 | 28.3% |
| alma-7b-slimgpt30-gen-multi-lora | MetricX-24 | 2.937 | 3.815 | 3.474 | -0.341 | 38.8% |
| alma-7b-slimgpt30-gen-pair-cs-lora | COMET | 0.8675 | 0.8500 | 0.8566 | +0.0065 | 37.3% |
| alma-7b-slimgpt30-gen-pair-cs-lora | BLEU | 33.87 | 29.44 | 31.70 | +2.26 | 51.0% |
| alma-7b-slimgpt30-gen-pair-cs-lora | chrF++ | 56.30 | 52.87 | 54.52 | +1.65 | 48.1% |
| alma-7b-slimgpt30-gen-pair-cs-lora | MetricX-24 | 3.698 | 4.357 | 4.082 | -0.274 | 41.6% |
| alma-7b-slimgpt30-gen-pair-de-lora | COMET | 0.8431 | 0.8298 | 0.8353 | +0.0055 | 41.1% |
| alma-7b-slimgpt30-gen-pair-de-lora | BLEU | 28.67 | 25.75 | 27.44 | +1.69 | 58.0% |
| alma-7b-slimgpt30-gen-pair-de-lora | chrF++ | 53.39 | 51.17 | 52.32 | +1.14 | 51.6% |
| alma-7b-slimgpt30-gen-pair-de-lora | MetricX-24 | 2.257 | 2.625 | 2.470 | -0.154 | 41.9% |
| alma-7b-slimgpt30-gen-pair-is-lora | COMET | 0.8508 | 0.8322 | 0.8365 | +0.0043 | 23.1% |
| alma-7b-slimgpt30-gen-pair-is-lora | BLEU | 29.26 | 26.03 | 25.47 | -0.57 | -17.6% |
| alma-7b-slimgpt30-gen-pair-is-lora | chrF++ | 52.60 | 49.84 | 49.27 | -0.57 | -20.5% |
| alma-7b-slimgpt30-gen-pair-is-lora | MetricX-24 | 3.757 | 4.757 | 4.590 | -0.167 | 16.7% |
| alma-7b-slimgpt30-gen-pair-ru-lora | COMET | 0.8538 | 0.8336 | 0.8395 | +0.0060 | 29.6% |
| alma-7b-slimgpt30-gen-pair-ru-lora | BLEU | 31.08 | 27.59 | 28.74 | +1.16 | 33.1% |
| alma-7b-slimgpt30-gen-pair-ru-lora | chrF++ | 54.76 | 51.95 | 53.00 | +1.05 | 37.3% |
| alma-7b-slimgpt30-gen-pair-ru-lora | MetricX-24 | 2.692 | 3.375 | 3.147 | -0.228 | 33.3% |
| alma-7b-slimgpt30-gen-pair-zh-lora | COMET | 0.8222 | 0.7980 | 0.8087 | +0.0106 | 44.1% |
| alma-7b-slimgpt30-gen-pair-zh-lora | BLEU | 28.73 | 25.11 | 26.86 | +1.75 | 48.4% |
| alma-7b-slimgpt30-gen-pair-zh-lora | chrF++ | 38.57 | 35.47 | 37.02 | +1.55 | 50.1% |
| alma-7b-slimgpt30-gen-pair-zh-lora | MetricX-24 | 2.281 | 2.956 | 2.566 | -0.390 | 57.8% |
| alma-7b-slimgpt40-gen-multi-lora | COMET | 0.8475 | 0.7937 | 0.8196 | +0.0259 | 48.2% |
| alma-7b-slimgpt40-gen-multi-lora | BLEU | 30.32 | 23.27 | 26.16 | +2.89 | 41.0% |
| alma-7b-slimgpt40-gen-multi-lora | chrF++ | 51.13 | 44.79 | 47.27 | +2.49 | 39.2% |
| alma-7b-slimgpt40-gen-multi-lora | MetricX-24 | 2.937 | 4.866 | 3.915 | -0.950 | 49.3% |
| alma-7b-slimgpt40-gen-pair-cs-lora | COMET | 0.8675 | 0.8310 | 0.8447 | +0.0137 | 37.5% |
| alma-7b-slimgpt40-gen-pair-cs-lora | BLEU | 33.87 | 27.45 | 29.77 | +2.31 | 36.0% |
| alma-7b-slimgpt40-gen-pair-cs-lora | chrF++ | 56.30 | 50.64 | 52.71 | +2.07 | 36.5% |
| alma-7b-slimgpt40-gen-pair-cs-lora | MetricX-24 | 3.698 | 5.089 | 4.519 | -0.570 | 41.0% |
| alma-7b-slimgpt40-gen-pair-de-lora | COMET | 0.8431 | 0.8153 | 0.8287 | +0.0134 | 48.2% |
| alma-7b-slimgpt40-gen-pair-de-lora | BLEU | 28.67 | 24.21 | 26.24 | +2.03 | 45.4% |
| alma-7b-slimgpt40-gen-pair-de-lora | chrF++ | 53.39 | 49.57 | 51.32 | +1.75 | 45.7% |
| alma-7b-slimgpt40-gen-pair-de-lora | MetricX-24 | 2.257 | 2.977 | 2.686 | -0.291 | 40.4% |
| alma-7b-slimgpt40-gen-pair-is-lora | COMET | 0.8508 | 0.8108 | 0.8252 | +0.0144 | 36.0% |
| alma-7b-slimgpt40-gen-pair-is-lora | BLEU | 29.26 | 22.65 | 23.53 | +0.88 | 13.3% |
| alma-7b-slimgpt40-gen-pair-is-lora | chrF++ | 52.60 | 46.45 | 47.59 | +1.14 | 18.6% |
| alma-7b-slimgpt40-gen-pair-is-lora | MetricX-24 | 3.757 | 5.863 | 5.251 | -0.613 | 29.1% |
| alma-7b-slimgpt40-gen-pair-ru-lora | COMET | 0.8538 | 0.8153 | 0.8306 | +0.0153 | 39.8% |
| alma-7b-slimgpt40-gen-pair-ru-lora | BLEU | 31.08 | 25.42 | 27.13 | +1.71 | 30.2% |
| alma-7b-slimgpt40-gen-pair-ru-lora | chrF++ | 54.76 | 49.74 | 51.71 | +1.97 | 39.3% |
| alma-7b-slimgpt40-gen-pair-ru-lora | MetricX-24 | 2.692 | 3.964 | 3.490 | -0.474 | 37.2% |
| alma-7b-slimgpt40-gen-pair-zh-lora | COMET | 0.8222 | 0.7718 | 0.7948 | +0.0229 | 45.6% |
| alma-7b-slimgpt40-gen-pair-zh-lora | BLEU | 28.73 | 21.92 | 25.01 | +3.09 | 45.4% |
| alma-7b-slimgpt40-gen-pair-zh-lora | chrF++ | 38.57 | 32.75 | 35.56 | +2.81 | 48.2% |
| alma-7b-slimgpt40-gen-pair-zh-lora | MetricX-24 | 2.281 | 3.621 | 2.971 | -0.650 | 48.5% |
| alma-7b-slimgpt40-ref-multi-lora | COMET | 0.8475 | 0.7931 | 0.8195 | +0.0263 | 48.4% |
| alma-7b-slimgpt40-ref-multi-lora | BLEU | 30.32 | 23.24 | 26.10 | +2.86 | 40.4% |
| alma-7b-slimgpt40-ref-multi-lora | chrF++ | 51.13 | 44.77 | 47.36 | +2.59 | 40.8% |
| alma-7b-slimgpt40-ref-multi-lora | MetricX-24 | 2.937 | 4.857 | 3.916 | -0.942 | 49.0% |

Repaired - pruned COMET with 95% CI:

- alma-7b-flap30-gen-multi-lora: +0.0416 [+0.0400, +0.0432] p<0.001
- alma-7b-flap30-gen-pair-cs-lora: +0.0193 [+0.0166, +0.0219] p<0.001
- alma-7b-flap30-gen-pair-de-lora: +0.0154 [+0.0134, +0.0176] p<0.001
- alma-7b-flap30-gen-pair-is-lora: +0.0308 [+0.0266, +0.0353] p<0.001
- alma-7b-flap30-gen-pair-ru-lora: +0.0233 [+0.0209, +0.0259] p<0.001
- alma-7b-flap30-gen-pair-zh-lora: +0.0361 [+0.0331, +0.0395] p<0.001
- alma-7b-flap40-gen-multi-lora: +0.0696 [+0.0677, +0.0714] p<0.001
- alma-7b-flap40-gen-pair-cs-lora: +0.0515 [+0.0479, +0.0553] p<0.001
- alma-7b-flap40-gen-pair-de-lora: +0.0351 [+0.0324, +0.0381] p<0.001
- alma-7b-flap40-gen-pair-is-lora: +0.0751 [+0.0704, +0.0801] p<0.001
- alma-7b-flap40-gen-pair-ru-lora: +0.0408 [+0.0376, +0.0440] p<0.001
- alma-7b-flap40-gen-pair-zh-lora: +0.0664 [+0.0631, +0.0700] p<0.001
- alma-7b-flap40-ref-multi-lora: +0.0697 [+0.0679, +0.0717] p<0.001
- alma-7b-slimgpt20-ref-multi-lora: +0.0030 [+0.0020, +0.0039] p<0.001
- alma-7b-slimgpt30-gen-multi-lora: +0.0085 [+0.0074, +0.0095] p<0.001
- alma-7b-slimgpt30-gen-pair-cs-lora: +0.0065 [+0.0043, +0.0088] p<0.001
- alma-7b-slimgpt30-gen-pair-de-lora: +0.0055 [+0.0039, +0.0069] p<0.001
- alma-7b-slimgpt30-gen-pair-is-lora: +0.0043 [+0.0019, +0.0068] p=0.002
- alma-7b-slimgpt30-gen-pair-ru-lora: +0.0060 [+0.0043, +0.0079] p<0.001
- alma-7b-slimgpt30-gen-pair-zh-lora: +0.0106 [+0.0086, +0.0128] p<0.001
- alma-7b-slimgpt40-gen-multi-lora: +0.0259 [+0.0245, +0.0273] p<0.001
- alma-7b-slimgpt40-gen-pair-cs-lora: +0.0137 [+0.0112, +0.0161] p<0.001
- alma-7b-slimgpt40-gen-pair-de-lora: +0.0134 [+0.0114, +0.0153] p<0.001
- alma-7b-slimgpt40-gen-pair-is-lora: +0.0144 [+0.0115, +0.0172] p<0.001
- alma-7b-slimgpt40-gen-pair-ru-lora: +0.0153 [+0.0131, +0.0177] p<0.001
- alma-7b-slimgpt40-gen-pair-zh-lora: +0.0229 [+0.0202, +0.0258] p<0.001
- alma-7b-slimgpt40-ref-multi-lora: +0.0263 [+0.0250, +0.0278] p<0.001

Repaired - pruned MetricX-24 with 95% CI (negative = the repair helped):

- alma-7b-flap30-gen-multi-lora: -1.400 [-1.460, -1.342] p<0.001
- alma-7b-flap30-gen-pair-cs-lora: -0.886 [-0.997, -0.771] p<0.001
- alma-7b-flap30-gen-pair-de-lora: -0.426 [-0.491, -0.360] p<0.001
- alma-7b-flap30-gen-pair-is-lora: -1.334 [-1.521, -1.165] p<0.001
- alma-7b-flap30-gen-pair-ru-lora: -0.808 [-0.888, -0.725] p<0.001
- alma-7b-flap30-gen-pair-zh-lora: -0.962 [-1.046, -0.881] p<0.001
- alma-7b-flap40-gen-multi-lora: -2.413 [-2.477, -2.348] p<0.001
- alma-7b-flap40-gen-pair-cs-lora: -2.091 [-2.248, -1.944] p<0.001
- alma-7b-flap40-gen-pair-de-lora: -0.883 [-0.976, -0.799] p<0.001
- alma-7b-flap40-gen-pair-is-lora: -3.259 [-3.484, -3.052] p<0.001
- alma-7b-flap40-gen-pair-ru-lora: -1.382 [-1.489, -1.271] p<0.001
- alma-7b-flap40-gen-pair-zh-lora: -1.822 [-1.917, -1.732] p<0.001
- alma-7b-flap40-ref-multi-lora: -2.480 [-2.551, -2.420] p<0.001
- alma-7b-slimgpt20-ref-multi-lora: -0.122 [-0.161, -0.084] p<0.001
- alma-7b-slimgpt30-gen-multi-lora: -0.341 [-0.384, -0.298] p<0.001
- alma-7b-slimgpt30-gen-pair-cs-lora: -0.274 [-0.369, -0.179] p<0.001
- alma-7b-slimgpt30-gen-pair-de-lora: -0.154 [-0.200, -0.105] p<0.001
- alma-7b-slimgpt30-gen-pair-is-lora: -0.167 [-0.290, -0.041] p=0.016
- alma-7b-slimgpt30-gen-pair-ru-lora: -0.228 [-0.295, -0.164] p<0.001
- alma-7b-slimgpt30-gen-pair-zh-lora: -0.390 [-0.446, -0.334] p<0.001
- alma-7b-slimgpt40-gen-multi-lora: -0.950 [-1.008, -0.899] p<0.001
- alma-7b-slimgpt40-gen-pair-cs-lora: -0.570 [-0.675, -0.470] p<0.001
- alma-7b-slimgpt40-gen-pair-de-lora: -0.291 [-0.346, -0.239] p<0.001
- alma-7b-slimgpt40-gen-pair-is-lora: -0.613 [-0.768, -0.451] p<0.001
- alma-7b-slimgpt40-gen-pair-ru-lora: -0.474 [-0.558, -0.390] p<0.001
- alma-7b-slimgpt40-gen-pair-zh-lora: -0.650 [-0.722, -0.577] p<0.001
- alma-7b-slimgpt40-ref-multi-lora: -0.942 [-0.999, -0.889] p<0.001

### Per-direction COMET

The macro column averages the directions all three systems have (each system's own while they share none yet).

| System | cs-en | de-en | is-en | ru-en | zh-en | en-cs | en-de | en-is | en-ru | en-zh | macro |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8475 |
| alma-7b-flap30-gen-multi | 0.8327 | 0.8192 | 0.8291 | 0.8185 | 0.7563 | 0.7882 | 0.7914 | 0.6817 | 0.7937 | 0.7122 | 0.7823 |
| alma-7b-flap30-gen-multi-lora | 0.8414 | 0.8281 | 0.8367 | 0.8308 | 0.7769 | 0.8450 | 0.8221 | 0.7977 | 0.8401 | 0.8202 | 0.8239 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8675 |
| alma-7b-flap30-gen-pair-cs | 0.8327 | - | - | - | - | 0.8212 | - | - | - | - | 0.8269 |
| alma-7b-flap30-gen-pair-cs-lora | 0.8421 | - | - | - | - | 0.8503 | - | - | - | - | 0.8462 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8431 |
| alma-7b-flap30-gen-pair-de | - | 0.8207 | - | - | - | - | 0.8070 | - | - | - | 0.8139 |
| alma-7b-flap30-gen-pair-de-lora | - | 0.8297 | - | - | - | - | 0.8290 | - | - | - | 0.8293 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8508 |
| alma-7b-flap30-gen-pair-is | - | - | 0.8287 | - | - | - | - | 0.7529 | - | - | 0.7908 |
| alma-7b-flap30-gen-pair-is-lora | - | - | 0.8399 | - | - | - | - | 0.8034 | - | - | 0.8216 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8538 |
| alma-7b-flap30-gen-pair-ru | - | - | - | 0.8145 | - | - | - | - | 0.8013 | - | 0.8079 |
| alma-7b-flap30-gen-pair-ru-lora | - | - | - | 0.8261 | - | - | - | - | 0.8363 | - | 0.8312 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8222 |
| alma-7b-flap30-gen-pair-zh | - | - | - | - | 0.7612 | - | - | - | - | 0.7708 | 0.7660 |
| alma-7b-flap30-gen-pair-zh-lora | - | - | - | - | 0.7811 | - | - | - | - | 0.8232 | 0.8021 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8475 |
| alma-7b-flap40-gen-multi | 0.8116 | 0.8014 | 0.8053 | 0.7965 | 0.7288 | 0.7263 | 0.7588 | 0.5822 | 0.7510 | 0.6554 | 0.7417 |
| alma-7b-flap40-gen-multi-lora | 0.8348 | 0.8208 | 0.8296 | 0.8236 | 0.7664 | 0.8318 | 0.8123 | 0.7706 | 0.8249 | 0.7979 | 0.8113 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8675 |
| alma-7b-flap40-gen-pair-cs | 0.8086 | - | - | - | - | 0.7508 | - | - | - | - | 0.7797 |
| alma-7b-flap40-gen-pair-cs-lora | 0.8274 | - | - | - | - | 0.8351 | - | - | - | - | 0.8312 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8431 |
| alma-7b-flap40-gen-pair-de | - | 0.7995 | - | - | - | - | 0.7655 | - | - | - | 0.7825 |
| alma-7b-flap40-gen-pair-de-lora | - | 0.8210 | - | - | - | - | 0.8142 | - | - | - | 0.8176 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8508 |
| alma-7b-flap40-gen-pair-is | - | - | 0.7973 | - | - | - | - | 0.6562 | - | - | 0.7268 |
| alma-7b-flap40-gen-pair-is-lora | - | - | 0.8244 | - | - | - | - | 0.7793 | - | - | 0.8018 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8538 |
| alma-7b-flap40-gen-pair-ru | - | - | - | 0.7881 | - | - | - | - | 0.7666 | - | 0.7773 |
| alma-7b-flap40-gen-pair-ru-lora | - | - | - | 0.8139 | - | - | - | - | 0.8223 | - | 0.8181 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8222 |
| alma-7b-flap40-gen-pair-zh | - | - | - | - | 0.7399 | - | - | - | - | 0.7080 | 0.7239 |
| alma-7b-flap40-gen-pair-zh-lora | - | - | - | - | 0.7695 | - | - | - | - | 0.8112 | 0.7904 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8475 |
| alma-7b-flap40-ref-multi | 0.8120 | 0.8016 | 0.8064 | 0.7978 | 0.7319 | 0.7223 | 0.7585 | 0.5807 | 0.7520 | 0.6505 | 0.7414 |
| alma-7b-flap40-ref-multi-lora | 0.8346 | 0.8211 | 0.8291 | 0.8217 | 0.7678 | 0.8239 | 0.8084 | 0.7750 | 0.8245 | 0.8047 | 0.8111 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8475 |
| alma-7b-slimgpt20-ref-multi | 0.8496 | 0.8331 | 0.8532 | 0.8363 | 0.7868 | 0.8598 | 0.8333 | 0.8309 | 0.8481 | 0.8295 | 0.8361 |
| alma-7b-slimgpt20-ref-multi-lora | 0.8490 | 0.8354 | 0.8447 | 0.8385 | 0.7907 | 0.8650 | 0.8390 | 0.8341 | 0.8558 | 0.8383 | 0.8391 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8475 |
| alma-7b-slimgpt30-gen-multi | 0.8441 | 0.8265 | 0.8486 | 0.8295 | 0.7787 | 0.8416 | 0.8186 | 0.8074 | 0.8318 | 0.8005 | 0.8227 |
| alma-7b-slimgpt30-gen-multi-lora | 0.8445 | 0.8301 | 0.8416 | 0.8342 | 0.7834 | 0.8566 | 0.8278 | 0.8204 | 0.8457 | 0.8280 | 0.8312 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8675 |
| alma-7b-slimgpt30-gen-pair-cs | 0.8437 | - | - | - | - | 0.8563 | - | - | - | - | 0.8500 |
| alma-7b-slimgpt30-gen-pair-cs-lora | 0.8502 | - | - | - | - | 0.8630 | - | - | - | - | 0.8566 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8431 |
| alma-7b-slimgpt30-gen-pair-de | - | 0.8292 | - | - | - | - | 0.8304 | - | - | - | 0.8298 |
| alma-7b-slimgpt30-gen-pair-de-lora | - | 0.8333 | - | - | - | - | 0.8373 | - | - | - | 0.8353 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8508 |
| alma-7b-slimgpt30-gen-pair-is | - | - | 0.8438 | - | - | - | - | 0.8206 | - | - | 0.8322 |
| alma-7b-slimgpt30-gen-pair-is-lora | - | - | 0.8452 | - | - | - | - | 0.8278 | - | - | 0.8365 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8538 |
| alma-7b-slimgpt30-gen-pair-ru | - | - | - | 0.8292 | - | - | - | - | 0.8379 | - | 0.8336 |
| alma-7b-slimgpt30-gen-pair-ru-lora | - | - | - | 0.8331 | - | - | - | - | 0.8459 | - | 0.8395 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8222 |
| alma-7b-slimgpt30-gen-pair-zh | - | - | - | - | 0.7773 | - | - | - | - | 0.8187 | 0.7980 |
| alma-7b-slimgpt30-gen-pair-zh-lora | - | - | - | - | 0.7856 | - | - | - | - | 0.8317 | 0.8087 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8475 |
| alma-7b-slimgpt40-gen-multi | 0.8311 | 0.8153 | 0.8353 | 0.8141 | 0.7541 | 0.7974 | 0.7890 | 0.7509 | 0.8007 | 0.7488 | 0.7937 |
| alma-7b-slimgpt40-gen-multi-lora | 0.8394 | 0.8257 | 0.8364 | 0.8278 | 0.7749 | 0.8324 | 0.8183 | 0.7971 | 0.8347 | 0.8092 | 0.8196 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8675 |
| alma-7b-slimgpt40-gen-pair-cs | 0.8331 | - | - | - | - | 0.8290 | - | - | - | - | 0.8310 |
| alma-7b-slimgpt40-gen-pair-cs-lora | 0.8394 | - | - | - | - | 0.8501 | - | - | - | - | 0.8447 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8431 |
| alma-7b-slimgpt40-gen-pair-de | - | 0.8201 | - | - | - | - | 0.8106 | - | - | - | 0.8153 |
| alma-7b-slimgpt40-gen-pair-de-lora | - | 0.8291 | - | - | - | - | 0.8283 | - | - | - | 0.8287 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8508 |
| alma-7b-slimgpt40-gen-pair-is | - | - | 0.8286 | - | - | - | - | 0.7930 | - | - | 0.8108 |
| alma-7b-slimgpt40-gen-pair-is-lora | - | - | 0.8367 | - | - | - | - | 0.8137 | - | - | 0.8252 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8538 |
| alma-7b-slimgpt40-gen-pair-ru | - | - | - | 0.8156 | - | - | - | - | 0.8150 | - | 0.8153 |
| alma-7b-slimgpt40-gen-pair-ru-lora | - | - | - | 0.8260 | - | - | - | - | 0.8353 | - | 0.8306 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8222 |
| alma-7b-slimgpt40-gen-pair-zh | - | - | - | - | 0.7573 | - | - | - | - | 0.7863 | 0.7718 |
| alma-7b-slimgpt40-gen-pair-zh-lora | - | - | - | - | 0.7723 | - | - | - | - | 0.8173 | 0.7948 |
| dense (alma-7b) | 0.8558 | 0.8375 | 0.8567 | 0.8445 | 0.7967 | 0.8792 | 0.8487 | 0.8448 | 0.8630 | 0.8477 | 0.8475 |
| alma-7b-slimgpt40-ref-multi | 0.8325 | 0.8134 | 0.8349 | 0.8155 | 0.7561 | 0.7967 | 0.7889 | 0.7471 | 0.8015 | 0.7446 | 0.7931 |
| alma-7b-slimgpt40-ref-multi-lora | 0.8393 | 0.8267 | 0.8371 | 0.8271 | 0.7774 | 0.8337 | 0.8170 | 0.7961 | 0.8313 | 0.8088 | 0.8195 |

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
