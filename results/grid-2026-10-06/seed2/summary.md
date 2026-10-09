# Pruning grid report

Suite `alma10-greedy-300`, generated 2026-10-06 19:36 UTC from `runs` and `configs/grid/manifest-seed2.json`. Paired bootstrap: 1000 resamples, seed 12345. Missing cells are `-`.

## Coverage

Grid models complete: **48/48** (BLEU, chrF++ and COMET on all 10 directions); 0 partially scored; 0 without a run. LoRA-repaired systems: 0/0 complete.

Dense baseline `alma-7b` (run 63d55369fea1, 10/10 directions): COMET 0.8474, BLEU 30.35, chrF++ 51.14.

| Config | multi | pair | dir |
| :--- | ---: | ---: | ---: |
| SlimGPT ref 20% | 1/1 | 5/5 | 10/10 |
| SlimGPT ref 30% | 1/1 | 5/5 | 10/10 |
| SlimGPT ref 40% | 1/1 | 5/5 | 10/10 |
| SlimGPT gen 20% | - | - | - |
| SlimGPT gen 30% | - | - | - |
| SlimGPT gen 40% | - | - | - |
| FLAP ref 20% | - | - | - |
| FLAP ref 30% | - | - | - |
| FLAP ref 40% | - | - | - |
| FLAP gen 20% | - | - | - |
| FLAP gen 30% | - | - | - |
| FLAP gen 40% | - | - | - |

## Headline: one multi model vs matched specialists

Macro over the 10 directions. For pair and dir, each direction is translated by the matching specialised model: `pair-<l>` for both directions of language l, `dir-<d>` for direction d. Bold marks the best complete scope per metric. The delta columns pool the segments of the shared directions and pair each specialist segment with multi's same segment (paired bootstrap 95% CI and two-sided p). `(n/10)` means only n directions exist yet.

| Method | Calib | Sparsity | multi COMET | multi BLEU | multi chrF++ | pair COMET | pair BLEU | pair chrF++ | dir COMET | dir BLEU | dir chrF++ | pair - multi COMET | dir - multi COMET |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | - | 0% | 0.8474 | 30.35 | 51.14 | 0.8474 | 30.35 | 51.14 | 0.8474 | 30.35 | 51.14 | - | - |
| SlimGPT | ref | 20% | 0.8360 | 27.73 | 49.07 | **0.8380** | 27.87 | **49.22** | 0.8363 | **27.93** | 49.21 | +0.0020 [-0.0001, +0.0039] p=0.056 | +0.0003 [-0.0018, +0.0024] p=0.755 |
| SlimGPT | ref | 30% | 0.8223 | 25.60 | 47.01 | 0.8268 | 26.08 | 47.69 | **0.8282** | **26.12** | **47.74** | +0.0044 [+0.0019, +0.0069] p=0.002 | +0.0059 [+0.0034, +0.0086] p<0.001 |
| SlimGPT | ref | 40% | 0.7954 | 23.24 | 44.79 | 0.8067 | 24.01 | 45.79 | **0.8094** | **24.12** | **45.83** | +0.0113 [+0.0079, +0.0143] p<0.001 | +0.0140 [+0.0108, +0.0174] p<0.001 |
| SlimGPT | gen | 20% | - | - | - | - | - | - | - | - | - | - | - |
| SlimGPT | gen | 30% | - | - | - | - | - | - | - | - | - | - | - |
| SlimGPT | gen | 40% | - | - | - | - | - | - | - | - | - | - | - |
| FLAP | ref | 20% | - | - | - | - | - | - | - | - | - | - | - |
| FLAP | ref | 30% | - | - | - | - | - | - | - | - | - | - | - |
| FLAP | ref | 40% | - | - | - | - | - | - | - | - | - | - | - |
| FLAP | gen | 20% | - | - | - | - | - | - | - | - | - | - | - |
| FLAP | gen | 30% | - | - | - | - | - | - | - | - | - | - | - |
| FLAP | gen | 40% | - | - | - | - | - | - | - | - | - | - | - |

## Ref vs gen calibration text

Calibrating on prompt + dense-generated continuation (gen) instead of prompt + reference (ref). Scores are the scope composites of the headline, over the directions both sides have. COMET pools segments (paired bootstrap); BLEU is the macro over directions with segments resampled within each direction.

| Method | Sparsity | Scope | COMET ref | COMET gen | gen - ref [95% CI] | p | BLEU ref | BLEU gen | gen - ref [95% CI] | p |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlimGPT | 20% | multi | - | - | - | - | - | - | - | - |
| SlimGPT | 20% | pair | - | - | - | - | - | - | - | - |
| SlimGPT | 20% | dir | - | - | - | - | - | - | - | - |
| SlimGPT | 30% | multi | - | - | - | - | - | - | - | - |
| SlimGPT | 30% | pair | - | - | - | - | - | - | - | - |
| SlimGPT | 30% | dir | - | - | - | - | - | - | - | - |
| SlimGPT | 40% | multi | - | - | - | - | - | - | - | - |
| SlimGPT | 40% | pair | - | - | - | - | - | - | - | - |
| SlimGPT | 40% | dir | - | - | - | - | - | - | - | - |
| FLAP | 20% | multi | - | - | - | - | - | - | - | - |
| FLAP | 20% | pair | - | - | - | - | - | - | - | - |
| FLAP | 20% | dir | - | - | - | - | - | - | - | - |
| FLAP | 30% | multi | - | - | - | - | - | - | - | - |
| FLAP | 30% | pair | - | - | - | - | - | - | - | - |
| FLAP | 30% | dir | - | - | - | - | - | - | - | - |
| FLAP | 40% | multi | - | - | - | - | - | - | - | - |
| FLAP | 40% | pair | - | - | - | - | - | - | - | - |
| FLAP | 40% | dir | - | - | - | - | - | - | - | - |

## SlimGPT vs FLAP

Deltas are SlimGPT minus FLAP at the same calibration text, sparsity and scope; positive means SlimGPT is better. Same composites and tests as above.

| Calib | Sparsity | Scope | COMET FLAP | COMET SlimGPT | SlimGPT - FLAP [95% CI] | p | BLEU FLAP | BLEU SlimGPT | SlimGPT - FLAP [95% CI] | p |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ref | 20% | multi | - | - | - | - | - | - | - | - |
| ref | 20% | pair | - | - | - | - | - | - | - | - |
| ref | 20% | dir | - | - | - | - | - | - | - | - |
| ref | 30% | multi | - | - | - | - | - | - | - | - |
| ref | 30% | pair | - | - | - | - | - | - | - | - |
| ref | 30% | dir | - | - | - | - | - | - | - | - |
| ref | 40% | multi | - | - | - | - | - | - | - | - |
| ref | 40% | pair | - | - | - | - | - | - | - | - |
| ref | 40% | dir | - | - | - | - | - | - | - | - |
| gen | 20% | multi | - | - | - | - | - | - | - | - |
| gen | 20% | pair | - | - | - | - | - | - | - | - |
| gen | 20% | dir | - | - | - | - | - | - | - | - |
| gen | 30% | multi | - | - | - | - | - | - | - | - |
| gen | 30% | pair | - | - | - | - | - | - | - | - |
| gen | 30% | dir | - | - | - | - | - | - | - | - |
| gen | 40% | multi | - | - | - | - | - | - | - | - |
| gen | 40% | pair | - | - | - | - | - | - | - | - |
| gen | 40% | dir | - | - | - | - | - | - | - | - |

## Per-direction scores, multi models

One row per multi model (plus dense). into-EN and out-of-EN are macros over the five directions on each side.

### COMET

| System | cs-en | de-en | is-en | ru-en | zh-en | into-EN | en-cs | en-de | en-is | en-ru | en-zh | out-of-EN | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (alma-7b) | 0.8524 | 0.8392 | 0.8627 | 0.8464 | 0.7961 | 0.8394 | 0.8703 | 0.8530 | 0.8418 | 0.8611 | 0.8507 | 0.8554 | 0.8474 |
| SlimGPT ref 20% | 0.8412 | 0.8306 | 0.8598 | 0.8357 | 0.7901 | 0.8315 | 0.8514 | 0.8432 | 0.8260 | 0.8449 | 0.8372 | 0.8405 | 0.8360 |
| SlimGPT ref 30% | 0.8366 | 0.8299 | 0.8467 | 0.8285 | 0.7714 | 0.8226 | 0.8384 | 0.8290 | 0.8039 | 0.8261 | 0.8129 | 0.8221 | 0.8223 |
| SlimGPT ref 40% | 0.8253 | 0.8215 | 0.8422 | 0.8134 | 0.7531 | 0.8111 | 0.7821 | 0.8017 | 0.7575 | 0.8003 | 0.7566 | 0.7796 | 0.7954 |
| SlimGPT gen 20% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| SlimGPT gen 30% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| SlimGPT gen 40% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP ref 20% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP ref 30% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP ref 40% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP gen 20% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP gen 30% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP gen 40% | - | - | - | - | - | - | - | - | - | - | - | - | - |

### BLEU

| System | cs-en | de-en | is-en | ru-en | zh-en | into-EN | en-cs | en-de | en-is | en-ru | en-zh | out-of-EN | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense (alma-7b) | 41.99 | 31.32 | 38.25 | 36.94 | 19.84 | 33.67 | 24.93 | 27.99 | 23.75 | 25.01 | 33.50 | 27.04 | 30.35 |
| SlimGPT ref 20% | 38.11 | 28.25 | 36.23 | 34.98 | 18.01 | 31.12 | 22.83 | 24.56 | 21.43 | 22.60 | 30.27 | 24.34 | 27.73 |
| SlimGPT ref 30% | 35.58 | 26.75 | 33.10 | 32.73 | 17.19 | 29.07 | 20.08 | 23.17 | 18.63 | 20.51 | 28.23 | 22.12 | 25.60 |
| SlimGPT ref 40% | 33.90 | 25.63 | 31.79 | 30.48 | 15.54 | 27.47 | 17.73 | 20.75 | 15.74 | 17.79 | 23.02 | 19.01 | 23.24 |
| SlimGPT gen 20% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| SlimGPT gen 30% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| SlimGPT gen 40% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP ref 20% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP ref 30% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP ref 40% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP gen 20% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP gen 30% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| FLAP gen 40% | - | - | - | - | - | - | - | - | - | - | - | - | - |

## Sparsity curve (macro COMET)

| Method | Calib | Scope | 0% (dense) | 20% | 30% | 40% |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| SlimGPT | ref | multi | 0.8474 | 0.8360 | 0.8223 | 0.7954 |
| SlimGPT | ref | pair | 0.8474 | 0.8380 | 0.8268 | 0.8067 |
| SlimGPT | ref | dir | 0.8474 | 0.8363 | 0.8282 | 0.8094 |
| SlimGPT | gen | multi | 0.8474 | - | - | - |
| SlimGPT | gen | pair | 0.8474 | - | - | - |
| SlimGPT | gen | dir | 0.8474 | - | - | - |
| FLAP | ref | multi | 0.8474 | - | - | - |
| FLAP | ref | pair | 0.8474 | - | - | - |
| FLAP | ref | dir | 0.8474 | - | - | - |
| FLAP | gen | multi | 0.8474 | - | - | - |
| FLAP | gen | pair | 0.8474 | - | - | - |
| FLAP | gen | dir | 0.8474 | - | - | - |

## Transfer

Every specialised model is evaluated on all ten directions. Each cell is the mean COMET over the matrix cells in that category, with the mean of (specialist minus the multi model of the same config, on the same direction) in parentheses. Pair: own = the two directions of the model's language, other = the remaining eight. Dir: own = the calibration direction, reverse = its reverse, same target / same source = other directions sharing the target / source language (into-English models have same-target neighbours, out-of-English models same-source ones), other = the rest. Full matrices are in `transfer/<config>.md` and `.csv`.

### Pair models (5 x 10)

| Config | Models | Own pair | Other 8 | Own - other |
| :--- | ---: | ---: | ---: | ---: |
| SlimGPT ref 20% | 5/5 | 0.8380 (+0.0020) | 0.8060 (-0.0300) | +0.0320 |
| SlimGPT ref 30% | 5/5 | 0.8268 (+0.0044) | 0.7624 (-0.0599) | +0.0644 |
| SlimGPT ref 40% | 5/5 | 0.8067 (+0.0113) | 0.7000 (-0.0954) | +0.1067 |
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
| SlimGPT ref 20% | 10/10 | 0.8363 (+0.0003) | 0.8368 (+0.0008) | 0.8272 (-0.0043) | 0.7820 (-0.0585) | 0.7978 (-0.0382) | -0.0005 |
| SlimGPT ref 30% | 10/10 | 0.8282 (+0.0059) | 0.8242 (+0.0019) | 0.8186 (-0.0040) | 0.7063 (-0.1158) | 0.7548 (-0.0675) | +0.0040 |
| SlimGPT ref 40% | 10/10 | 0.8094 (+0.0140) | 0.7986 (+0.0033) | 0.8001 (-0.0110) | 0.5967 (-0.1830) | 0.6866 (-0.1087) | +0.0107 |
| SlimGPT gen 20% | 0/10 | - | - | - | - | - | - |
| SlimGPT gen 30% | 0/10 | - | - | - | - | - | - |
| SlimGPT gen 40% | 0/10 | - | - | - | - | - | - |
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
| dense (alma-7b) | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% | 0.964 |
| SlimGPT ref 20% | 0.4% | 0.0% | 0.0% | 0.0% | 0.0% | 0.3% | 0.962 |
| SlimGPT ref 30% | 0.4% | 0.0% | 0.1% | 0.1% | 0.0% | 0.2% | 0.953 |
| SlimGPT ref 40% | 0.3% | 0.1% | 0.3% | 0.3% | 0.0% | 0.2% | 0.947 |
| SlimGPT gen 20% | - | - | - | - | - | - | - |
| SlimGPT gen 30% | - | - | - | - | - | - | - |
| SlimGPT gen 40% | - | - | - | - | - | - | - |
| FLAP ref 20% | - | - | - | - | - | - | - |
| FLAP ref 30% | - | - | - | - | - | - | - |
| FLAP ref 40% | - | - | - | - | - | - | - |
| FLAP gen 20% | - | - | - | - | - | - | - |
| FLAP gen 30% | - | - | - | - | - | - | - |
| FLAP gen 40% | - | - | - | - | - | - | - |

## Repair (LoRA)

No `-lora` systems found yet.

## Structure

Removed parameters as a share of the whole model (embeddings included), from kept head and FFN channel counts. Found for 48/48 models; pair and dir cells show the mean and range over their models. Per-model figures are in structure.csv.

| Config | multi | pair | dir |
| :--- | ---: | ---: | ---: |
| SlimGPT ref 20% | 19.3% | 19.3% (19.3% to 19.3%, n=5) | 19.3% (19.3% to 19.3%, n=10) |
| SlimGPT ref 30% | 28.8% | 28.8% (28.8% to 28.8%, n=5) | 28.8% (28.8% to 28.8%, n=10) |
| SlimGPT ref 40% | 38.4% | 38.4% (38.4% to 38.4%, n=5) | 38.4% (38.4% to 38.4%, n=10) |
| SlimGPT gen 20% | - | - | - |
| SlimGPT gen 30% | - | - | - |
| SlimGPT gen 40% | - | - | - |
| FLAP ref 20% | - | - | - |
| FLAP ref 30% | - | - | - |
| FLAP ref 40% | - | - | - |
| FLAP gen 20% | - | - | - |
| FLAP gen 30% | - | - | - |
| FLAP gen 40% | - | - | - |

### Per-layer kept units, multi models

Heads are counts per layer (of 32); FFN is the kept share of 11008 channels. min / mean / max over layers, then mean of the first four / last four layers.

| System | Params removed | Heads min/mean/max | Heads first4/last4 | FFN min/mean/max | FFN first4/last4 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SlimGPT ref 20% | 19.3% | 24 / 25.5 / 30 | 28.8 / 24.0 | 75% / 80% / 95% | 90% / 75% |
| SlimGPT ref 30% | 28.8% | 20 / 22.4 / 30 | 27.5 / 20.0 | 62% / 70% / 92% | 85% / 62% |
| SlimGPT ref 40% | 38.4% | 16 / 19.2 / 29 | 26.0 / 16.0 | 49% / 60% / 90% | 81% / 50% |

## Warnings

None.
