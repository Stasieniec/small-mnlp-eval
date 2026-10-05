# Direction-specific pruning pilot

Positive deltas favour direction-only calibration. COMET is the primary metric.
Holm correction covers all direction-versus-pair and direction-versus-multi contrasts.

| Direction | Control scope | COMET delta | 95% CI | Holm p |
| --- | --- | ---: | --- | ---: |
| de-en | multi | -0.05079 | [-0.09722, -0.00838] | 0.1050 |
| de-en | pair-de | +0.01702 | [-0.04030, +0.07425] | 1.0000 |
| en-de | multi | -0.01693 | [-0.06192, +0.02543] | 1.0000 |
| en-de | pair-de | -0.02120 | [-0.07897, +0.03663] | 1.0000 |
| is-en | multi | -0.09627 | [-0.14247, -0.04815] | 0.0040 |
| is-en | pair-is | -0.06114 | [-0.10185, -0.02034] | 0.0315 |
| en-is | multi | +0.05720 | [+0.01558, +0.09741] | 0.0480 |
| en-is | pair-is | +0.04519 | [+0.00566, +0.08620] | 0.1259 |

One calibration seed; segment bootstrap does not cover calibration variability.
Budgets match segment counts, not token counts; language lengths can differ.
The first evaluation segments are a pilot, not a random sample.
Other directions in by-direction.csv are exploratory transfer measurements.
