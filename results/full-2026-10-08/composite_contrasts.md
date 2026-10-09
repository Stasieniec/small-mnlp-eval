# Composite contrasts

## Specialisation after repair

Five pair models, each LoRA-repaired on its own pair, against the multi model
LoRA-repaired on all ten directions; generated calibration. Macro over the ten
directions; paired bootstrap stratified by direction (1,000 resamples). COMET:
positive favours the pair models; MetricX-24 (lower is better): negative does.

| method | removed | multi + LoRA COMET | pair + LoRA COMET | difference [95% CI] | p | multi + LoRA MetricX | pair + LoRA MetricX | difference [95% CI] | p |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SlimGPT | 30% | 0.8312 | 0.8353 | +0.0041 [+0.0031, +0.0051] | <0.001 | 3.474 | 3.371 | -0.103 [-0.146, -0.062] | <0.001 |
| SlimGPT | 40% | 0.8196 | 0.8248 | +0.0052 [+0.0041, +0.0064] | <0.001 | 3.915 | 3.783 | -0.132 [-0.179, -0.084] | <0.001 |
| FLAP | 30% | 0.8239 | 0.8261 | +0.0022 [+0.0010, +0.0034] | <0.001 | 3.831 | 3.758 | -0.073 [-0.124, -0.029] | 0.003 |
| FLAP | 40% | 0.8113 | 0.8118 | +0.0006 [-0.0008, +0.0020] | 0.402 | 4.252 | 4.288 | +0.037 [-0.021, +0.090] | 0.168 |

## Pilot against full suite

Headline composites (method x calibration x sparsity x scope, 36 cells):
Pearson r = 0.9995, Spearman rho = 0.9923,
mean |full - pilot| = 0.0013 COMET, largest 0.0044.
