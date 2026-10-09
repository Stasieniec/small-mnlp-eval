# Generated calibration against the references

Dense ALMA-7B greedy translations of the calibration sources, scored against
the reference each record carries. High agreement means the reference and
generated calibration arms see similar target text.

| cache | direction | n | chrf++ | bleu | exact_match | empty | hit_token_budget | source_truncated | generated_tokens | input_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dir-cs-en-1280-generated | cs-en | 1280 | 60.2 | 37.68 | 0.0227 | 0 | 0 | 0 | 41159 | 86734 |
| dir-de-en-1280-generated | de-en | 1280 | 65.97 | 45.12 | 0.0281 | 0 | 0 | 0 | 39958 | 74231 |
| dir-en-cs-1280-generated | en-cs | 1280 | 49.36 | 25.12 | 0.0094 | 0 | 0 | 0 | 65419 | 61552 |
| dir-en-de-1280-generated | en-de | 1280 | 59.13 | 33.59 | 0.0133 | 0 | 0 | 0 | 55203 | 60911 |
| dir-en-is-1280-generated | en-is | 1280 | 50.08 | 26.4 | 0.0055 | 0 | 0 | 0 | 76395 | 65584 |
| dir-en-ru-1280-generated | en-ru | 1280 | 54.92 | 32.39 | 0.0219 | 0 | 0 | 0 | 58717 | 59027 |
| dir-en-zh-1280-generated | en-zh | 1280 | 29.69 | 38.74 | 0.0039 | 0 | 6 | 0 | 83363 | 68712 |
| dir-is-en-1280-generated | is-en | 1280 | 62.26 | 41.42 | 0.007 | 0 | 0 | 0 | 39069 | 103313 |
| dir-ru-en-1280-generated | ru-en | 1280 | 63.0 | 41.49 | 0.0148 | 0 | 2 | 0 | 39841 | 80221 |
| dir-zh-en-1280-generated | zh-en | 1280 | 56.15 | 30.71 | 0.0016 | 0 | 0 | 0 | 54217 | 116097 |
| multi-10dir-generated | cs-en | 128 | 59.85 | 37.19 | 0.0078 | 0 | 0 | 0 | 3975 | 8608 |
| multi-10dir-generated | de-en | 128 | 67.03 | 46.07 | 0.0312 | 0 | 0 | 0 | 3974 | 7400 |
| multi-10dir-generated | en-cs | 128 | 49.47 | 23.98 | 0.0156 | 0 | 0 | 0 | 7051 | 6462 |
| multi-10dir-generated | en-de | 128 | 60.89 | 35.53 | 0.0312 | 0 | 0 | 0 | 5746 | 6147 |
| multi-10dir-generated | en-is | 128 | 50.15 | 27.29 | 0.0156 | 0 | 0 | 0 | 7431 | 6503 |
| multi-10dir-generated | en-ru | 128 | 53.65 | 29.29 | 0.0234 | 0 | 0 | 0 | 5858 | 5779 |
| multi-10dir-generated | en-zh | 128 | 30.07 | 40.42 | 0.0078 | 0 | 0 | 0 | 8383 | 7009 |
| multi-10dir-generated | is-en | 128 | 61.95 | 41.64 | 0.0078 | 0 | 0 | 0 | 3740 | 10264 |
| multi-10dir-generated | ru-en | 128 | 61.79 | 38.96 | 0.0312 | 0 | 0 | 0 | 3966 | 8070 |
| multi-10dir-generated | zh-en | 128 | 57.01 | 31.56 | 0.0 | 0 | 0 | 0 | 5733 | 11985 |
| pair-cs-640-generated | cs-en | 640 | 59.97 | 37.35 | 0.0203 | 0 | 0 | 0 | 20541 | 43618 |
| pair-cs-640-generated | en-cs | 640 | 48.77 | 24.41 | 0.0063 | 0 | 0 | 0 | 32633 | 30892 |
| pair-de-640-generated | de-en | 640 | 65.94 | 44.83 | 0.0203 | 0 | 0 | 0 | 19613 | 36788 |
| pair-de-640-generated | en-de | 640 | 59.48 | 34.25 | 0.0156 | 0 | 0 | 0 | 27536 | 30328 |
| pair-is-640-generated | en-is | 640 | 49.57 | 26.02 | 0.0063 | 0 | 0 | 0 | 37423 | 32619 |
| pair-is-640-generated | is-en | 640 | 62.26 | 41.65 | 0.0047 | 0 | 0 | 0 | 19417 | 51340 |
| pair-ru-640-generated | en-ru | 640 | 54.68 | 32.11 | 0.0219 | 0 | 0 | 0 | 29350 | 29362 |
| pair-ru-640-generated | ru-en | 640 | 62.85 | 41.45 | 0.0172 | 0 | 1 | 0 | 19958 | 40184 |
| pair-zh-640-generated | en-zh | 640 | 30.08 | 38.72 | 0.0063 | 0 | 3 | 0 | 41540 | 34289 |
| pair-zh-640-generated | zh-en | 640 | 56.32 | 30.57 | 0.0016 | 0 | 0 | 0 | 27748 | 59059 |
