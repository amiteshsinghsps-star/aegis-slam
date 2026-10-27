# Trust / risk–coverage metrics

_High-error label = framed error ≥ max(5 cm, run 75th percentile). AUROC > 0.5 means the score ranks high-error frames higher; lower AURC is better. Silent-failure rate = high-error frames whose confidence was still ≥ 0.5._

| Cell | Condition | AUROC conf | AUROC inliers | AUROC landmarks | AURC conf | Silent failure |
|---|---|---|---|---|---|---|
| clean | supervised | 0.153 | 0.153 | 0.879 | 0.0645 | 1.0 |
| clean | supervised_multi | 0.153 | 0.153 | 0.879 | 0.0645 | 1.0 |
| frame_drop | ablation | 0.51 | 0.457 | 0.183 | 0.3341 | 0.833 |
| frame_drop | baseline_classical | 0.112 | 0.125 | 0.354 | 0.0407 | 0.923 |
| frame_drop | supervised | 0.043 | 0.043 | 0.467 | 0.0464 | 0.9 |
| frame_drop | supervised_multi | 0.112 | 0.125 | 0.354 | 0.0407 | 0.923 |
| low_light | ablation | 0.388 | 0.382 | 0.75 | 0.054 | 1.0 |
| low_light | baseline_classical | 0.388 | 0.382 | 0.75 | 0.054 | 1.0 |
| low_light | supervised | 0.388 | 0.382 | 0.75 | 0.054 | 1.0 |
| low_light | supervised_multi | 0.388 | 0.382 | 0.75 | 0.054 | 1.0 |
| motion_blur | ablation | 0.39 | 0.442 | 0.398 | 0.0906 | 0.615 |
| motion_blur | baseline_classical | 0.39 | 0.442 | 0.398 | 0.0906 | 0.615 |
| motion_blur | supervised | 0.39 | 0.442 | 0.398 | 0.0906 | 0.615 |
| motion_blur | supervised_multi | 0.39 | 0.442 | 0.398 | 0.0906 | 0.615 |
| texture_loss | ablation | 0.243 | 0.275 | 0.561 | 0.091 | 0.867 |
| texture_loss | baseline_classical | 0.243 | 0.275 | 0.561 | 0.091 | 0.867 |
| texture_loss | supervised | 0.243 | 0.275 | 0.561 | 0.091 | 0.867 |
| texture_loss | supervised_multi | 0.243 | 0.275 | 0.561 | 0.091 | 0.867 |

## Trusted ATE at matched coverage
Agent-gated 0.0969 m vs baseline-gated 0.0761 m across 12 paired cells (Δ +38.64 %, positive = worse).
