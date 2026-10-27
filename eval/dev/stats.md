# Paired statistics (ΔATE% = condition − baseline; positive = worse)

_Seeds: [0]. Paired on the same synthetic scene. CI = 10,000-resample bootstrap; p = two-sided Wilcoxon signed-rank._

| Condition | n | ΔATE% mean [95% CI] | Median | W / T / L | Wilcoxon p | ATE base → cond (m) |
|---|---|---|---|---|---|---|
| supervised | 4 | +2.46 [0.0, 7.37] | +0.00 | 0 / 3 / 1 | 1.0 | 0.0905 → 0.0922 |
| supervised_multi | 4 | +0.00 [0.0, 0.0] | +0.00 | 0 / 4 / 0 | 1.0 | 0.0905 → 0.0905 |
| ablation | 4 | +83.07 [0.0, 249.22] | +0.00 | 0 / 3 / 1 | 1.0 | 0.0905 → 0.1487 |

## Per scenario

| Scenario | Condition | n | ΔATE% mean [95% CI] | W / T / L | p |
|---|---|---|---|---|---|
| motion_blur | supervised | 1 | +0.00 — | 0 / 1 / 0 | 1.0 |
| motion_blur | supervised_multi | 1 | +0.00 — | 0 / 1 / 0 | 1.0 |
| motion_blur | ablation | 1 | +0.00 — | 0 / 1 / 0 | 1.0 |
| low_light | supervised | 1 | +0.00 — | 0 / 1 / 0 | 1.0 |
| low_light | supervised_multi | 1 | +0.00 — | 0 / 1 / 0 | 1.0 |
| low_light | ablation | 1 | +0.00 — | 0 / 1 / 0 | 1.0 |
| texture_loss | supervised | 1 | +0.00 — | 0 / 1 / 0 | 1.0 |
| texture_loss | supervised_multi | 1 | +0.00 — | 0 / 1 / 0 | 1.0 |
| texture_loss | ablation | 1 | +0.00 — | 0 / 1 / 0 | 1.0 |
| frame_drop | supervised | 1 | +9.83 — | 0 / 0 / 1 | 1.0 |
| frame_drop | supervised_multi | 1 | +0.00 — | 0 / 1 / 0 | 1.0 |
| frame_drop | ablation | 1 | +332.29 — | 0 / 0 / 1 | 1.0 |
