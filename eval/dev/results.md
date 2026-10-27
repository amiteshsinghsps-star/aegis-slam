# AEGIS-SLAM — Evaluation results

_Synthetic sequences, 60 frames/clip, faults injected mid-sequence. Each cell is the mean over **1 distinct scenes (seed 0)**. `ATE all` is the primary metric: every frame from the first pose, with missing poses filled by the last one. `ATE trusted` uses only frames the agent did not distrust, so it must be read with coverage._

| Scenario | Condition | ATE all (m) | ATE trusted (m) | Trusted cov | Estimated cov | Recover (frames) | Recovered n | Not recovered | Decisions | Switches | Rollbacks | Wall (s) | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| low_light | baseline_classical | 0.0806 ± 0.0 | 0.0811 ± 0.0 | 1.0 | 0.95 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 51.41 | 1.2 |
| low_light | supervised | 0.0806 ± 0.0 | 0.0811 ± 0.0 | 1.0 | 0.95 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 47.32 | 1.3 |
| low_light | supervised_multi | 0.0806 ± 0.0 | 0.0811 ± 0.0 | 1.0 | 0.95 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 47.36 | 1.3 |
| low_light | ablation | 0.0806 ± 0.0 | 0.0811 ± 0.0 | 1.0 | 0.95 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 47.72 | 1.3 |
| texture_loss | baseline_classical | 0.1068 ± 0.0 | 0.107 ± 0.0 | 1.0 | 0.95 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 45.02 | 1.3 |
| texture_loss | supervised | 0.1068 ± 0.0 | 0.107 ± 0.0 | 1.0 | 0.95 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 45.34 | 1.3 |
| texture_loss | supervised_multi | 0.1068 ± 0.0 | 0.107 ± 0.0 | 1.0 | 0.95 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 42.61 | 1.4 |
| texture_loss | ablation | 0.1068 ± 0.0 | 0.107 ± 0.0 | 1.0 | 0.95 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 41.88 | 1.4 |
| frame_drop | baseline_classical | 0.0701 ± 0.0 | 0.0668 ± 0.0 | 1.0 | 0.817 | 11 (n=1) | 1 | 0 | 0 | 0 | 0 | 48.41 | 1.2 |
| frame_drop | supervised | 0.077 ± 0.0 | 0.0768 ± 0.0 | 1.0 | 0.65 | 11 (n=1) | 1 | 0 | 1 | 0 | 1 | 45.99 | 1.3 |
| frame_drop | supervised_multi | 0.0701 ± 0.0 | 0.0256 ± 0.0 | 0.667 | 0.817 | 11 (n=1) | 1 | 0 | 1 | 0 | 0 | 38.66 | 1.6 |
| frame_drop | ablation | 0.3032 ± 0.0 | 0.3242 ± 0.0 | 1.0 | 0.783 | 11 (n=1) | 1 | 0 | 1 | 0 | 0 | 26.69 | 2.2 |
| motion_blur | baseline_classical | 0.1044 ± 0.0 | 0.1004 ± 0.0 | 1.0 | 0.85 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 28.11 | 2.1 |
| motion_blur | supervised | 0.1044 ± 0.0 | 0.1004 ± 0.0 | 1.0 | 0.85 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 27.7 | 2.2 |
| motion_blur | supervised_multi | 0.1044 ± 0.0 | 0.1004 ± 0.0 | 1.0 | 0.85 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 28.11 | 2.1 |
| motion_blur | ablation | 0.1044 ± 0.0 | 0.1004 ± 0.0 | 1.0 | 0.85 | 2 (n=1) | 1 | 0 | 0 | 0 | 0 | 26.95 | 2.2 |

`Recover (frames)` is **conditional on recovery**: it averages only the scenes that recovered. `Not recovered` counts right-censored scenes that never stabilised inside the observation window — a small conditional mean next to a non-zero censored count is a failure, not a speed-up.

## False interventions on clean video

| Seed | Condition | Decisions | Diagnoses | Verdicts | ATE all (m) |
|---|---|---|---|---|---|
| 0 | supervised | 0 | {} | {} | 0.1072 |
| 0 | supervised_multi | 0 | {} | {} | 0.1072 |
