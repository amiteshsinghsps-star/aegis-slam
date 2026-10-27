# AEGIS-SLAM — Evaluation results

_Synthetic sequences, 90 frames/clip, faults injected mid-sequence. Each cell is the mean over **3 distinct scenes (seeds 0..2)**; ATE std is across those scenes._

| Scenario | Condition | ATE mean (m) | ATE std | Coverage | Recover(frames) | Not-recovered | Decisions | Switches | Rollbacks |
|---|---|---|---|---|---|---|---|---|---|
| motion_blur | baseline_classical | 0.1644 | 0.0653 | 1.0 | 19.3 | 0 | 0 | 0 | 0 |
| motion_blur | supervised | 0.193 | 0.0865 | 1.0 | 12.7 | 0 | 3 | 0 | 2.3 |
| motion_blur | supervised_multi | 0.1566 | 0.0544 | 1.0 | 10.7 | 0 | 3 | 0 | 2 |
| motion_blur | ablation | 0.204 | 0.0781 | 1.0 | 4.5 | 1 | 3 | 0 | 0 |
| low_light | baseline_classical | 0.1794 | 0.0807 | 1.0 | 11 | 0 | 0 | 0 | 0 |
| low_light | supervised | 0.2408 | 0.1081 | 1.0 | 2 | 1 | 2.3 | 0 | 2 |
| low_light | supervised_multi | 0.2363 | 0.1071 | 1.0 | 2 | 1 | 2.3 | 0 | 2 |
| low_light | ablation | 0.1962 | 0.0753 | 1.0 | 2 | 1 | 2.3 | 0 | 0 |
| texture_loss | baseline_classical | 0.2001 | 0.0496 | 1.0 | 17 | 0 | 0 | 0 | 0 |
| texture_loss | supervised | 0.1971 | 0.0593 | 1.0 | 2 | 1 | 2.3 | 0 | 2.3 |
| texture_loss | supervised_multi | 0.1973 | 0.0533 | 1.0 | 2 | 1 | 2.3 | 0 | 2.3 |
| texture_loss | ablation | 0.2599 | 0.1048 | 1.0 | 2 | 1 | 2.3 | 0 | 0 |
| frame_drop | baseline_classical | 0.2125 | 0.0616 | 1.0 | 23.3 | 0 | 0 | 0 | 0 |
| frame_drop | supervised | 0.1903 | 0.0707 | 1.0 | 23.3 | 0 | 2.3 | 0 | 1.7 |
| frame_drop | supervised_multi | 0.2207 | 0.0682 | 0.926 | 11.5 | 1 | 2.7 | 0 | 1.7 |
| frame_drop | ablation | 0.2969 | 0.1161 | 1.0 | 11.5 | 1 | 2.3 | 0 | 0 |

## False interventions on clean video
- decisions per run: [2, 3, 3] (mean 2.67)
