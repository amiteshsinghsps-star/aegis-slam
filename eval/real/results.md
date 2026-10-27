# AEGIS-SLAM - real-footage benchmark (TUM RGB-D)

_Real handheld RGB used monocularly, first 90 frames per sequence, distortion removed at preparation. Faults injected on frames 35-65% at severity 0.6. ATE = Umeyama Sim(3) RMSE on camera centres, the same metric as the synthetic grid. One run per cell._

| Sequence | Fault | Condition | ATE (m) | vs clean | Tracking | Coverage | Recover(frames) | Decisions | Rollbacks |
|---|---|---|---|---|---|---|---|---|---|
| fr1_desk | clean | baseline_classical | 0.0495 | - | 0.978 | 1.0 | - | 0 | 0 |
| fr1_desk | clean | supervised | 0.0495 | - | 0.978 | 1.0 | - | 0 | 0 |
| fr1_desk | clean | supervised_multi | 0.0495 | - | 0.978 | 1.0 | - | 0 | 0 |
| fr1_desk | low_light | baseline_classical | 0.0943 | +90.5% | 0.967 | 1.0 | 2 | 0 | 0 |
| fr1_desk | low_light | supervised | 0.0943 | +90.5% | 0.967 | 1.0 | 2 | 0 | 0 |
| fr1_desk | low_light | supervised_multi | 0.0943 | +90.5% | 0.967 | 1.0 | 2 | 0 | 0 |
| fr1_desk | motion_blur | baseline_classical | 0.147 | +197.0% | 0.911 | 1.0 | 7 | 0 | 0 |
| fr1_desk | motion_blur | supervised | 0.2136 | +331.5% | 0.911 | 1.0 | 7 | 1 | 0 |
| fr1_desk | motion_blur | supervised_multi | 0.2136 | +331.5% | 0.911 | 1.0 | 7 | 1 | 0 |
| fr1_desk | texture_loss | baseline_classical | 0.0699 | +41.2% | 0.956 | 1.0 | 2 | 0 | 0 |
| fr1_desk | texture_loss | supervised | 0.1038 | +109.7% | 0.944 | 1.0 | 2 | 2 | 2 |
| fr1_desk | texture_loss | supervised_multi | 0.113 | +128.3% | 0.956 | 1.0 | 2 | 2 | 2 |
| fr1_xyz | clean | baseline_classical | 0.0298 | - | 0.989 | 1.0 | - | 0 | 0 |
| fr1_xyz | clean | supervised | 0.0353 | - | 0.989 | 1.0 | - | 3 | 3 |
| fr1_xyz | clean | supervised_multi | 0.0347 | - | 0.989 | 1.0 | - | 3 | 2 |
| fr1_xyz | low_light | baseline_classical | 0.0277 | -7.0% | 0.989 | 1.0 | 2 | 0 | 0 |
| fr1_xyz | low_light | supervised | 0.0318 | -9.9% | 0.989 | 1.0 | 2 | 3 | 3 |
| fr1_xyz | low_light | supervised_multi | 0.0338 | -2.6% | 0.989 | 1.0 | 2 | 3 | 2 |
| fr1_xyz | motion_blur | baseline_classical | 0.0361 | +21.1% | 0.989 | 1.0 | 2 | 0 | 0 |
| fr1_xyz | motion_blur | supervised | 0.0557 | +57.8% | 0.878 | 1.0 | 16 | 2 | 1 |
| fr1_xyz | motion_blur | supervised_multi | 0.0557 | +60.5% | 0.878 | 1.0 | 16 | 2 | 1 |
| fr1_xyz | texture_loss | baseline_classical | 0.0451 | +51.3% | 0.989 | 1.0 | 2 | 0 | 0 |
| fr1_xyz | texture_loss | supervised | 0.0385 | +9.1% | 0.989 | 1.0 | 2 | 3 | 3 |
| fr1_xyz | texture_loss | supervised_multi | 0.0354 | +2.0% | 0.989 | 1.0 | 2 | 3 | 1 |

## Across sequences

_Mean over the sequences and the paired per-sequence change against the classical baseline for the same sequence and fault (negative = lower ATE)._

| Fault | Condition | Mean ATE (m) | Paired ATE vs baseline (%) | Tracking | Decisions | Rollbacks |
|---|---|---|---|---|---|---|
| clean | baseline_classical | 0.0397 | +0.0% | 0.984 | 0 | 0 |
| clean | supervised | 0.0424 | +9.2% | 0.984 | 1.5 | 1.5 |
| clean | supervised_multi | 0.0421 | +8.2% | 0.984 | 1.5 | 1 |
| motion_blur | baseline_classical | 0.0915 | +0.0% | 0.95 | 0 | 0 |
| motion_blur | supervised | 0.1346 | +49.8% | 0.895 | 1.5 | 0.5 |
| motion_blur | supervised_multi | 0.1346 | +49.8% | 0.895 | 1.5 | 0.5 |
| low_light | baseline_classical | 0.061 | +0.0% | 0.978 | 0 | 0 |
| low_light | supervised | 0.063 | +7.4% | 0.978 | 1.5 | 1.5 |
| low_light | supervised_multi | 0.064 | +11.0% | 0.978 | 1.5 | 1 |
| texture_loss | baseline_classical | 0.0575 | +0.0% | 0.972 | 0 | 0 |
| texture_loss | supervised | 0.0712 | +16.9% | 0.966 | 2.5 | 2.5 |
| texture_loss | supervised_multi | 0.0742 | +20.1% | 0.972 | 2.5 | 1.5 |
