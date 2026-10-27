# Severity sweep - dATE% = (supervised - baseline) / baseline, positive = worse

_Recomputed from the stored paired ATE arrays; the earlier table had the sign inverted._

| Severity | Scenario | Baseline ATE +- std | Supervised ATE +- std | dATE% | Pairs won |
|---|---|---|---|---|---|
| 0.35 | motion_blur | 0.2094 +-0.0459 | 0.2231 +-0.0487 | +6.6 | 0/2 |
| 0.35 | low_light | 0.1777 +-0.0743 | 0.1746 +-0.0799 | -3.8 | 1/2 |
| 0.35 | texture_loss | 0.1906 +-0.0514 | 0.2039 +-0.0611 | +6.0 | 0/2 |
| 0.70 | motion_blur | 0.2383 +-0.1414 | 0.1890 +-0.0614 | -1.2 | 1/2 |
| 0.70 | low_light | 0.1657 +-0.0737 | 0.1740 +-0.0677 | +8.3 | 0/2 |
| 0.70 | texture_loss | 0.1843 +-0.0541 | 0.2016 +-0.0714 | +7.3 | 0/2 |
| 1.00 | motion_blur | 0.1720 +-0.0789 | 0.1735 +-0.1004 | -6.2 | 1/2 |
| 1.00 | low_light | 0.1839 +-0.0985 | 0.1853 +-0.0910 | +4.1 | 1/2 |
| 1.00 | texture_loss | 0.1813 +-0.0514 | 0.2097 +-0.0692 | +14.1 | 0/2 |
