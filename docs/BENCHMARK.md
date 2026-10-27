# AEGIS-SLAM — benchmark

_Measured 2026-10-27 on the machine in §8. Every number below was produced by the shipped scripts; raw JSON lives next to each table. Sign conventions are stated where they apply._

## 1. Protocol

* **Clips:** synthetic TUM-style scenes from `aegis.synth`, **60 frames**, one distinct scene per seed. Faults injected on frames 21–39 (35–65 %); `frame_drop` is an 8-frame blackout burst.
* **Seeds:** 0 only (one scene — dev pilot). `docs/PREREG.md` records the full protocol (90 frames, seeds 0–4 dev, 10–17 locked) and the deviation: the locked split was **not run** — every table here is dev-split and labelled so.
* **Conditions:** `baseline_classical` (no agent), `supervised` (single agent), `supervised_multi` (five roles), `ablation` (same agent, verification off).
* **Primary metric:** `ATE all` — RMSE of camera centres over *every* frame after Sim(3) alignment, hold-last across frames with no pose. `ATE trusted` covers only trusted frames and is always printed with `coverage trusted`.
* **Recovery:** frames from fault onset to 3 consecutive stable frames; runs that never recover are right-censored and counted, never dropped.
* **Sign convention:** ΔATE % = (condition − baseline) / baseline × 100, so **positive = worse**. Every table states it again.

Failed runs: **0**. All cells completed.

## 2. Main table (dev, paired per scene)

| Scenario | Condition | ATE all (m) | ATE trusted (m) | Cov trusted | Cov est. | Recovered (frames) | Not rec. | Decisions | Rollbacks | Wall (s) | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| motion_blur | baseline_classical | 0.1044 ±0.0000 | 0.1004 ±0.0000 | 1.000 | 0.850 | 2 (n=1) | 0 | 0 | 0 | 28.1 | 2.1 |
| motion_blur | supervised | 0.1044 ±0.0000 | 0.1004 ±0.0000 | 1.000 | 0.850 | 2 (n=1) | 0 | 0 | 0 | 27.7 | 2.2 |
| motion_blur | supervised_multi | 0.1044 ±0.0000 | 0.1004 ±0.0000 | 1.000 | 0.850 | 2 (n=1) | 0 | 0 | 0 | 28.1 | 2.1 |
| motion_blur | ablation | 0.1044 ±0.0000 | 0.1004 ±0.0000 | 1.000 | 0.850 | 2 (n=1) | 0 | 0 | 0 | 26.9 | 2.2 |
| low_light | baseline_classical | 0.0806 ±0.0000 | 0.0811 ±0.0000 | 1.000 | 0.950 | 2 (n=1) | 0 | 0 | 0 | 51.4 | 1.2 |
| low_light | supervised | 0.0806 ±0.0000 | 0.0811 ±0.0000 | 1.000 | 0.950 | 2 (n=1) | 0 | 0 | 0 | 47.3 | 1.3 |
| low_light | supervised_multi | 0.0806 ±0.0000 | 0.0811 ±0.0000 | 1.000 | 0.950 | 2 (n=1) | 0 | 0 | 0 | 47.4 | 1.3 |
| low_light | ablation | 0.0806 ±0.0000 | 0.0811 ±0.0000 | 1.000 | 0.950 | 2 (n=1) | 0 | 0 | 0 | 47.7 | 1.3 |
| texture_loss | baseline_classical | 0.1068 ±0.0000 | 0.1070 ±0.0000 | 1.000 | 0.950 | 2 (n=1) | 0 | 0 | 0 | 45.0 | 1.3 |
| texture_loss | supervised | 0.1068 ±0.0000 | 0.1070 ±0.0000 | 1.000 | 0.950 | 2 (n=1) | 0 | 0 | 0 | 45.3 | 1.3 |
| texture_loss | supervised_multi | 0.1068 ±0.0000 | 0.1070 ±0.0000 | 1.000 | 0.950 | 2 (n=1) | 0 | 0 | 0 | 42.6 | 1.4 |
| texture_loss | ablation | 0.1068 ±0.0000 | 0.1070 ±0.0000 | 1.000 | 0.950 | 2 (n=1) | 0 | 0 | 0 | 41.9 | 1.4 |
| frame_drop | baseline_classical | 0.0701 ±0.0000 | 0.0668 ±0.0000 | 1.000 | 0.817 | 11 (n=1) | 0 | 0 | 0 | 48.4 | 1.2 |
| frame_drop | supervised | 0.0770 ±0.0000 | 0.0768 ±0.0000 | 1.000 | 0.650 | 11 (n=1) | 0 | 1 | 1 | 46.0 | 1.3 |
| frame_drop | supervised_multi | 0.0701 ±0.0000 | 0.0256 ±0.0000 | 0.667 | 0.817 | 11 (n=1) | 0 | 1 | 0 | 38.7 | 1.6 |
| frame_drop | ablation | 0.3032 ±0.0000 | 0.3242 ±0.0000 | 1.000 | 0.783 | 11 (n=1) | 0 | 1 | 0 | 26.7 | 2.2 |

`Recovered` averages only scenes that recovered; `Not rec.` is the count that did not, so the two must be read together.

## 3. Paired deltas and statistics

Paired bootstrap (10 000 resamples, 95 % percentile CI) on per-scene ΔATE all, positive = worse. Wilcoxon signed-rank two-sided, n = 1 pairs per cell. **With one scene the bootstrap CI is undefined and Wilcoxon has no
power: every p = 1.0 is a property of n = 1, not evidence of equivalence.**

| Scenario | Comparison | ΔATE % (median) | 95 % CI | Wins/Ties/Losses | Wilcoxon p |
|---|---|---|---|---|---|
| motion_blur | supervised | +0.0 | — | 0/1/0 | 1.0 |
| motion_blur | supervised_multi | +0.0 | — | 0/1/0 | 1.0 |
| motion_blur | ablation | +0.0 | — | 0/1/0 | 1.0 |
| low_light | supervised | +0.0 | — | 0/1/0 | 1.0 |
| low_light | supervised_multi | +0.0 | — | 0/1/0 | 1.0 |
| low_light | ablation | +0.0 | — | 0/1/0 | 1.0 |
| texture_loss | supervised | +0.0 | — | 0/1/0 | 1.0 |
| texture_loss | supervised_multi | +0.0 | — | 0/1/0 | 1.0 |
| texture_loss | ablation | +0.0 | — | 0/1/0 | 1.0 |
| frame_drop | supervised | +9.8 | — | 0/0/1 | 1.0 |
| frame_drop | supervised_multi | +0.0 | — | 0/1/0 | 1.0 |
| frame_drop | ablation | +332.3 | — | 0/0/1 | 1.0 |

_Pooled rows are in `eval/dev/stats.md`._

## 4. Clean video: false interventions

| Seed | Condition | Decisions | Diagnoses | Verdicts | ATE all (m) |
|---|---|---|---|---|---|
| 0 | supervised | 0 | {} | {} | 0.1072 |
| 0 | supervised_multi | 0 | {} | {} | 0.1072 |

**0 decisions across 2 clean runs.** A non-zero cell is a false intervention; we report them rather than hiding them.

## 5. Trust and risk–coverage

High-error label = framed error ≥ max(5 cm, run 75th percentile). AUROC > 0.5 means the score ranks high-error frames higher; lower AURC is better. `Silent` = high-error frames whose confidence was still ≥ 0.5.

| Scenario | Condition | AUROC confidence | AUROC inliers | AUROC landmarks | AURC | Silent failure |
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

**Trusted ATE at matched coverage:** agent-gated 0.0969 m vs baseline-gated 0.0761 m over 12 paired cells (Δ +38.64 %, positive = worse).

## 6. Severity sweep (v2 data, corrected sign)

The sweep was measured before the v3 supervisor changes and has **not** been re-run; it is included because the sign convention in the earlier report was wrong (it printed improvement as positive). Deltas below are recomputed from the stored paired ATE arrays as (supervised − baseline) / baseline, so **positive = worse**.

| Severity | Scenario | Baseline ATE | Supervised ATE | ΔATE % | Pairs won |
|---|---|---|---|---|---|
| 0.35 | motion_blur | 0.2094 | 0.2231 | +6.6 | 0/2 |
| 0.35 | low_light | 0.1777 | 0.1746 | -3.8 | 1/2 |
| 0.35 | texture_loss | 0.1906 | 0.2039 | +6.0 | 0/2 |
| 0.70 | motion_blur | 0.2383 | 0.1890 | -1.2 | 1/2 |
| 0.70 | low_light | 0.1657 | 0.1740 | +8.3 | 0/2 |
| 0.70 | texture_loss | 0.1843 | 0.2016 | +7.3 | 0/2 |
| 1.00 | motion_blur | 0.1720 | 0.1735 | -6.2 | 1/2 |
| 1.00 | low_light | 0.1839 | 0.1853 | +4.1 | 1/2 |
| 1.00 | texture_loss | 0.1813 | 0.2097 | +14.1 | 0/2 |

Reading: there is **no severity trend**; the win/loss pattern flips between classes and none of the per-cell differences separate from a 2-scene noise floor.

## 7. Real footage (TUM RGB-D, v2 data)

Sequences: fr1_xyz, fr1_desk; 90 frames; fault severity 0.6 (v2 data).

| Fault | Baseline ATE | Single ATE (Δ) | Multi ATE (Δ) |
|---|---|---|---|
| clean | 0.0397 | 0.0424 (+9.2%) | 0.0421 (+8.2%) |
| motion_blur | 0.0915 | 0.1346 (+49.8%) | 0.1346 (+49.8%) |
| low_light | 0.0610 | 0.0630 (+7.4%) | 0.0640 (+11.0%) |
| texture_loss | 0.0575 | 0.0712 (+16.9%) | 0.0742 (+20.1%) |

On real video the agent is net-negative on ATE in every fault class
(the v2 measurement; the v3 guards were built from exactly these failures,
including the proxy trap where inlier ratio rose while ATE fell apart).

## 8. Machine and component timings

Host: Windows 11, 12 logical cores, 16.8 GB RAM, Python 3.14.5, OpenCV 5.0.0.

Component timings (`aegis.bench`, p50/p95 per OpenCV call) are in `eval/bench.json`; the per-component breakdown is summarised in the technical report §6.2.

Server benchmark (`scripts/server_bench.py`, same host, one run; latencies are local-loopback, x86-only):

* FastAPI startup → first `200 /healthz`: **9.01 s**.
* End-to-end supervised job: **30.73 s**, status `done`, trusted coverage 1.0 (the run predates `ate_all` in the JSON).
* Concurrency probe: **48/48 ok** in 0.059 s (814.1 req/s).
* MCP stdio: spawn → `initialize` **340 ms**; destructive `rebootstrap` refused: **True**. The run advertised 6 tools (pre-`abstain`); the server now advertises 7 and was not re-benched.

Endpoint medians/p95 are in `eval/server_bench.json`.

## 9. Honest limitations

* **n = 1 scene** per cell (dev pilot); no CI, no p-value, no significance. The locked split (seeds 10–17) was not run — see `PREREG.md` §8.
* The synthetic faults are image transforms, not a camera in a real room; the TUM runs are 2 sequences at one severity.
* `ate_all` uses hold-last imputation across pose gaps, which favours neither condition but is not a metric of lost tracking on its own — read it with trusted coverage and the censoring count.
* Wall times were measured under 8–10-way parallel load and are informational; the same 30-frame job has varied 10.2–30.7 s across invocations on this host.
* The severity sweep and TUM tables are v2 data; the supervisor changed in v3 (adaptive thresholds, persistence, trial, multi-criteria verify, natural-recovery control). They are labelled, not re-labelled as v3.
