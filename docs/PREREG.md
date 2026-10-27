# AEGIS-SLAM — preregistration (v3 benchmark)

_Written 2026-10-27, before the locked test run. The dev runs used to tune the
supervisor are seeds 0–4 and live in `eval/dev/`. The locked run uses seeds
10–17, is executed once, and is published whatever it shows — including losses._

## 1. Question

Does the supervision layer (observe → diagnose → act → verify → rollback) reduce
trajectory error under injected faults compared with the same OpenCV-5 core run
without supervision, without paying for it with false interventions on clean
video?

## 2. System under test

| Condition | What runs |
|---|---|
| `baseline_classical` | Reused OpenCV-5 SLAM core, fixed configuration, no supervisor |
| `supervised` | Same core + single-agent supervisor (adaptive thresholds, persistence, trial, multi-criteria verify, natural-recovery control) |
| `supervised_multi` | Same core + five-role multi-agent supervisor (perception, diagnosis, critic, safety, arbiter) |
| `ablation` | Supervisor with verification disabled (never rolls back) |

`baseline_learned` exists in the harness but is not part of this preregistration
(no LightGlue weights are pinned in this run).

## 3. Data

* Synthetic clips from `aegis.synth`: 90 frames, camera trajectory and texture
  phase determined by the seed. A seed produces one scene; every condition is
  paired on that same scene.
* Fault injection (`aegis.faults.standard_scenarios`, severity 1.0): four
  injectable classes — `motion_blur`, `low_light`, `texture_loss`,
  `frame_drop` — over frames 35–65 % of the clip (frame_drop: 8-frame blackout
  burst at 35 %).
* `clean` (no fault) is the false-intervention control.
* Seven *diagnosis* classes exist in the supervisor; only the four above are
  physically injectable, so four are benchmarked. This distinction is stated in
  the README and the technical report.

## 4. Seeds and freeze

| Split | Seeds | Use |
|---|---|---|
| Dev | 0, 1, 2, 3, 4 | Tuning (threshold calibration caps, persistence length, critic alternatives) |
| Locked | 10–17 | One run after this document is frozen; no re-runs for a better result |

There is no git working copy, so the freeze is recorded as SHA-256 over the
evaluated source files (listed in `eval/locked/SOURCE_HASHES.txt`, generated
immediately before the locked run). Any post-freeze code change invalidates the
locked run and is listed in §8.

## 5. Metrics

Primary: **ATE_all** — RMSE of camera centres over *all* frames after Sim(3)
alignment, with hold-last imputation across frames without a pose. Every frame
is counted, so a tracker that gives up is penalised instead of dropped.

Secondary, always reported together:

* `ate_trusted` — RMSE over frames the system trusts, with `coverage_trusted`
  next to it (never quoted alone).
* `coverage_estimated` — frames with any pose / all frames.
* `recovered` — frames from fault start until tracking is stable for 3 frames.
  Runs that never recover are **right-censored**, not dropped; recovered-mean is
  always shown with the not-recovered count.
* Agent activity: decisions, backend switches, rollbacks, re-captures.
* Wall time and FPS (single machine, informational; machine load sensitive).

Sign conventions: **ΔATE% = (condition − baseline) / baseline × 100; positive =
worse.** Confidence AUROC > 0.5 means the confidence ranks high-error frames
higher; lower AURC is better.

## 6. Hypotheses and decision rules

| ID | Hypothesis | Pass rule |
|---|---|---|
| H1 | Supervision reduces ATE on faulted scenes | Median paired ΔATE_all < 0 with 95 % bootstrap CI upper bound < 0 in at least 3 of 4 fault classes (locked seeds, per-class n = 8) |
| H2 | No false interventions on clean video | 0 decisions in clean runs, on average across conditions with verification; any decision is reported per-run |
| H3 | Multi-agent parity | \|mean ΔATE_all(supervised_multi − supervised)\| ≤ 10 % of supervised mean per fault class; if outside, the loss is published and not hidden |
| H4 | Confidence is informative | AUROC(confidence) > AUROC(inlier ratio) and > AUROC(landmark count) on pooled faulted frames, majority of classes |
| H5 | Trusted ATE at matched coverage favours the agent | Mean paired delta (agent-gated − baseline-gated) < 0 across paired cells; reported alongside |

Statistics: paired bootstrap (10 000 resamples, percentile CI) and a two-sided
Wilcoxon signed-rank test (SciPy; exact for n ≤ 18). p-values are reported
unadjusted with the per-class test count stated; no Holm correction is claimed.

## 7. Exclusions and reporting

* A run that crashes or produces no alignment is excluded from that cell and
  counted in a "failed runs" column. It is not silently replaced.
* Censored runs stay in the table. Means over recovered runs carry the
  not-recovered count.
* Every scenario/seed cell is published, not just aggregates.
* Machine-load-sensitive wall times are labelled informational.

## 8. Deviations log

* **2026-10-27 — reduced compute budget.** The benchmark was run at **60 frames
  × seed 0** (16 faulted cells + 2 clean) instead of 90 frames × seeds 0–4, and
  the locked seeds 10–17 were **not run**. Reason: a hard wall-clock limit; the
  full protocol takes ~40 min on this machine, and parallel workers memory-thrash
  above ~5 processes (two larger attempts were abandoned mid-run and are not
  quoted). All numbers in `docs/BENCHMARK.md` come from this **pilot** (n = 1
  scene), are labelled as such, and **no significance claim is made**. The
  preregistered pass rules (n = 8 per class) are therefore **not evaluated** in
  this release; §6 stays as the plan for the full run.
