# AEGIS-SLAM — Architecture

**AEGIS** = Adaptive Estimation via Guarded Intelligent Supervision. One
sentence: a standard monocular feature-based SLAM core, wrapped in a bounded
agent that measures *why* tracking is failing, takes one reversible action,
verifies it on the next window, and records every step in an auditable trace.

This is the only architecture document. Measured results live in
[`BENCHMARK.md`](BENCHMARK.md); the full method and retraction history live in
[`technical_report.md`](technical_report.md).

---

## 1. Control flow

```
frames ──► SLAM core (OpenCV 5) ──► poses + map ──► trajectory / artifacts
              │  ▲
   health     ▼  │  bounded tools mutate one RuntimeConfig
 signals ──► AGENT SUPERVISOR
             observe → diagnose → decide → act → verify → rollback
                 │                              │
                 ▼                              ▼
          decision trace.json  ◄────── approval gate (human, live)
```

There is exactly one direction of authority: evidence goes up, actions go down.
No component can bypass the tool registry, and no action is considered
successful without a later measurement.

### Layers

| Layer | Responsibility | Code |
|---|---|---|
| L1 ingress | FastAPI job service, web UI, presigned S3 uploads, judge mode | `aegis/server.py`, `web/` |
| L2 SLAM kernel | ORB/SIFT/ALIKED features, H-vs-F bootstrap, PnP+RANSAC, keyframes, triangulation, best-effort local BA | `slam/` (reused) |
| L3 perception | per-frame blur, brightness, contrast, texture, feature count, PnP inliers/ratio, reprojection error, landmarks, lost flag, confidence | `aegis/health.py` |
| L4 supervision | single rule agent or five-agent blackboard panel | `aegis/supervisor.py`, `aegis/multi.py` |
| L5 guarded act | fixed tool registry over `RuntimeConfig`, with hard rails | `aegis/tools.py` |
| L6 verify/rollback | next-window measurement, multi-criteria guards, config/map restore | `aegis/supervisor.py`, `aegis/pipeline.py` |
| L7 audit/cloud | JSONL-style trace, live feed, CloudWatch metrics, Graviton deployment | `aegis/trace.py`, `aegis/metrics.py`, `deploy/` |

---

## 2. Observe

For every frame the pipeline records two families of signals.

* **Image quality** (pure OpenCV): variance of the Laplacian (blur), mean
  brightness, standard deviation (contrast), Sobel gradient energy (texture),
  detector keypoint count.
* **Tracking health** (from the SLAM core): PnP inliers and candidates, inlier
  ratio, mean reprojection error, tracked landmark count, frame-to-frame
  matches, keyframe flag, tracking-lost flag.

The supervisor aggregates a sliding window (default 20 frames, evaluated every
10 frames). Image statistics use medians — a short burst should not move the
median — while tracking statistics use means. The composite score is

```
health_score = 0.6 · min(inlier_ratio / 0.5, 1) + 0.4 · min(landmarks / 120, 1)
```

Each frame also gets a confidence in `[0,1]` from the same tracking evidence
(0 while lost or uninitialised). Confidence is written to `result.json`,
`health_timeline.json`, `trajectory.npz`, and the trajectory plot, so downstream
consumers can gate on trust instead of accepting every pose.

### Self-calibrated thresholds

Absolute cut-offs tuned on one camera do not transfer. During the first healthy
windows of each run the supervisor records reference blur, brightness,
contrast, feature count and inlier ratio, then sets the failure cut-offs as
bounded fractions of those references (for example, the geometry floor is 40 %
of the clean reference inlier ratio, clamped to `[0.12, 0.30]`). The fixed
values remain the cold-start fallback and the bounds. This is what stops healthy
synthetic footage — whose natural inlier ratio is near 0.25 — from being
diagnosed as repetitive texture.

---

## 3. Diagnose

Ordered rules, first match wins. `ratio` means window-mean PnP inlier ratio.
Thresholds marked *adaptive* are the calibrated values described above.

| Diagnosis | Evidence | Action |
|---|---|---|
| `corrupted_input` | ≥3 lost frames **and** ≥50 % of the window is near-black (brightness <40, blur var <2) | `mark_segment_untrusted` — wait, never re-bootstrap |
| `tracking_lost_unrecoverable` | after 2 re-bootstraps or 5 lost windows, with the counters recorded as evidence | `request_recapture` (human approval); `abstain` if the evidence allowlist is not met |
| `tracking_lost_persistent` | ≥3 lost frames in two consecutive windows | `rebootstrap` (map snapshot first; verified on map size) |
| `motion_blur` | blur var < adaptive blur floor **and** (ratio < adaptive geometry floor **or** inliers < adaptive floor) | `abstain` — sharpen lifted the proxy but worsened measured ATE |
| `low_light` | brightness < adaptive dark floor **or** contrast < adaptive contrast floor, with <800 features | `apply_preprocessing(clahe)` |
| `textureless` | feature count < adaptive feature floor | `set_feature_backend(sift)`, else mark untrusted |
| `dynamic_or_repetitive` | >600 features, no lost frames, ratio < adaptive geometry floor | `retune_thresholds(usac, tighter px)` |
| `weak_geometry` | ratio still low after USAC | `retune_thresholds(+2000 max_features)` |
| healthy | health score ≥ 0.6 | restore the baseline config (false-intervention suppression) |

Two gates sit in front of the ladder:

* **Persistence / hysteresis.** Image-only diagnoses must recur in two
  consecutive failure windows before they may change the configuration. A
  single noisy window is not evidence. The counter belongs to the *episode*,
  not the label: blur often presents as low inlier ratio in one window and as
  motion blur in the next, and both are the same unresolved fault.
* **Trial before commit.** For a preprocessing proposal the agent applies the
  transform to the latest frame and recomputes the signal the diagnosis is
  about. If the trial does not improve it by at least 5 %, the agent records
  `abstain` and leaves the configuration untouched.

---

## 4. Act — the bounded tool registry

Every tool mutates one `RuntimeConfig` that the pipeline reads each frame.

| Tool | Effect | Reversible? |
|---|---|---|
| `set_feature_backend` | ORB / SIFT / ALIKED+LightGlue | yes; a descriptor-family change snapshots the map and can reinstate it |
| `apply_preprocessing` | CLAHE (LAB), NL-means denoise, unsharp mask | yes |
| `retune_thresholds` | RANSAC px, PnP min-inliers, feature cap, estimator `ransac → usac_accurate → usac_magsac` | yes |
| `rebootstrap` | drop the map and reinitialise | yes; a map snapshot is taken first and restored if verification fails |
| `mark_segment_untrusted` | tag `[start,end]` frames as untrusted | not undone (the segment stays distrusted) |
| `request_recapture` | ask a human; without approval the segment is marked untrusted | human-gated |
| `abstain` | deliberate no-op, recorded with a reason | n/a |

Hard rails: a 20-frame cooldown between actions, at most 6 backend switches per
minute of video, re-capture never executed autonomously, and no tool that can
delete data or stop the run.

### USAC

OpenCV 5's USAC estimators are the key tool for degenerate geometry, not a
headline feature. `USAC_ACCURATE` (unified sampling + ACS + MLESAC) is used for
repetitive or dynamic texture; `USAC_MAGSAC` is threshold-free and is the
second-line fix when blur makes the epipolar noise scale uncertain. On the
`opencv-python 5.0.0.93` wheel the `cv2.usac` module and `createUSAC*` factories
do not exist; the supported method constants are used, with a `cv2.error`
fallback to RANSAC. `GET /version` reports which variants the build provides.

---

## 5. Verify and roll back

Each verifiable action declares the metric it should improve and its
pre-action value. At the next window:

1. **Primary criterion.** Improvement ≥3 % relative keeps the change.
2. **Guard criteria.** A primary win cannot hide a regression: if health score
   drops by more than 0.10 or lost frames increase by more than one, the action
   is rolled back even when the primary metric improved.
3. **Natural-recovery control.** If the metric was already improving at the
   same rate before the action, the verdict is `inconclusive`, not `improved`.
   The config is kept, but the run does not claim credit for a recovery the
   trend already predicted.
4. **Rollback.** Everything else restores the config snapshot; a failed
   re-bootstrap also restores the map snapshot.

Ablation mode (`ablation`) performs the same actions with verification
disabled. That is the causal control for the verification layer: it measures
what the same policy costs without the undo.

---

## 6. Multi-agent supervision

`supervised_multi` replaces the single rule engine with five roles over a
shared blackboard. Every vote and veto is written to the trace.

| Agent | Role |
|---|---|
| Perception | aggregates the window and exposes derived facts (e.g. the corrupted-frame ratio that medians hide) |
| Diagnosis | the reproducible rule ladder, emitting scored proposals |
| Bedrock planner | optional second opinion, accepted only if it validates against the tool schema |
| Critic | re-scores outcomes; on a failed verify it queues the alternative fix (0.95, outranking a repeat of the primary); blur-aware swap to MAGSAC; thrash guard after two strikes |
| Safety officer | vetoes unsafe proposals even when rules suggested them (re-bootstrap during a corrupt burst, backend cap, distrust of a healthy scene) |
| Arbiter | ranks by confidence, takes the highest non-vetoed proposal; with no safe action it abstains unless corrupt evidence justifies distrust |

The single-agent path remains the A/B baseline. Both paths share the same
calibration, persistence gate, trial, verification guards and map rollback, so
differences in the benchmark come from the deliberation layer, not from a
different safety policy.

---

## 7. Audit and service

Every decision records: `decision_id`, frame, compact OpenCV evidence,
diagnosis, `decided_by` (`rules`, `bedrock`, `critic`, `safety`, `diagnosis`),
action and arguments, config before/after, and outcome verdict. The trace is
written incrementally and streamed to the web UI.

The FastAPI service exposes:

| Method & path | Purpose |
|---|---|
| `GET /healthz` | liveness |
| `GET /version` | OpenCV build, Graviton/IMDS facts, USAC availability |
| `GET /api/config` | defaults, modes, scenarios |
| `POST /jobs` | start a job (`mode`, `fault`, `dataset` or video ≤200 MB, token) |
| `GET /jobs` · `GET /jobs/{id}` | list / status, progress, trace summary |
| `GET /jobs/{id}/trace` | decision trace (live tail) |
| `POST /jobs/{id}/decisions/{decision_id}` | approve/deny a re-capture |
| `GET /jobs/{id}/result` | result JSON, including Sim(3)-aligned trajectory |
| `GET /jobs/{id}/frame/{n}` | evidence frame JPEG |
| `POST /api/presign-upload` | presigned S3 PUT for a private bucket |
| `GET /preloaded` · `GET /preloaded/{run}` | recorded fallback runs for judging |
| `GET /` · `GET /docs` | web UI and OpenAPI |

Judge mode is read-only: uploads are blocked, arbitrary dataset paths are
rejected, concurrency is capped, and a shared token protects mutating routes.

---

## 8. Deployment

Multi-arch Docker on Graviton4 (`c8g.large`), built for arm64. Secrets live in
SSM Parameter Store as SecureString and are rendered into a root-owned `.env`
at boot; nothing sensitive is in the repository. S3 is private with a one-day
lifecycle and uploads use presigned URLs. The security group opens only the
owner's IP and fails closed. Administration uses SSM Session Manager, not SSH.
A Budgets alarm stops the instance when spend crosses the cap. See
[`deploy/README.md`](../deploy/README.md).

---

## 9. Invariants and known limits

**Invariants**

* Same dataset and mode produce identical outputs: seeds control scene
  generation, faults are deterministic, and the pipeline has no wall-clock
  decisions.
* A destructive action is always preceded by a snapshot; a failed verification
  restores it.
* Reported ATE is Sim(3)/Umeyama RMSE on camera centres, paired with the
  ground-truth frame of each pose.
* The verification layer never treats an unverified action as a success; the
  ablation mode exists to measure the difference.

**Limits**

* Synthetic scenes share one camera-path family; only texture and phase vary by
  seed. Results are not evidence of generalization to real camera motion.
* CLIP/ALIKED model downloads and local BA are best-effort; the code falls back
  to classical features and no-BA operation.
* The online proxy metric (inlier ratio, map size) is not accuracy. A retune can
  pass verification and still worsen final ATE; the benchmark reports that case
  rather than hiding it.
* Runtime is seconds per video frame on commodity CPUs, not real-time.
* No loop closure, relocalisation or dynamic-object segmentation.
