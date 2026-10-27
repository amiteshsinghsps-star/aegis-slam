# AEGIS-SLAM

**Adaptive Estimation via Guarded Intelligent Supervision**
Self-healing monocular visual SLAM on **OpenCV 5**, with an auditing agent supervisor,
a live web service, and AWS Graviton deployment scripts.

![OpenCV](https://img.shields.io/badge/OpenCV-5.0-green) ![Python](https://img.shields.io/badge/Python-3.12-blue) ![AWS](https://img.shields.io/badge/AWS-Graviton-orange) ![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey)

## Quick links

**Report:** `docs/technical_report.md` · **Benchmarks:** `docs/BENCHMARK.md` · **Deployment:** `deploy/README.md`

**At a glance:** observe → diagnose → act → **verify** → **rollback**, with a full audit trace.

## 1. Problem

Monocular feature-based SLAM (ORB + RANSAC + PnP) degrades **silently**.
Motion blur, low light, textureless regions and dropped frames break its assumptions,
yet the estimator keeps emitting confident poses that nothing checks.
The trajectory looks as plausible as a good one. There is no record of why it failed,
no recovery policy, and no way to know which frames to trust.

## 2. Solution

AEGIS-SLAM wraps a SLAM core in a small, auditable agent that:

1. **Observes** per-frame OpenCV health signals.
2. **Diagnoses** the failure mode with an ordered rule ladder.
3. **Acts** with one bounded tool from a fixed registry.
4. **Verifies** the effect on the next window.
5. **Rolls back** the change if it did not help, including a map-snapshot restore for
   destructive re-bootstraps.

Every decision is written to a trace and shown live in the web UI, with a human
approval gate for the one costly action (re-capture).

## 3. Features

- **Health signals** (OpenCV 5): Laplacian variance, brightness, contrast, Sobel texture
  energy, keypoint count, PnP inliers and ratio, reprojection error, tracked landmarks, lost flag
- **Diagnosis** of seven failure classes: `motion_blur`, `low_light`, `textureless`,
  `dynamic_or_repetitive`, `weak_geometry`, `tracking_lost`, `corrupted_input`
- **Six bounded tools** plus an `abstain` outcome (see §5)
- **Verify and rollback** with undoable re-bootstrap (map snapshot)
- **Safety rails:** cooldowns, backend-switch cap, persistence gate, thrash guard
- **Self-calibrated thresholds** relative to each video's own healthy baseline
- **Multi-agent mode:** Perception, Diagnosis, Critic, Safety officer, Arbiter, plus an
  optional Bedrock planner, with votes and vetoes in the trace
- **Per-frame confidence** in [0, 1], exported to JSON/NPZ and plotted on the trajectory
- **Decision trace** (`trace.json`): evidence → diagnosis → decider → action → outcome
- **Human approval gate** with evidence frame in the UI
- **Web UI:** trajectory, health timeline, decision feed, capture advice, fault scenarios
- **Judge mode:** read-only service with a shared token and recorded demo fallback
- **MCP server:** external agents read signals and call the reversible tools
- **Evaluation harness:** synthetic dataset, fault injection, ATE (Sim(3)), coverage,
  recovery, paired statistics, trust metrics, severity sweep, TUM RGB-D adapter
- **Deployment scripts:** Graviton EC2, S3, IAM, SSM, CloudWatch, Budgets, teardown

## 4. How OpenCV 5 is used

| Area | OpenCV 5 functionality |
|---|---|
| Robust estimation | `USAC_ACCURATE`, `USAC_MAGSAC` (chosen by failure mode); RANSAC fallback |
| Geometry | `findEssentialMat`, `recoverPose`, `triangulatePoints`, `solvePnPRansac` |
| Features | ORB, SIFT, `goodFeaturesToTrack`; optional ALIKED + LightGlue |
| Preprocessing | CLAHE (LAB), `fastNlMeansDenoisingColored`, unsharp mask via `addWeighted` |
| Health signals | Laplacian, Sobel, mean/std statistics |
| Fault injection | `filter2D` motion blur, gamma darkening, regional blur, blackout bursts |

The service pins `opencv-python-headless==5.0.0.93`. On that wheel there is no `cv2.usac`
module and no `AKAZE_create`; the USAC method constants are used with a RANSAC fallback.
`GET /version` reports the running OpenCV build.

## 5. Agent design

**Diagnosis ladder** (first match wins): `corrupted_input` → mark untrusted · `tracking_lost_persistent` → `rebootstrap` (undoable) · `tracking_lost_unrecoverable` → human-approved re-capture · `motion_blur` → `abstain` · `low_light` → CLAHE · `textureless` → SIFT or untrusted · `dynamic_or_repetitive` → USAC + tighter threshold · `weak_geometry` → raise feature cap.

**Tool registry:** `set_feature_backend` · `apply_preprocessing` · `retune_thresholds` ·
`rebootstrap` · `mark_segment_untrusted` · `request_recapture`

**Verify:** each action declares the metric it should improve. If the next window does not
improve it by ≥ 3 %, the previous config is restored (`rolled_back`). Ablation mode acts
without verifying, to measure what verification is worth.

## 6. How AWS is used

| Service | Purpose |
|---|---|
| EC2 Graviton4 (`c8g.large`) | Runs the service in a multi-arch Docker container |
| S3 (private) | Presigned uploads, 1-day lifecycle, public access blocked |
| IAM | Least-privilege instance role scoped to one bucket and one metric namespace |
| SSM | Session Manager instead of SSH; Parameter Store for config and secrets |
| CloudWatch | Per-job metrics, logs, dashboard |
| Budgets + EventBridge | Cost guard that stops (never deletes) the instance |
| Bedrock (optional) | Second-opinion planner, schema-validated, off by default |

Deployment scripts live in `deploy/`.

## 7. Graviton setup

```bash
# prerequisites: AWS CLI v2 configured, jq, curl
export AWS_REGION=us-east-1
export INSTANCE_TYPE=c8g.large            # Graviton4
export REPO_URL=https://github.com/amiteshsinghsps-star/aegis-slam.git
export INSTANCE_USD_PER_HOUR=0.085       # set to your region's on-demand price
export BUDGET_USD=5
bash deploy/deploy-graviton.sh            # or: make deploy

# the script prints the public DNS; open :8000/version to confirm OpenCV 5 on arm64

# teardown
CONFIRM=yes bash deploy/teardown.sh
```

**Free-plan accounts:** `c8g` may not be allowed. Use a free-tier-eligible type such as
`t4g.small` (Graviton2) with `INSTANCE_TYPE=t4g.small COST_GUARD=0`.

## 8. Local setup

```bash
# install
pip install -r requirements.txt
# optional learned backend
pip install torch "lightglue @ git+https://github.com/cvg/LightGlue.git"

# reproducible synthetic dataset
python -m aegis.synth --out Dataset/synthetic --frames 200 --no_mp4

# one supervised run with an injected fault
python -m aegis.run --data Dataset/synthetic --mode supervised \
    --fault motion_blur --out runs/demo

# web service and UI  ->  http://localhost:8000
python -m aegis.server --port 8000

# read-only judge mode
AEGIS_JUDGE_MODE=1 AEGIS_JUDGE_TOKEN=demo-token python -m aegis.server
python -m scripts.preload_demo            # record the three preset demo runs

# your own footage
python -m aegis.run --data clip.mp4 --mode supervised --out runs/real

# component timings
python -m aegis.bench --frames 120 --reps 3 --out eval/bench.json

# one-command demo
bash scripts/demo.sh                      # or: make demo
```

**Modes:** `baseline_classical` · `baseline_learned` · `supervised` · `supervised_multi` · `ablation`

## 9. Docker

```bash
docker build -t aegis-slam deploy
docker run --rm -p 8000:8000 -v "$PWD:/app" aegis-slam
# UI: http://localhost:8000    OpenCV build proof: http://localhost:8000/version
```

The image is multi-arch (amd64/arm64) with pinned headless OpenCV 5.
`deploy/docker-compose.yml` caps CPU and memory and wires the environment.

## 10. Architecture

![Architecture](docs/diagrams/architecture.svg)

```
frames ─► SLAM core (OpenCV 5) ─► poses + map ─► trajectory, artifacts
             │  ▲
 health      ▼  │  bounded tools (RuntimeConfig)
 signals ─► AGENT SUPERVISOR: observe → diagnose → act → verify → rollback
                 │                                │
                 ▼                                ▼
          decision trace.json  ◄────── human approval gate (UI)
```

**Layers:** L1 ingress · L2 SLAM kernel · L3 perception · L4 supervision · L5 guarded act · L6 verify/rollback · L7 audit + cloud.

### Agent loop and multi-agent panel

| Agent loop | Multi-agent supervision |
|---|---|
| ![Agent loop](docs/diagrams/agent_loop.svg) | ![Multi-agent](docs/diagrams/multi_agent.svg) |

### Evidence figures

![Annotated trace: motion blur](docs/diagrams/trace_walkthrough.png)
![Severity sweep](docs/diagrams/severity_sweep.png)
![Real footage (TUM RGB-D)](docs/diagrams/real_bench.png)

## 11. HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness |
| `GET /version` | OpenCV build, architecture, instance facts |
| `GET /api/config` | Defaults, modes, fault scenarios |
| `POST /jobs` | Start a job (`mode`, `fault`, dataset or uploaded video ≤ 200 MB) |
| `GET /jobs`, `/jobs/{id}` | List jobs, status and progress |
| `GET /jobs/{id}/trace` | Decision trace (live tail) |
| `POST /jobs/{id}/decisions/{decision_id}` | Approve or deny a re-capture request |
| `GET /jobs/{id}/result` | ATE, coverage, confidence, trace summary |
| `GET /jobs/{id}/frame/{n}` | Evidence frame (JPEG) |
| `POST /api/presign-upload` | Presigned S3 PUT |
| `GET /preloaded`, `/preloaded/{run}` | Recorded runs for read-only judging |

## 12. MCP server

Run `python -m aegis.mcp_server`. It exposes the reversible tools over stdio. Destructive actions (`rebootstrap`,
`request_recapture`, `mark_segment_untrusted`) are deliberately not exposed.

## 13. Benchmark

Synthetic pilot: **1 scene (seed 0), 60 frames**, faults on frames 21–39.
ΔATE vs baseline, **positive = worse**.

| Fault | Baseline ATE (m) | Single | Multi | No-verify ablation |
|---|---|---|---|---|
| motion_blur | 0.104 | +0.0 % | +0.0 % | +0.0 % |
| low_light | 0.081 | +0.0 % | +0.0 % | +0.0 % |
| texture_loss | 0.107 | +0.0 % | +0.0 % | +0.0 % |
| frame_drop | 0.070 | +9.8 % | +0.0 % | +332 % |

- Clean video: **0 false interventions** (single and multi).
- Verification prevents a large regression on `frame_drop` (+332 % without it) but does not beat baseline.
- On three faults the agent chose not to act and matched baseline.
- **n = 1:** no confidence intervals, no significance claim. The preregistered protocol
  (90 frames, 5 dev + 8 locked seeds) was not run; see `docs/PREREG.md` §8.

```bash
python -m aegis.evaluate --data Dataset/synthetic --frames 60 --seeds 0 \
    --conditions baseline_classical,supervised,supervised_multi,ablation \
    --out eval/dev --jobs 5
python scripts/stats.py --results eval/dev/results.json --out eval/dev/stats.json
python scripts/trust.py --eval eval/dev --out eval/dev/trust.json
```

## 14. Tests

```bash
pytest tests -q
# AEGIS-only subset
pytest tests/test_geom.py tests/test_health.py tests/test_supervisor.py \
       tests/test_tools_faults.py tests/test_multi.py
```

CI (`.github/workflows/ci.yml`) runs the unit tests and a 30-frame supervised smoke run.

## 15. Repository layout

`aegis/` agent, service, evaluation · `slam/` reused core · `web/` UI · `deploy/` AWS and Docker · `docs/` report and diagrams · `eval/` results · `scripts/` tools · `tests/`

## 16. Limitations

- Benchmark is a **single-scene pilot**; the locked split was not run.
- Earlier multi-scene and real-footage (TUM RGB-D) results showed the agent is **not
  net-positive on mean ATE**; its measured value is damage control.
- Verification uses **online proxy metrics**, which can pass while accuracy worsens.
- No loop closure or dynamic-object segmentation. Not real-time.
- Timings are x86 development hardware, not Graviton.

## 17. Responsible use

Research demo, **not for safety-critical navigation**. Output is confidence-gated, with
explicit untrusted segments. Uploads are capped at 200 MB, retained 1 day, and never used
for training. The only costly action (re-capture) requires a human decision.

## 18. License
Apache-2.0 — see `LICENSE` and `NOTICE`.