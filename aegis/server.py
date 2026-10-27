"""FastAPI job service (FR-7, FR-10) + version/health endpoints.

Endpoints
  POST /jobs                create a job (dataset path or uploaded video) → job id
                            form fields: mode, fault (clean|motion_blur|low_light|
                            texture_loss|frame_drop), dataset, file, token
  GET  /jobs/{id}           status, progress, health summary
  GET  /jobs/{id}/trace     full decision trace (live)
  POST /jobs/{id}/decisions/{decision_id}  approve/skip a re-capture request
  GET  /jobs/{id}/result    trajectory + map + report
  GET  /jobs/{id}/frame/{n} evidence frame snapshot
  GET  /version             OpenCV version + build info + instance type (proof)
  GET  /healthz             liveness
  GET  /                    the web UI

Runs locally or on AWS Graviton. Set AEGIS_S3_BUCKET to enable presigned
S3 uploads (see deploy/). Everything degrades to local disk when AWS is absent.
"""
from __future__ import annotations

import json
import os
import platform
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.client import HTTPConnection
from typing import Dict, List, Optional

import cv2
import numpy as np

from fastapi import HTTPException

from aegis import version as _pkg_version
from aegis.faults import FAULTS, FaultPlan, FaultSegment
from aegis.pipeline import load_source, run_pipeline, FrameSource
from aegis.tools import RuntimeConfig
from aegis.supervisor import SupervisorParams
from aegis.trace import DecisionTrace, json_safe
from aegis.approval import ApprovalGate


# --------------------------------------------------------------------------- #
#  Instance / version helpers (FR-10)
# --------------------------------------------------------------------------- #
def _imds(path: str, timeout: float = 0.4) -> Optional[str]:
    """Best-effort EC2 metadata read over IMDSv2 only (the launch template
    enforces HttpTokens=required); None off EC2 or if the token is refused."""
    try:
        c = HTTPConnection("169.254.169.254", timeout=timeout)
        c.request("PUT", "/latest/api/token", headers={"X-aws-ec2-metadata-token-ttl-seconds": "30"})
        tok = c.getresponse().read().decode()
        if not tok or "<" in tok:
            return None                      # no token => v2-only mode, stop here
        c.request("GET", path, headers={"X-aws-ec2-metadata-token": tok})
        v = c.getresponse().read().decode()
        return v if v and "<" not in v else None
    except Exception:
        return None


def _ec2_instance_type(timeout: float = 0.4) -> Optional[str]:
    return _imds("/latest/meta-data/instance-type", timeout)


def version_info() -> dict:
    itype = _ec2_instance_type()
    arch = platform.machine().lower()
    from aegis.robust_geometry import usac_support
    return {
        "opencv_version": cv2.__version__,
        "opencv_build_info": cv2.getBuildInformation(),
        "robust_estimators": usac_support(),
        "numpy_version": np.__version__,
        "aegis_version": _pkg_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "arch": arch,
        "is_arm_graviton": arch in ("aarch64", "arm64"),
        "ec2_instance_type": itype,
        "ec2_instance_id": _imds("/latest/meta-data/instance-id") if itype else None,
        "aws_region": (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
                       or (_imds("/latest/meta-data/placement/region") if itype else None)),
        # --- which AWS integrations are actually live on this host ---------
        "cloud": {
            "imdsv2_only": bool(itype),
            "cloudwatch_namespace": "AEGISSlam" if itype else None,
            "s3_presign_enabled": bool(os.environ.get("AEGIS_S3_BUCKET")),
            "s3_bucket": os.environ.get("AEGIS_S3_BUCKET") or None,
            "bedrock_planner": os.environ.get("AEGIS_BEDROCK_MODEL") or "off",
            "cost_metric_usd_per_hour": os.environ.get("AEGIS_INSTANCE_USD_PER_HOUR") or None,
            "judge_mode": _JUDGE_MODE,
        },
    }


# --------------------------------------------------------------------------- #
#  Job store
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    mode: str
    fault: str = "clean"
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    frame: int = 0
    total: int = 0
    source: str = ""
    out_dir: str = ""
    created: float = field(default_factory=time.time)
    error: Optional[str] = None
    result: Optional[dict] = None
    trace: DecisionTrace = field(default_factory=DecisionTrace)
    gate: ApprovalGate = field(default_factory=lambda: ApprovalGate(default_timeout_s=30.0))
    evidence_dir: str = ""


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, mode: str, source: str, out_dir: str, fault: str = "clean") -> Job:
        j = Job(id="job-" + uuid.uuid4().hex[:8], mode=mode, source=source,
                out_dir=out_dir, fault=fault,
                evidence_dir=os.path.join(out_dir, "frames"))
        os.makedirs(j.evidence_dir, exist_ok=True)
        with self._lock:
            self._jobs[j.id] = j
        return j

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> List[dict]:
        with self._lock:
            return [{"id": j.id, "mode": j.mode, "fault": j.fault, "status": j.status,
                     "progress": j.progress, "created": j.created}
                    for j in self._jobs.values()]


STORE = JobStore()
_RUNS_DIR = os.environ.get("AEGIS_RUNS", "runs/server")
_DEFAULT_DATASET = os.environ.get("AEGIS_DATASET", "Dataset/synthetic")
_DATA_ROOT = os.path.abspath(os.environ.get("AEGIS_DATA_ROOT", "Dataset"))
_MAX_UPLOAD_BYTES = int(os.environ.get("AEGIS_MAX_UPLOAD_MB", "200")) * 1024 * 1024
_SCENARIOS = ("clean",) + tuple(FAULTS)


def _fault_plan(fault: Optional[str], n_frames: int) -> Optional[FaultPlan]:
    """Inject the chosen scenario on frames 35-65%, the same window the
    benchmark harness uses, so live runs and the report are comparable."""
    if not fault or fault == "clean" or n_frames <= 0:
        return None
    lo, hi = int(n_frames * 0.35), int(n_frames * 0.65)
    return FaultPlan([FaultSegment(fault, lo, hi, severity=1.0)])

# Judge access mode (PRD U2 / G3): read-only demo that only ever runs the
# built-in synthetic dataset, blocks uploads & arbitrary paths, and caps
# concurrency so a public :8000 during judging can't be overrun.
_JUDGE_MODE = os.environ.get("AEGIS_JUDGE_MODE") == "1"
_JUDGE_MAX_RUNNING = int(os.environ.get("AEGIS_JUDGE_MAX_RUNNING", "2"))
_JUDGE_TOKEN = os.environ.get("AEGIS_JUDGE_TOKEN", "")


def _safe_dataset_path(path: str) -> str:
    """Restrict caller-supplied dataset paths to the data root (no LFI)."""
    p = os.path.abspath(path)
    root = os.path.abspath(_DEFAULT_DATASET)
    if not (p == _DATA_ROOT or p.startswith(_DATA_ROOT + os.sep)
            or p == root or p.startswith(root + os.sep)
            or p == os.path.abspath(_RUNS_DIR)
            or p.startswith(os.path.abspath(_RUNS_DIR) + os.sep)):
        raise HTTPException(400, "dataset path must be under the data root")
    if not os.path.exists(p):
        raise HTTPException(404, "dataset not found")
    return p


def _run_job(job: Job):
    job.status = "running"
    t0 = time.time()

    def progress_cb(frame, total):
        job.frame = frame
        job.total = total
        job.progress = round(100.0 * frame / max(1, total), 1)

    try:
        source = load_source(job.source)
        cfg = RuntimeConfig()
        sup = SupervisorParams()
        job.trace = DecisionTrace()
        job.gate = ApprovalGate(default_timeout_s=30.0)
        plan = _fault_plan(job.fault, len(source.frames))
        res = run_pipeline(source, job.out_dir, mode=job.mode, cfg=cfg,
                           fault=plan, sup_params=sup, save_images=True,
                           trace=job.trace, gate=job.gate, progress_cb=progress_cb)
        # save evidence frames for re-capture requests already resolved
        job.result = res
        job.status = "done"
        job.progress = 100.0
        from aegis.metrics import publish_job_metrics
        publish_job_metrics(res, job.id, time.time() - t0)
    except Exception as e:
        import traceback
        job.status = "error"
        job.error = f"{e}\n{traceback.format_exc()[-800:]}"


# --------------------------------------------------------------------------- #
#  App
# --------------------------------------------------------------------------- #
def create_app():
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
    from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="AEGIS-SLAM", version=_pkg_version())

    web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
    if os.path.isdir(web_dir):
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "opencv": cv2.__version__}

    @app.get("/version")
    def version():
        return version_info()

    @app.get("/api/config")
    def api_config():
        return {"default_dataset": _DEFAULT_DATASET,
                "modes": ["baseline_classical", "baseline_learned", "supervised", "supervised_multi", "ablation"],
                "scenarios": list(_SCENARIOS),
                "runs_dir": _RUNS_DIR,
                "judge_mode": _JUDGE_MODE}

    @app.post("/jobs")
    def create_job(mode: str = Form("supervised"),
                   fault: str = Form("clean"),
                   dataset: Optional[str] = Form(None),
                   file: Optional[UploadFile] = File(None),
                   token: Optional[str] = Form(None)):
        mode = mode or "supervised"
        if mode not in {"baseline_classical", "baseline_learned", "supervised", "supervised_multi", "ablation"}:
            raise HTTPException(400, "unknown mode")
        fault = fault or "clean"
        if fault not in _SCENARIOS:
            raise HTTPException(400, f"unknown scenario (expected one of {', '.join(_SCENARIOS)})")
        if _JUDGE_MODE:
            if _JUDGE_TOKEN and token != _JUDGE_TOKEN:
                raise HTTPException(401, "judge mode requires the shared demo token")
            if file is not None:
                raise HTTPException(403, "judge mode: uploads disabled, use the built-in dataset")
            if dataset and os.path.abspath(dataset) != os.path.abspath(_DEFAULT_DATASET):
                raise HTTPException(403, "judge mode: only the built-in dataset is allowed")
            running = sum(1 for j in STORE.list() if j["status"] in ("queued", "running"))
            if running >= _JUDGE_MAX_RUNNING:
                raise HTTPException(429, "judge mode: demo is busy, try again shortly")
            dataset = None
        if file is not None:
            job_id_hint = "upload-" + uuid.uuid4().hex[:6]
            up_dir = os.path.join(_RUNS_DIR, "uploads")
            os.makedirs(up_dir, exist_ok=True)
            suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
            src_path = os.path.join(up_dir, job_id_hint + suffix)
            _validate_ext(file.filename)
            total = 0
            try:
                with open(src_path, "wb") as f:
                    while True:
                        chunk = file.file.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > _MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                413,
                                f"upload exceeds {os.environ.get('AEGIS_MAX_UPLOAD_MB', '200')}MB")
                        f.write(chunk)
            except HTTPException:
                if os.path.exists(src_path):
                    os.remove(src_path)
                raise
            if total == 0:
                os.remove(src_path)
                raise HTTPException(400, "empty upload")
            source = src_path
        elif dataset:
            source = _safe_dataset_path(dataset)
        else:
            source = _DEFAULT_DATASET

        out_dir = os.path.join(_RUNS_DIR, uuid.uuid4().hex)
        job = STORE.create(mode, source, out_dir, fault=fault)
        threading.Thread(target=_run_job, args=(job,), daemon=True).start()
        return {"job_id": job.id, "status": job.status, "mode": mode, "fault": fault}

    @app.get("/jobs")
    def jobs():
        return {"jobs": STORE.list()}

    @app.get("/api/presign-upload")
    def presign_upload(filename: str = "video.mp4"):
        """Presigned PUT URL for direct-to-S3 upload (bucket stays private)."""
        bucket = os.environ.get("AEGIS_S3_BUCKET")
        if not bucket:
            raise HTTPException(501, "S3 not configured (set AEGIS_S3_BUCKET)")
        _validate_ext(filename)
        import boto3
        key = f"uploads/{uuid.uuid4().hex[:12]}-{os.path.basename(filename)}"
        s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION")
                          or os.environ.get("AWS_DEFAULT_REGION"))
        url = s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": key,
                    "ContentType": "video/mp4"},
            ExpiresIn=900)
        return {"upload_url": url, "key": key,
                "max_bytes": _MAX_UPLOAD_BYTES, "expires_in": 900}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        job = STORE.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        return {
            "job_id": job.id, "mode": job.mode, "fault": job.fault, "status": job.status,
            "progress": job.progress, "frame": job.frame, "total": job.total,
            "error": job.error,
            "health_summary": _health_summary(job),
            "pending_decisions": job.gate.pending(),
            "trace_summary": job.trace.summary(),
        }

    @app.get("/jobs/{job_id}/trace")
    def job_trace(job_id: str):
        job = STORE.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        return JSONResponse({"summary": job.trace.summary(), "decisions": job.trace.records})

    @app.post("/jobs/{job_id}/decisions/{decision_id}")
    def decide(job_id: str, decision_id: str, payload: dict = Body(...)):
        job = STORE.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        if _JUDGE_MODE and _JUDGE_TOKEN and payload.get("token") != _JUDGE_TOKEN:
            raise HTTPException(401, "judge mode: the shared token is required to answer approvals")
        approve = bool(payload.get("approve"))
        # audit line: who answered, and when (resolved_at), recorded on the request
        reviewer = str(payload.get("reviewer") or "web-session")[:64]
        req = job.gate.resolve(decision_id, approve, by=reviewer)
        if req is None:
            raise HTTPException(404, "no such decision")
        return req.to_dict()

    @app.get("/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = STORE.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        if job.status != "done":
            return {"status": job.status, "result": None}
        traj = np.load(os.path.join(job.out_dir, "trajectory.npz"))
        est = (traj["est_aligned"] if "est_aligned" in traj.files else traj["est"]).tolist()
        gt = traj["gt"].tolist() if traj["gt"].size else []
        return json_safe({
            "status": "done",
            "result": job.result,
            "trajectory": {"est": est, "gt": gt},
            "trace": job.trace.records,
        })

    @app.get("/jobs/{job_id}/frame/{n}")
    def job_frame(job_id: str, n: int):
        job = STORE.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        fp = os.path.join(job.evidence_dir, f"{n}.jpg")
        if os.path.isfile(fp):
            return FileResponse(fp, media_type="image/jpeg")
        # render on demand from source
        try:
            src = load_source(job.source)
            from aegis.pipeline import _read, _to_u8_bgr
            img = _to_u8_bgr(_read(src.frames[min(n, len(src.frames) - 1)]))
            ok, buf = cv2.imencode(".jpg", img)
            out = os.path.join(job.evidence_dir, f"{n}.jpg")
            buf.tofile(out)
            return FileResponse(out, media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.get("/preloaded")
    def preloaded_runs():
        """Finished runs browsable without starting a job (judge fallback)."""
        import glob
        items = []
        for pattern in (os.path.join(_RUNS_DIR, "*", "result.json"),
                        os.path.join(_RUNS_DIR, "*", "*", "result.json"),
                        "docs/demo/*/result.json"):
            for fp in sorted(glob.glob(pattern)):
                try:
                    with open(fp) as f:
                        r = json.load(f)
                except Exception:
                    continue
                run_dir = os.path.dirname(fp)
                items.append({
                    "run": os.path.basename(run_dir),
                    "mode": r.get("mode"),
                    "frames": r.get("frames"),
                    "ate_rmse": (r.get("ate") or {}).get("rmse"),
                    "coverage_trusted": r.get("coverage_trusted"),
                    "has_trajectory": os.path.exists(os.path.join(run_dir, "trajectory.npz")),
                    "has_trace": os.path.exists(os.path.join(run_dir, "trace.json")),
                })
        return {"runs": items}

    @app.get("/preloaded/{run}")
    def preloaded_run(run: str):
        """Read-only view of one finished run; run may be a name under
        docs/demo or a job id under the runs dir (no path traversal)."""
        import glob
        if "/" in run or "\\" in run or ".." in run:
            raise HTTPException(400, "bad run name")
        cands = [os.path.join(_RUNS_DIR, run), os.path.join("docs/demo", run)]
        cands += [os.path.dirname(p) for p in glob.glob(
            os.path.join(_RUNS_DIR, "*", run, "result.json"))]
        for d in cands:
            rfp = os.path.join(d, "result.json")
            if os.path.isfile(rfp):
                with open(rfp) as f:
                    result = json.load(f)
                traj = {}
                tfp = os.path.join(d, "trajectory.npz")
                if os.path.isfile(tfp):
                    t = np.load(tfp)
                    traj = {"est": (t["est_aligned"] if "est_aligned" in t.files else t["est"]).tolist(),
                            "gt": (t["gt"].tolist() if t["gt"].size else [])}
                trace = []
                xfp = os.path.join(d, "trace.json")
                if os.path.isfile(xfp):
                    with open(xfp) as f:
                        tr = json.load(f)
                    trace = tr.get("decisions", tr if isinstance(tr, list) else [])
                return {"run": run, "result": result,
                        "trajectory": traj, "trace": trace}
        raise HTTPException(404, "no such preloaded run")

    @app.get("/")
    def index():
        idx = os.path.join(web_dir, "index.html")
        if os.path.isfile(idx):
            return FileResponse(idx)
        return HTMLResponse("<h1>AEGIS-SLAM</h1><p>web/ UI not found. See /version, /jobs.</p>")

    return app


def _validate_ext(filename: Optional[str]):
    exts = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
    if filename and not filename.lower().endswith(exts):
        raise HTTPException(415, "only video files accepted")


def _health_summary(job: Job) -> dict:
    fp = os.path.join(job.out_dir, "health_timeline.json")
    if not os.path.isfile(fp):
        return {}
    with open(fp) as f:
        rows = json.load(f)
    if not rows:
        return {}
    last = rows[-10:]
    import statistics
    return {
        "frames_processed": len(rows),
        "recent_mean_inlier_ratio": round(statistics.mean(
            (r.get("pnp_inliers", 0) / r.get("pnp_candidates", 1)) if r.get("pnp_candidates") else 0 for r in last), 3),
        "recent_blur_var": round(statistics.mean(r.get("blur_var", 0) for r in last), 2),
        "recent_brightness": round(statistics.mean(r.get("brightness", 0) for r in last), 1),
        "lost_frames": int(sum(1 for r in rows if r.get("tracking_lost"))),
    }


def serve(host: Optional[str] = None, port: Optional[int] = None):
    import uvicorn
    global app
    host = host or os.environ.get("AEGIS_HOST", "0.0.0.0")
    port = port or int(os.environ.get("AEGIS_PORT", "8000"))
    if app is None:
        app = create_app()
    uvicorn.run(app, host=host, port=port)


# Module-level ASGI app for `uvicorn aegis.server:app`.
try:
    app = create_app()
except Exception:  # pragma: no cover - surfaced at startup logs
    app = None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("AEGIS-SLAM server")
    ap.add_argument("--host", default=os.environ.get("AEGIS_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("AEGIS_PORT", "8000")))
    a = ap.parse_args()
    serve(a.host, a.port)
