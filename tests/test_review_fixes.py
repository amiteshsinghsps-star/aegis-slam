"""Regression tests for issues found in the external code review."""
import json, math, os
import numpy as np
import pytest


def test_features_utils_imports_without_torch():
    import slam.core.features_utils as f          # used to raise AttributeError when torch is absent
    assert hasattr(f, "HAS_LIGHTGLUE")


def test_bedrock_planner_uses_system_param_and_parses_fenced_json():
    from aegis.llm import BedrockPlanner
    from aegis.supervisor import window_stats
    from aegis.tools import RuntimeConfig
    calls = {}

    class Fake:
        def converse(self, **kw):
            calls.update(kw)
            return {"output": {"message": {"content": [{"text": "```json\n{\"tool\": \"apply_preprocessing\", \"args\": {\"mode\": \"clahe\"}}\n```"}]}}}

    p = BedrockPlanner(model_id="m", enabled=True)
    p._client = Fake()
    out = p.propose(window_stats([]), "retune_thresholds", {}, RuntimeConfig())
    assert out == {"tool": "apply_preprocessing", "args": {"mode": "clahe"}}
    assert "system" in calls and all(m["role"] != "system" for m in calls["messages"])


def test_llm_choice_bounds_and_destructive_tools_rejected():
    from aegis.llm import _validate
    assert not _validate({"tool": "retune_thresholds", "args": {"max_features": 10**9}})
    assert not _validate({"tool": "retune_thresholds", "args": {"ransac_thresh": -1}})
    assert not _validate({"tool": "rebootstrap", "args": {}})
    assert _validate({"tool": "retune_thresholds", "args": {"max_features": 6000}})


def test_metrics_drop_nan_and_scale_percent(monkeypatch):
    import sys, types
    sent = {}

    class CW:
        def put_metric_data(self, Namespace, MetricData):
            sent["data"] = MetricData

    fake = types.SimpleNamespace(client=lambda *a, **k: CW())
    monkeypatch.setitem(sys.modules, "boto3", fake)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    from aegis.metrics import publish_job_metrics
    res = {"ate": {"rmse": float("nan")}, "coverage_trusted": 0.926, "frames": 90,
           "trace_summary": {"num_decisions": 2}, "confidence": {"mean": 0.5}}
    publish_job_metrics(res, "job-x", 3.0)
    vals = {d["MetricName"]: d["Value"] for d in sent["data"]}
    assert "AteRmseM" not in vals                       # NaN would make CloudWatch reject the whole batch
    assert all(math.isfinite(v) for v in vals.values())
    assert abs(vals["CoverageTrusted"] - 92.6) < 1e-6   # unit is Percent (0-100)


def test_video_loader_caps_frames_and_width(tmp_path, monkeypatch):
    import cv2
    from aegis.pipeline import load_source
    p = str(tmp_path / "v.avi")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
    for i in range(40):
        w.write(np.full((720, 1280, 3), i * 5 % 255, np.uint8))
    w.release()
    monkeypatch.setenv("AEGIS_MAX_FRAMES", "12")
    monkeypatch.setenv("AEGIS_MAX_WIDTH", "640")
    src = load_source(p)
    assert len(src.frames) == 12 and src.frames[0].shape[1] == 640


def test_supervisor_skips_aliked_when_unavailable(monkeypatch):
    import aegis.supervisor as sv
    from aegis.tools import RuntimeConfig
    monkeypatch.setattr(sv, "_aliked_available", lambda: False)
    s = sv.Supervisor(RuntimeConfig())
    w = sv.WindowStats(n=20, frame_lo=0, frame_hi=19, blur_var=10.0, brightness=120, contrast=40,
                       feature_count=900, inlier_ratio=0.1, tracked_landmarks=100)
    diag = s._diagnose(w)
    assert diag[0] == "motion_blur" and diag[1] == "abstain"


def test_shell_scripts_have_lf_endings():
    import glob
    here = os.path.dirname(os.path.dirname(__file__))
    for f in glob.glob(os.path.join(here, "deploy", "*.sh")):
        assert b"\r\n" not in open(f, "rb").read(), f
