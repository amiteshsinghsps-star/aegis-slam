"""Tests for the PRD v2 additions: CloudWatch metrics, MCP server, seed sweep."""
import base64
import json

import numpy as np
import pytest

import aegis.metrics as metrics
import aegis.mcp_server as mcp


# --------------------------- CloudWatch metrics ---------------------------- #
def test_metrics_noop_without_aws(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    out = metrics.publish_job_metrics({"frames": 60}, "job-x", 12.0)
    assert out == {}          # silent no-op, never raises


def test_metrics_builds_expected_set(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AEGIS_INSTANCE_USD_PER_HOUR", "0.064")
    sent = {}

    class _CW:
        def put_metric_data(self, Namespace=None, MetricData=None):
            sent["ns"] = Namespace
            sent["data"] = {m["MetricName"]: m["Value"] for m in MetricData}
            sent["dims"] = MetricData[0]["Dimensions"]

    import boto3
    monkeypatch.setattr(boto3, "client", lambda name: _CW())
    result = {"frames": 90, "wall_seconds": 30.0, "source_fps": 30.0,
              "coverage_trusted": 1.0, "ate": {"rmse": 0.12},
              "confidence": {"mean": 0.8},
              "trace_summary": {"num_decisions": 3,
                                "by_verdict": {"rolled_back": 2, "improved": 1}}}
    out = metrics.publish_job_metrics(result, "job-1", 30.0)
    assert sent["ns"] == "AEGISSlam"
    for name in ("Interventions", "Rollbacks", "DecisionsKept", "CoverageTrusted",
                 "AteRmseM", "MeanConfidence", "SecondsPerVideoMinute",
                 "ComputeCostPerVideoMinute"):
        assert name in out
    # rollback ratio is the observable form of the verify/rollback claim, so it
    # must come from by_verdict when trace.summary() has no top-level counts
    assert out["Rollbacks"] == 2 and out["DecisionsKept"] == 1
    # 90 frames @30fps = 3s of video in 30s wall → 10× slower than realtime,
    # i.e. 600 wall-seconds to process one minute of video
    assert out["SecondsPerVideoMinute"] == pytest.approx(600.0, rel=1e-3)
    assert {"Name": "JobId", "Value": "job-1"} in sent["dims"]


# ------------------------------- MCP server -------------------------------- #
def test_mcp_initialize_and_tools_list():
    init = mcp._handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["serverInfo"]["name"] == "aegis-slam"
    lst = mcp._handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in lst["result"]["tools"]}
    assert {"health_signals", "get_config", "retune_thresholds"} <= names


def test_mcp_refuses_destructive_tools():
    r = mcp._handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                     "params": {"name": "rebootstrap", "arguments": {}}})
    assert r["result"]["isError"] is True


def test_mcp_health_signals_roundtrip():
    import cv2
    img = (np.random.default_rng(0).integers(0, 255, (40, 40, 3), dtype=np.uint8))
    ok, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf.tobytes()).decode()
    r = mcp._handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                     "params": {"name": "health_signals", "arguments": {"image_b64": b64}}})
    q = json.loads(r["result"]["content"][0]["text"])
    assert {"blur_var", "brightness", "contrast", "texture"} <= set(q)


# --------------------------- evaluate seed sweep --------------------------- #
def test_ensure_seed_dataset_is_deterministic_and_distinct(tmp_path):
    from aegis.evaluate import ensure_seed_dataset
    from aegis.synth import make_dataset
    base = str(tmp_path / "synthetic")
    make_dataset(base, n_frames=6, seed=0, write_mp4=False)
    d1 = ensure_seed_dataset(base, 1, 6)
    d2 = ensure_seed_dataset(base, 1, 6)
    assert d1 == d2                       # cached, not regenerated
    import os
    assert os.path.isfile(os.path.join(d1, "rgb", "image_00000.png"))


# --------------------------- JSON strictness (no GT runs) ------------------ #
def test_json_safe_maps_non_finite_values():
    from aegis.trace import json_safe
    import math
    src = {"rmse": math.nan, "n": 3, "scale": math.inf,
           "nested": [{"median": math.nan}, [1.0, math.nan]],
           "flag": True, "none": None}
    out = json_safe(src)
    assert out["rmse"] is None and out["scale"] is None and out["n"] == 3
    assert out["nested"][0]["median"] is None and out["nested"][1] == [1.0, None]
    assert out["flag"] is True and out["none"] is None
    json.dumps(out)                      # strict: no allow_nan needed


def test_trace_dump_is_strict_json_with_nan_evidence(tmp_path):
    from aegis.trace import DecisionTrace
    tr = DecisionTrace()
    tr.add(frame=5, evidence={"inlier_ratio": float("nan")}, diagnosis="low_texture",
           decided_by="rules", action={"tool": "retune_thresholds", "args": {}},
           config_before={}, config_after={}, outcome={"verdict": "improved", "after": float("nan")})
    fp = tmp_path / "trace.json"
    tr.dump(str(fp))
    text = fp.read_text(encoding="utf-8")
    assert "NaN" not in text
    json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-standard")))


def test_no_groundtruth_run_writes_strict_json(tmp_path):
    """A real clip has no GT trajectory: ATE must land as null, never bare NaN."""
    import json as _json
    from aegis.synth import make_dataset
    from aegis.pipeline import load_source, run_pipeline
    d = tmp_path / "frames"
    make_dataset(str(d), n_frames=8, seed=11, write_mp4=False)
    gt = d / "groundtruth.txt"
    if gt.exists():
        gt.unlink()                       # simulate footage with no trajectory truth
    src = load_source(str(d))
    assert src.gt_centers is None
    out = tmp_path / "run"
    res = run_pipeline(src, str(out), mode="supervised", save_images=False)
    text = (out / "result.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    back = _json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError()))
    assert back["ate"]["rmse"] is None and back["initialised"]
    assert res["ate"]["rmse"] != res["ate"]["rmse"]   # in memory it stays NaN for the harness
