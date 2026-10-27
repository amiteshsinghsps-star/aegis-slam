import cv2
import numpy as np
import pytest

from aegis.tools import (RuntimeConfig, invoke, apply_preprocess,
                              clahe, sharpen, TOOLS)
from aegis.faults import (FaultPlan, FaultSegment, standard_scenarios,
                               motion_blur, frame_drop, FAULTS)
from aegis.health import blur_score, brightness_score


# --------------------------------------------------------------------------- #
#  Tool registry
# --------------------------------------------------------------------------- #
def test_invoke_unknown_tool():
    r = invoke("nope", RuntimeConfig(), {}, 0)
    assert r.ok is False


def test_apply_preprocessing_tool_sets_mode():
    cfg = RuntimeConfig()
    r = invoke("apply_preprocessing", cfg, {"mode": "clahe"}, 10)
    assert r.ok and r.changed and cfg.preprocess == "clahe"
    r2 = invoke("apply_preprocessing", cfg, {"mode": "bogus"}, 11)
    assert r2.ok is False


def test_retune_thresholds_tool():
    cfg = RuntimeConfig()
    r = invoke("retune_thresholds", cfg,
               {"ransac_thresh": 1.8, "robust_estimator": "usac"}, 5)
    assert cfg.ransac_thresh == pytest.approx(1.8)
    assert cfg.robust_estimator == "usac"
    assert r.changed


def test_rebootstrap_and_untrusted_merge():
    cfg = RuntimeConfig()
    assert invoke("rebootstrap", cfg, {}, 3).changed
    assert cfg.rebootstrap_requested
    invoke("mark_segment_untrusted", cfg, {"start": 10, "end": 20}, 10)
    invoke("mark_segment_untrusted", cfg, {"start": 21, "end": 30}, 21)
    assert len(cfg.untrusted) == 1 and cfg.untrusted[0] == [10, 30]


def test_snapshot_and_restore_roundtrip():
    cfg = RuntimeConfig()
    snap = cfg.snapshot()
    invoke("apply_preprocessing", cfg, {"mode": "sharpen"}, 1)
    invoke("retune_thresholds", cfg, {"ransac_thresh": 1.0}, 1)
    assert cfg.snapshot() != snap
    cfg.preprocess = snap["preprocess"]
    cfg.ransac_thresh = snap["ransac_thresh"]
    assert cfg.snapshot() == snap


def test_preprocess_image_ops_run():
    img = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    assert clahe(img).shape == img.shape
    assert sharpen(img).shape == img.shape
    assert apply_preprocess(img, "none") is img
    assert apply_preprocess(img, "garbage").shape == img.shape


# --------------------------------------------------------------------------- #
#  Fault injection
# --------------------------------------------------------------------------- #
def test_motion_blur_reduces_sharpness():
    img = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    blurred = motion_blur(img, 1.0)
    assert blur_score(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)) < \
           blur_score(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def test_frame_drop_is_dark():
    img = np.full((16, 16, 3), 200, np.uint8)
    assert brightness_score(cv2.cvtColor(frame_drop(img), cv2.COLOR_BGR2GRAY)) < 30


def test_fault_plan_window_gating():
    img = np.full((16, 16, 3), 200, np.uint8)
    plan = FaultPlan([FaultSegment("frame_drop", 10, 12)])
    assert np.array_equal(plan.apply(img, 9), img)
    assert not np.array_equal(plan.apply(img, 10), img)
    assert np.array_equal(plan.apply(img, 13), img)


def test_standard_scenarios_all_known_faults():
    plans = standard_scenarios(100)
    assert len(plans) == 4
    for p in plans:
        for s in p.segments:
            assert s.kind in FAULTS
            assert 30 <= s.start < s.end <= 70


def test_standard_scenarios_severity_is_monotonic():
    """U4: a lower severity level must produce a milder fault, not a no-op.

    Uses a textured image because blur/texture removal are a no-op on a
    uniform one — a real camera image always has gradient content to destroy.
    """
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    lo = standard_scenarios(100, severity=0.3)
    hi = standard_scenarios(100, severity=1.0)
    kinds = {p.segments[0].kind: p for p in lo}
    for p in hi:
        kind = p.segments[0].kind
        f = int(p.segments[0].start + 1)
        mild = kinds[kind].apply(img.copy(), f)
        hard = p.apply(img.copy(), f)
        if kind == "frame_drop":
            # duration-scaled: the mild plan corrupts fewer frames than nominal
            assert kinds[kind].segments[0].end < p.segments[0].end
            assert mild.max() < 128 and hard.max() < 128      # both are blackouts
        else:
            assert np.abs(mild.astype(int) - img.astype(int)).sum() < \
                   np.abs(hard.astype(int) - img.astype(int)).sum(), kind


def test_standard_scenarios_severity_clamps():
    assert standard_scenarios(100, severity=3.0)[0].segments[0].severity == 1.0
    assert standard_scenarios(100, severity=0.0)[0].segments[0].severity == 0.05
    # nominal level keeps the published grid byte-identical
    plans = standard_scenarios(100)
    assert [p.segments[0].severity for p in plans] == [1.0, 1.0, 0.8, 1.0]
    assert plans[3].segments[0].end - plans[3].segments[0].start == 8


# --------------------------------------------------------------------------- #
#  Human approval audit trail (U9)
# --------------------------------------------------------------------------- #
def test_approval_auto_skip_is_audited():
    from aegis.approval import ApprovalGate
    gate = ApprovalGate(auto="skip")
    req = gate.request(frame=12, reason="unrecoverable", evidence={"inliers": 3})
    assert req.wait() == "skipped"
    assert req.decided_by == "system:auto(skip)"
    assert req.resolved_at >= req.created_at


def test_approval_records_the_reviewer():
    from aegis.approval import ApprovalGate
    gate = ApprovalGate(default_timeout_s=5.0)
    req = gate.request(frame=4, reason="unrecoverable", evidence={})
    out = gate.resolve(req.decision_id, True, by="vj")
    assert out.status == "approved" and out.decided_by == "vj"
    d = out.to_dict()
    assert {"decided_by", "resolved_at", "status"} <= set(d)


def test_approval_timeout_is_attributable_to_no_human():
    from aegis.approval import ApprovalGate
    gate = ApprovalGate(default_timeout_s=0.05)
    req = gate.request(frame=4, reason="unrecoverable", evidence={})
    assert req.wait() == "timeout"
    assert req.decided_by == "system:timeout"
