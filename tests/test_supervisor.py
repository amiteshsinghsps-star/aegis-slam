import numpy as np
import pytest

from aegis.health import HealthSignals
from aegis.supervisor import Supervisor, SupervisorParams, window_stats
from aegis.tools import RuntimeConfig
from aegis.trace import DecisionTrace


def sig(frame, blur=300.0, bright=120.0, contrast=60.0, feats=900,
        inl=40, cand=50, land=120, lost=False):
    return HealthSignals(
        frame=frame, blur_var=blur, brightness=bright, contrast=contrast,
        texture=40.0, feature_count=feats, pnp_inliers=inl, pnp_candidates=cand,
        mean_reproj_px=1.0, tracked_landmarks=land, frame_to_frame_matches=200,
        tracking_lost=lost, initialised=True)


def window(n=20, start=0, **kw):
    return [sig(start + i, **kw) for i in range(n)]


BLUR_KW = dict(blur=10.0, inl=10, cand=50, land=80)   # ratio 0.2, blurred
LIGHT_KW = dict(bright=25.0, feats=420)               # dim + feature drop -> clahe
HEALTHY_KW = dict(blur=300.0, inl=45, cand=50, land=200)


def make_sup(verify=True, **pkw):
    cfg = RuntimeConfig()
    pkw.setdefault("persist_windows", 1)   # persistence is covered separately
    p = SupervisorParams(**pkw)
    sup = Supervisor(cfg, p, mode="supervised", verify=verify)
    # exhaust backend-switch budget so motion_blur takes the preprocess path
    sup.backend_switch_times = [0, 1, 2, 3, 4, 5]
    return sup, cfg


def test_window_stats_aggregation():
    w = window_stats(window(5, **BLUR_KW))
    assert w.n == 5
    assert w.blur_var == pytest.approx(10.0)
    assert w.inlier_ratio == pytest.approx(0.2)
    assert 0.0 < w.health_score < 1.0


def test_diagnose_motion_blur_abstains_and_traces():
    # sharpen passed the inlier proxy but worsened ATE on the dev pilot, so the
    # measured policy is an evidence-carrying abstain, not a config change.
    sup, cfg = make_sup()
    tr = DecisionTrace()
    out = sup.step(window(20, start=0, **BLUR_KW), tr)
    d = out["decision"]
    assert d["diagnosis"] == "motion_blur"
    assert d["action"]["tool"] == "abstain"
    assert cfg.preprocess == "none"
    assert d["outcome"]["verdict"] == "pending"
    assert d["config_before"] == d["config_after"]


def test_verify_improved_keeps_action():
    sup, cfg = make_sup()
    tr = DecisionTrace()
    out = sup.step(window(20, start=0, **LIGHT_KW), tr)
    did = out["decision"]["decision_id"]
    sup.step(window(20, start=20, **HEALTHY_KW), tr)
    rec = tr.records[0]
    assert rec["decision_id"] == did
    assert rec["outcome"]["verdict"] == "improved"
    assert sup.rolled_back == 0


def test_verify_worsened_rolls_back():
    sup, cfg = make_sup()
    tr = DecisionTrace()
    sup.step(window(20, start=0, **LIGHT_KW), tr)
    # next window even worse -> no measurable gain -> rollback
    sup.step(window(20, start=20, bright=25.0, feats=150), tr)
    assert tr.records[0]["outcome"]["verdict"] == "rolled_back"
    assert sup.rolled_back == 1
    # config was restored (before the next diagnosis could re-act)
    assert tr.records[0]["config_before"]["preprocess"] == "none"


def test_ablation_no_verify_no_rollback():
    sup, cfg = make_sup(verify=False)
    tr = DecisionTrace()
    sup.step(window(20, start=0, **LIGHT_KW), tr)
    sup.step(window(20, start=20, bright=25.0, feats=150), tr)
    assert tr.records[0]["outcome"]["verdict"] == "not_verified"
    assert sup.rolled_back == 0


def lost_window(n=20, start=0, n_lost=3, **kw):
    # a SHARP image that still loses tracking: isolates the tracking-loss branch
    base = dict(blur=300.0, inl=10, cand=50, land=80)
    base.update(kw)
    sigs = window(n, start=start, **base)
    for i in range(n_lost):
        sigs[i * 6 % n].tracking_lost = True
    return sigs


def test_transient_burst_does_not_rebootstrap():
    # one window of losses = transient fault; map is kept, no destructive action
    # (an image-level fix may still be proposed — the map must not be touched)
    sup, cfg = make_sup()
    tr = DecisionTrace()
    out = sup.step(lost_window(start=0), tr)
    assert cfg.rebootstrap_requested is False
    if "decision" in out:
        assert out["decision"]["action"]["tool"] != "rebootstrap"


def test_persistent_loss_queues_verifiable_rebootstrap():
    cfg = RuntimeConfig()
    sup = Supervisor(cfg, SupervisorParams(), mode="supervised")
    tr = DecisionTrace()
    sup.step(lost_window(start=0), tr)              # streak 1 -> no action
    out = sup.step(lost_window(start=20), tr)       # streak 2 -> act
    assert out["decision"]["diagnosis"] == "tracking_lost_persistent"
    assert out["decision"]["action"]["tool"] == "rebootstrap"
    assert cfg.rebootstrap_requested is True


def test_corrupted_burst_marks_untrusted_not_rebootstrap():
    # black drop frames: brightness ~14, zero variance — agent must wait, not wipe map
    sup, cfg = make_sup()
    tr = DecisionTrace()
    out = sup.step(lost_window(start=0, bright=14.0, blur=0.0), tr)
    d = out["decision"]
    assert d["diagnosis"] == "corrupted_input"
    assert d["action"]["tool"] == "mark_segment_untrusted"
    assert cfg.rebootstrap_requested is False
    assert cfg.untrusted == [[0, 19]]


def test_rebootstrap_kept_only_if_map_beaten():
    cfg = RuntimeConfig()
    sup = Supervisor(cfg, SupervisorParams(), mode="supervised")
    tr = DecisionTrace()
    sup.step(lost_window(start=0), tr)                    # streak 1 -> wait
    out = sup.step(lost_window(start=20), tr)             # streak 2 -> reboot
    did = out["decision"]["decision_id"]
    # rebuilt map still SMALLER than the destroyed one -> undo via snapshot
    sup.step(window(20, start=40, blur=10.0, inl=1, cand=50, land=5), tr)
    rec = [r for r in tr.records if r["decision_id"] == did][0]
    assert rec["outcome"]["verdict"] == "rolled_back"
    assert cfg._restore_requested is True


def test_rebootstrap_kept_when_map_regrows():
    cfg = RuntimeConfig()
    sup = Supervisor(cfg, SupervisorParams(), mode="supervised")
    tr = DecisionTrace()
    sup.step(lost_window(start=0), tr)
    out = sup.step(lost_window(start=20), tr)
    did = out["decision"]["decision_id"]
    # rebuilt map larger than before (200 > 80) -> keep
    sup.step(window(20, start=40, **HEALTHY_KW), tr)
    rec = [r for r in tr.records if r["decision_id"] == did][0]
    assert rec["outcome"]["verdict"] == "improved"
    assert cfg._restore_requested is False


def test_repeated_loss_escalates_to_recapture():
    cfg = RuntimeConfig()
    sup = Supervisor(cfg, SupervisorParams(), mode="supervised")
    sup._reboot_count = 2
    tr = DecisionTrace()
    out = sup.step(lost_window(start=0, n_lost=4), tr)
    assert out["decision"]["action"]["tool"] == "request_recapture"


def test_cooldown_blocks_rapid_reactions():
    sup, cfg = make_sup(cooldown=20)
    tr = DecisionTrace()
    sup.step(window(20, start=0, **BLUR_KW), tr)
    n0 = len(tr.records)
    out = sup.step(window(20, start=15, **BLUR_KW), tr)  # hi=34, 15 frames after 19
    assert len(tr.records) == n0
    assert "decision" not in out


def test_healthy_window_restores_baseline_preprocess():
    sup, cfg = make_sup()
    tr = DecisionTrace()
    cfg.preprocess = "clahe"          # a stale intervention
    sup.step(window(20, start=0, **HEALTHY_KW), tr)
    assert cfg.preprocess == "none"


def test_baseline_mode_never_acts():
    cfg = RuntimeConfig()
    sup = Supervisor(cfg, SupervisorParams(), mode="baseline")
    tr = DecisionTrace()
    out = sup.step(window(20, start=0, **BLUR_KW), tr)
    assert "decision" not in out
    assert tr.records == []


def test_diagnosis_requires_persistence_by_default():
    sup, cfg = make_sup(persist_windows=2)
    tr = DecisionTrace()
    first = sup.step(window(20, start=0, **BLUR_KW), tr)
    assert "decision" not in first                      # one noisy window is not evidence
    second = sup.step(window(20, start=20, **BLUR_KW), tr)
    assert second["decision"]["diagnosis"] == "motion_blur"


def test_thresholds_self_calibrate_from_healthy_frames():
    sup, cfg = make_sup(persist_windows=1)
    tr = DecisionTrace()
    # two clean windows with much sharper texture than the fixed 55 cut-off
    for start in (0, 20):
        sup.step(window(20, start=start, blur=1000.0, inl=45, cand=50, land=200), tr)
    assert sup._th["blur_low"] > 55.0
    # blur=100 would pass the fixed threshold but fails the calibrated one
    out = sup.step(window(20, start=40, blur=100.0, inl=10, cand=50, land=80), tr)
    assert out["decision"]["diagnosis"] == "motion_blur"


def test_preprocess_trial_rejects_no_gain():
    import numpy as np
    flat = np.full((48, 64, 3), 128, dtype=np.uint8)
    sup, cfg = make_sup(persist_windows=1)
    tr = DecisionTrace()
    out = sup.step(window(20, start=0, **BLUR_KW), tr, image=flat)
    assert out["decision"]["action"]["tool"] == "abstain"
    assert cfg.preprocess == "none"


def test_multicriteria_verify_rolls_back_hidden_regression():
    sup, cfg = make_sup(persist_windows=1)
    tr = DecisionTrace()
    sup.step(window(20, start=0, **LIGHT_KW), tr)
    # primary metric (feature count) returns, but two frames are now lost
    worse = window(20, start=20, bright=25.0, feats=500)
    for i in (0, 1):
        worse[i].tracking_lost = True
    sup.step(worse, tr)
    assert tr.records[0]["outcome"]["verdict"] == "rolled_back"
    assert "guards" in tr.records[0]["outcome"]


def test_natural_recovery_is_inconclusive_not_credited():
    sup, cfg = make_sup(persist_windows=1)
    tr = DecisionTrace()
    sup._metric_prev["feature_count"] = 300.0    # metric was already climbing
    sup.step(window(20, start=0, **LIGHT_KW), tr)
    # small gain consistent with the pre-action trend, not with the tool
    sup.step(window(20, start=20, bright=25.0, feats=460), tr)
    assert tr.records[0]["outcome"]["verdict"] == "inconclusive"
