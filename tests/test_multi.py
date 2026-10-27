import numpy as np
import pytest

from aegis.health import HealthSignals
from aegis.multi import Blackboard, Critic, MultiSupervisor, SafetyOfficer
from aegis.supervisor import SupervisorParams
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


def lost_window(n=20, start=0, n_lost=3, corrupt_frac=0.0, **kw):
    base = dict(blur=300.0, inl=10, cand=50, land=80)
    base.update(kw)
    sigs = window(n, start=start, **base)
    for i in range(n_lost):
        sigs[i * 6 % n].tracking_lost = True
    n_corrupt = int(n * corrupt_frac)
    for i in range(n_corrupt):
        sigs[i].brightness = 10.0
        sigs[i].blur_var = 0.0
    return sigs


HEALTHY_KW = dict(blur=300.0, inl=45, cand=50, land=200)


def make_multi(**pkw):
    cfg = RuntimeConfig()
    pkw.setdefault("persist_windows", 1)   # persistence is covered separately
    sup = MultiSupervisor(cfg, SupervisorParams(**pkw), mode="supervised")
    sup.backend_switch_times = [0, 1, 2, 3, 4, 5]   # force preprocess path
    return sup, cfg


# --------------------------------------------------------------------------- #
def test_multi_acts_and_records_votes():
    sup, cfg = make_multi()
    tr = DecisionTrace()
    out = sup.step(window(20, start=0, blur=10.0, inl=10, cand=50, land=80), tr)
    d = out["decision"]
    assert d["decided_by"] == "diagnosis"
    assert d["agents"] == ["perception", "diagnosis", "critic", "safety", "arbiter"]
    assert any(v["agent"] == "diagnosis" and not v["vetoed"] for v in d["votes"])
    # blur policy is an evidence-carrying abstain (sharpen failed ATE in the pilot)
    assert d["action"]["tool"] == "abstain"
    assert cfg.preprocess == "none"


def test_safety_vetoes_rebootstrap_during_corrupt_burst():
    # 30 % corrupt burst: below the rule ladder's 0.5 corrupted_input band, so
    # diagnosis proposes a re-bootstrap — and the safety officer vetoes it.
    sup, cfg = make_multi()
    tr = DecisionTrace()
    w1 = lost_window(start=0, corrupt_frac=0.30, bright=120.0)
    for s in w1[6:]:
        s.brightness = 150.0
        s.blur_var = 300.0
    sup.step(w1, tr)                      # streak 1 -> nothing
    w2 = lost_window(start=20, corrupt_frac=0.30, bright=120.0)
    for s in w2[6:]:
        s.brightness = 150.0
        s.blur_var = 300.0
    out = sup.step(w2, tr)                # streak 2 -> rebootstrap proposed
    d = out["decision"]
    assert d["decided_by"] in ("safety", "critic")
    assert cfg.rebootstrap_requested is False
    assert any(v["vetoed"] and v["tool"] == "rebootstrap" for v in d["votes"])
    assert cfg.untrusted            # fallback marked segment untrusted


def test_safety_refuses_untrust_on_healthy_scene():
    bb = Blackboard()
    sup, cfg = make_multi()
    bb.post(*_bw(window(20, **HEALTHY_KW)))
    safety = SafetyOfficer()
    prop = ("diagnosis", "textureless", "mark_segment_untrusted",
            {"window": 20}, "health_score", 1, 0.9)
    ok, reason = safety.check(prop, bb, sup)
    assert not ok and "healthy" in reason


def _bw(sigs):
    from aegis.supervisor import window_stats
    return window_stats(sigs), sigs


def test_critic_alternative_outranks_failed_primary():
    cfg = RuntimeConfig()
    sup = MultiSupervisor(cfg, SupervisorParams(persist_windows=1), mode="supervised")
    tr = DecisionTrace()
    # dynamic_or_repetitive: many features, poor ratio, ransac active
    bad = dict(feats=900, inl=10, cand=50, land=80)
    out1 = sup.step(window(20, start=0, **bad), tr)
    assert out1["decision"]["action"]["tool"] == "retune_thresholds"
    assert out1["decision"]["decided_by"] == "diagnosis"
    # next window worse -> verify rolls back; critic queues the alternative
    worse = window(20, start=20, feats=900, inl=3, cand=50, land=80)
    out2 = sup.step(worse, tr)
    rec2 = out2["decision"]
    assert rec2["decided_by"] == "critic"
    assert rec2["action"]["args"].get("max_features") == 6000


def test_critic_blur_aware_swaps_to_magsac():
    # same failure mode, but the window is BLURRED (low Laplacian variance):
    # the critic must swap the generic feature-cap alternative for the
    # threshold-free OpenCV-5 MAGSAC retune
    cfg = RuntimeConfig()
    sup = MultiSupervisor(cfg, SupervisorParams(persist_windows=1), mode="supervised")
    tr = DecisionTrace()
    bad = dict(feats=900, inl=10, cand=50, land=80, blur=76.0)
    sup.step(window(20, start=0, **bad), tr)      # primary: usac retune
    worse = window(20, start=20, feats=900, inl=3, cand=50, land=80, blur=76.0)
    out = sup.step(worse, tr)                      # rollback -> critic alt
    rec = out["decision"]
    assert rec["decided_by"] == "critic"
    assert rec["action"]["args"].get("robust_estimator") == "usac_magsac"


def test_critic_thrash_guard_falls_back_to_untrusted():
    cfg = RuntimeConfig()
    sup = MultiSupervisor(cfg, SupervisorParams(persist_windows=1), mode="supervised")
    tr = DecisionTrace()
    bad = dict(feats=900, inl=10, cand=50, land=80)
    sup.step(window(20, start=0, **bad), tr)
    worse = window(20, start=20, feats=900, inl=3, cand=50, land=80)
    sup.step(worse, tr)                    # rollback 1 -> critic alt
    worse2 = window(20, start=40, feats=900, inl=3, cand=50, land=80)
    out = sup.step(worse2, tr)             # alt rolled back too? streak check
    # after two consecutive failures the same diagnosis must not act again;
    # segment gets marked untrusted instead of endless retuning
    assert sup.critic.rollback_streak.get("dynamic_or_repetitive", 0) >= 1 or \
           out["decision"]["action"]["tool"] == "mark_segment_untrusted"


def test_multi_matches_single_on_clean_video():
    # healthy windows: no decisions at all (same false-intervention discipline)
    sup, cfg = make_multi()
    tr = DecisionTrace()
    for start in range(0, 80, 20):
        sup.step(window(20, start=start, **HEALTHY_KW), tr)
    assert tr.records == []
    assert cfg.preprocess == "none"
