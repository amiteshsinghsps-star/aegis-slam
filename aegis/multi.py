"""Multi-agent supervisor: role-separated cooperating agents over a blackboard.

Where the single ``Supervisor`` is one rule engine, ``MultiSupervisor`` splits
the same loop into five cooperating agents that exchange proposals on a shared
:class:`Blackboard`:

  * Perception agent — aggregates the frame window into WindowStats and
    per-frame evidence (also exposes derived facts like the corrupted-frame
    ratio, which window medians can hide).
  * Diagnosis agent  — the reproducible rule ladder, emitting scored proposals.
  * Critic agent     — independently re-scores the outcome of the previous
    action; after a failed verification it queues the *alternative* tool for
    that failure mode instead of blindly re-issuing the primary one, and
    detects thrashing (same diagnosis rolled back repeatedly).
  * Safety officer   — vetoes proposals that violate hard rails even when the
    rules suggested them (e.g. re-bootstrap while a third of the window is
    corrupted frames, backend-switch budget exhausted, destructive loops).
  * Arbiter          — collects votes (rule agent, critic, optional Bedrock
    planner), picks the highest-confidence non-vetoed proposal, and records
    every vote + veto in the decision trace for auditability.

Every proposal is a tuple ``(agent, diagnosis, tool, params, metric, direction,
confidence)``. The trace gains ``votes`` and the ``decided_by`` becomes the
winning agent's name — so multi-agent deliberation is fully auditable, and the
single-agent path stays available for A/B measurement (mode ``supervised``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from aegis.health import HealthSignals
from aegis.supervisor import (Supervisor, SupervisorParams, WindowStats,
                              window_stats, _Pending, _VALID_TOOLS,
                              _VERIFIABLE_TOOLS)
from aegis.tools import RuntimeConfig, invoke

Proposal = Tuple  # (agent, diagnosis, tool, params, metric, direction, confidence)


# --------------------------------------------------------------------------- #
#  Blackboard
# --------------------------------------------------------------------------- #
@dataclass
class Blackboard:
    """Shared workspace every agent can read; the only channel between them."""
    window: Optional[WindowStats] = None
    signals: List[HealthSignals] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)

    def post(self, w: WindowStats, signals: List[HealthSignals]) -> None:
        self.window = w
        self.signals = list(signals)

    def corrupt_ratio(self) -> float:
        """Fraction of window frames that are near-black AND variance-free
        (dropped/corrupted). Medians hide a 30-50 % burst; this does not."""
        if not self.signals:
            return 0.0
        n = sum(1 for s in self.signals if s.brightness < 40.0 and s.blur_var < 2.0)
        return n / float(len(self.signals))

    def lost_ratio(self) -> float:
        if not self.signals:
            return 0.0
        return sum(1 for s in self.signals if s.tracking_lost) / float(len(self.signals))

    def note(self, **kw) -> None:
        self.events.append(kw)


# --------------------------------------------------------------------------- #
#  Safety officer
# --------------------------------------------------------------------------- #
class SafetyOfficer:
    name = "safety"

    def __init__(self):
        self.vetoes = 0

    def check(self, prop: Proposal, bb: Blackboard, sup: "MultiSupervisor") -> Tuple[bool, str]:
        agent, diagnosis, tool, params, metric, direction, conf = prop
        if tool == "rebootstrap" and bb.corrupt_ratio() >= 0.30:
            self.vetoes += 1
            return False, (f"corrupt_ratio={bb.corrupt_ratio():.2f} — burst frames, "
                           f"waiting is recoverable; re-bootstrap is not")
        if tool == "set_feature_backend" and bb.window is not None \
                and not sup._can_switch_backend(bb.window.frame_hi):
            self.vetoes += 1
            return False, "backend-switch cap reached"
        if tool == "mark_segment_untrusted" and bb.corrupt_ratio() < 0.05 \
                and (bb.window is None or bb.window.health_score >= 0.6):
            self.vetoes += 1
            return False, "scene healthy — refusing to distrust good frames"
        return True, ""


# --------------------------------------------------------------------------- #
#  Critic
# --------------------------------------------------------------------------- #
class Critic:
    """Independent outcome assessor: learns from rollbacks, proposes
    alternatives, and flags thrashing."""
    name = "critic"

    # second-line actions when an agent's primary fix failed verification
    ALTERNATIVES = {
        "dynamic_or_repetitive": ("retune_thresholds", {"max_features": 6000}, "inlier_ratio", +1),
        # OpenCV-5 MAGSAC is threshold-free and integrates over consensus
        # sizes — the right second line when blur makes the noise scale itself
        # uncertain and a fixed px threshold (the primary sharpen/USAC path) failed
        "motion_blur":           ("retune_thresholds", {"robust_estimator": "usac_magsac"}, "inlier_ratio", +1),
        "low_light":             ("apply_preprocessing", {"mode": "denoise"}, "feature_count", +1),
        "textureless":           ("mark_segment_untrusted", {"window": 20}, "health_score", +1),
        "weak_geometry":         ("apply_preprocessing", {"mode": "sharpen"}, "inlier_ratio", +1),
    }

    def __init__(self):
        self.rollback_streak = {}
        self.pending_alt: Optional[Proposal] = None

    def record(self, diagnosis: str, verdict: str) -> None:
        if verdict == "rolled_back":
            self.rollback_streak[diagnosis] = self.rollback_streak.get(diagnosis, 0) + 1
            alt = self.ALTERNATIVES.get(diagnosis)
            if alt:
                # alternative outranks the (just-failed) primary proposal
                self.pending_alt = (("critic", diagnosis + "_alt") + alt + (0.95,))
        else:
            self.rollback_streak[diagnosis] = 0
            self.pending_alt = None

    def thrashing(self, diagnosis: str) -> bool:
        return self.rollback_streak.get(diagnosis, 0) >= 2


# --------------------------------------------------------------------------- #
#  Multi-supervisor (composition over the single-agent Supervisor)
# --------------------------------------------------------------------------- #
class MultiSupervisor(Supervisor):
    """Same step(signals, trace) contract as Supervisor — drop-in for the
    pipeline — but actions go through propose -> vote -> veto -> act -> verify,
    with the critic feeding learning back into the next proposal round."""

    def __init__(self, cfg: RuntimeConfig, params: Optional[SupervisorParams] = None,
                 mode: str = "supervised", llm=None, verify: bool = True):
        super().__init__(cfg, params, mode, llm, verify)
        self.bb = Blackboard()
        self.safety = SafetyOfficer()
        self.critic = Critic()
        self.last_diag = ""

    # -- main entry -------------------------------------------------------- #
    def step(self, signals: List[HealthSignals], trace,
             image=None, detector=None) -> dict:
        # 1) perception
        w = window_stats(signals)
        self.bb.post(w, signals)
        if self.mode == "baseline":
            return {"window": w}

        # 2) critic scores the pending action (verify/rollback via base)
        verdict = self._verify(w, trace)
        self._calibrate(w)
        if self.last_diag and verdict in ("rolled_back", "improved",
                                          "inconclusive", "not_verified"):
            self.critic.record(self.last_diag, verdict)
            if verdict != "rolled_back":
                self.last_diag = ""

        # 3) healthy → restore baseline (false-intervention suppression)
        if w.health_score >= 0.6 and self.cfg.preprocess != self._baseline_preprocess:
            self.cfg.preprocess = self._baseline_preprocess
        if w.health_score >= 0.65 and self.cfg.backend != self._baseline_backend \
                and self.cfg.backend != "aliked":
            self.cfg.backend = self._baseline_backend
        if (w.frame_hi - self.last_action_frame) < self.p.cooldown:
            return {"window": w}

        # 4) diagnosis agent proposes
        proposals: List[Proposal] = []
        diag = self._diagnose_gated(w)
        if diag is not None:
            diagnosis, tool, params, metric, direction = diag
            proposals.append(("diagnosis", diagnosis, tool, params, metric, direction, 0.9))

        # 4a) optional Bedrock planner vote (validated against tool schema)
        if self.p.allow_llm and self.llm is not None and diag is not None:
            choice = self.llm.propose(w, tool, params, self.cfg)
            if choice and choice.get("tool") in _VALID_TOOLS:
                proposals.append(("bedrock", diagnosis, choice["tool"],
                                  choice.get("args", {}), metric, direction, 0.6))

        # 4b) critic's learned alternative
        if diag is not None and self.critic.pending_alt is not None \
                and self.critic.pending_alt[1].startswith(diag[0]):
            alt = self.critic.pending_alt
            if alt[2] == "mark_segment_untrusted":
                # re-bind the static segment to the current offending window
                alt = alt[:3] + ({"start": w.frame_lo, "end": w.frame_hi},) + alt[4:]
            # blur-aware alternative selection: a window whose Laplacian
            # variance sits below the blur band but not low enough for the
            # motion_blur rule (median ~2.5×blur_low on our synthetic clips)
            # is a NOISE-SCALE problem — swap the feature-cap retune for
            # OpenCV-5's threshold-free MAGSAC
            if diag[0] == "dynamic_or_repetitive" and w.blur_var < 2.5 * self.p.blur_low:
                alt = (alt[0], alt[1], "retune_thresholds",
                       {"robust_estimator": "usac_magsac"}, "inlier_ratio", +1, alt[6])
            proposals.append(alt)

        if not proposals:
            return {"window": w}

        # 5) arbiter: highest-confidence proposal that passes safety
        proposals.sort(key=lambda p: -p[6])
        # F8 — when every proposal is vetoed, abstain unless the window itself
        # contains corrupted evidence that justifies distrusting the segment.
        if w.corrupt_ratio >= 0.05:
            fallback: Proposal = ("safety", "unsafe_conditions", "mark_segment_untrusted",
                                  {"start": w.frame_lo, "end": w.frame_hi},
                                  "health_score", +1, 0.2)
        else:
            fallback = ("safety", "unsafe_conditions", "abstain",
                        {"reason": "all proposals vetoed; no safe action"}, "health_score", +1, 0.2)
        chosen, votes = None, []
        for prop in proposals:
            ok, reason = self.safety.check(prop, self.bb, self)
            votes.append({"agent": prop[0], "tool": prop[2],
                          "confidence": prop[6], "vetoed": (not ok), "reason": reason})
            if ok and chosen is None:
                chosen = prop
        if chosen is None:
            chosen = fallback
            votes.append({"agent": "safety", "tool": fallback[2],
                          "confidence": 0.2, "vetoed": False, "reason": "fallback"})
            self.bb.note(action=f"safety_fallback_{fallback[2]}", frame=w.frame_hi)
        # thrash guard: critic vetoed repeating a twice-failed fix
        if self.critic.thrashing(chosen[1]) and chosen[2] not in ("mark_segment_untrusted", "abstain"):
            chosen = fallback
            self.bb.note(action=f"thrash_guard_{fallback[2]}", frame=w.frame_hi)

        agent, diagnosis, tool, params, metric, direction, conf = chosen

        # F4 (parity with the single agent) — trial the preprocess on the frame
        trial_ok, trial_note = self._trial_ok(tool, params, image, detector)
        if not trial_ok:
            tool, params = "abstain", {"reason": "preprocess trial showed no gain"}
            trial_note["trial_rejected"] = True

        # 6) recovery acts
        config_before = self.cfg.snapshot()
        result = invoke(tool, self.cfg, params, w.frame_hi)
        config_after = self.cfg.snapshot()

        rec = trace.add(
            frame=w.frame_hi, evidence=w.as_evidence(), diagnosis=diagnosis,
            decided_by=agent, action={"tool": tool, "args": params},
            config_before=config_before, config_after=config_after,
            outcome={"window": [w.frame_hi, w.frame_hi], "verdict": "pending"},
            extra={"votes": votes, "trial": trial_note,
                   "agents": ["perception", "diagnosis",
                              "critic", "safety", "arbiter"]},
        )
        self.last_action_frame = w.frame_hi
        self.decisions += 1
        if tool == "set_feature_backend" and result.changed:
            self.backend_switch_times.append(w.frame_hi)
        if result.ok and result.changed and tool in _VERIFIABLE_TOOLS:
            before = float(getattr(w, metric, 0.0))
            prev = self._metric_prev.get(metric)
            prior = 0.0 if prev is None else (before - prev) * direction
            self._metric_prev[metric] = before
            self.pending = _Pending(
                rec["decision_id"], metric, before, direction, config_before,
                tool=tool, diag=diagnosis,
                guard_before={"health_score": w.health_score,
                              "lost_frames": w.lost_frames,
                              "inlier_ratio": w.inlier_ratio},
                prior_slope=prior)
            self.last_diag = diagnosis
        return {"window": w, "decision": rec}
