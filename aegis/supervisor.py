"""The agent supervisor (FR-3, FR-5): observe → diagnose → decide → act → verify.

This is the heart of AEGIS-SLAM. It reads aggregated image + tracking evidence
over a sliding window, maps it to a failure mode, picks one bounded action from
the tool registry, and — crucially — verifies the effect on the *next* window
and rolls the change back if it made tracking worse. A rule engine always runs
(reproducible); an optional Bedrock planner can refine ambiguous cases but its
output is validated against the tool schema and discarded otherwise, falling
back to the rules.

Safety rails (section 7 of the PRD):
  * cooldown after every change,
  * a cap on backend switches per minute of video,
  * automatic rollback on regression,
  * no destructive tools,
  * re-capture requests need human approval (handled by the pipeline/approval gate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from aegis.health import HealthSignals
from aegis.tools import RuntimeConfig, invoke, ToolResult


# --------------------------------------------------------------------------- #
#  Window statistics
# --------------------------------------------------------------------------- #
@dataclass
class WindowStats:
    n: int = 0
    frame_lo: int = 0
    frame_hi: int = 0
    blur_var: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    texture: float = 0.0
    feature_count: float = 0.0
    inlier_ratio: float = 0.0
    pnp_inliers: float = 0.0
    tracked_landmarks: float = 0.0
    mean_reproj_px: float = 0.0
    frame_to_frame_matches: float = 0.0
    lost_frames: int = 0
    corrupt_ratio: float = 0.0
    kf_frames: int = 0
    health_score: float = 0.0

    def as_evidence(self) -> dict:
        return {
            "inliers": round(self.pnp_inliers, 1),
            "inlier_ratio": round(self.inlier_ratio, 3),
            "blur_var": round(self.blur_var, 2),
            "brightness": round(self.brightness, 1),
            "contrast": round(self.contrast, 1),
            "texture": round(self.texture, 2),
            "feature_count": round(self.feature_count, 1),
            "tracked_landmarks": round(self.tracked_landmarks, 1),
            "mean_reproj_px": round(self.mean_reproj_px, 3),
            "frame_to_frame_matches": round(self.frame_to_frame_matches, 1),
            "lost_frames": self.lost_frames,
            "corrupt_ratio": round(self.corrupt_ratio, 3),
            "health_score": round(self.health_score, 3),
        }


def window_stats(signals: List[HealthSignals]) -> WindowStats:
    if not signals:
        return WindowStats()
    arr = lambda f: np.array([getattr(s, f) for s in signals], dtype=np.float64)
    ratios = np.array([s.inlier_ratio for s in signals], dtype=np.float64)
    w = WindowStats(
        n=len(signals),
        frame_lo=signals[0].frame,
        frame_hi=signals[-1].frame,
        blur_var=float(np.median(arr("blur_var"))),
        brightness=float(np.median(arr("brightness"))),
        contrast=float(np.median(arr("contrast"))),
        texture=float(np.median(arr("texture"))),
        feature_count=float(np.mean(arr("feature_count"))),
        inlier_ratio=float(np.mean(ratios)),
        pnp_inliers=float(np.mean(arr("pnp_inliers"))),
        tracked_landmarks=float(np.mean(arr("tracked_landmarks"))),
        mean_reproj_px=float(np.mean(arr("mean_reproj_px"))),
        frame_to_frame_matches=float(np.mean(arr("frame_to_frame_matches"))),
        lost_frames=int(sum(1 for s in signals if s.tracking_lost)),
        kf_frames=int(sum(1 for s in signals if s.is_keyframe)),
    )
    # per-frame corrupted fraction — medians hide a 30-50 % burst window
    w.corrupt_ratio = float(sum(1 for s in signals
                                if s.brightness < 40.0 and s.blur_var < 2.0) / len(signals))
    # composite health in [0,1]: mostly inlier ratio + landmark coverage
    w.health_score = float(np.clip(
        0.6 * min(w.inlier_ratio / 0.5, 1.0)
        + 0.4 * min(w.tracked_landmarks / 120.0, 1.0), 0.0, 1.0))
    return w


# --------------------------------------------------------------------------- #
#  Tunable thresholds + policy params
# --------------------------------------------------------------------------- #
@dataclass
class SupervisorParams:
    cadence: int = 10              # evaluate every N frames
    window: int = 20               # frames aggregated per decision
    cooldown: int = 20             # min frames between actions
    blur_low: float = 55.0         # Laplacian var below → blur (fallback)
    dark: float = 60.0             # brightness below → low light (fallback)
    low_contrast: float = 22.0     # fallback contrast floor
    few_features: int = 90         # feature count below → textureless (fallback)
    low_inlier_ratio: float = 0.35
    min_landmarks: float = 40.0
    lost_streak_trigger: int = 3   # lost frames in window → re-bootstrap
    max_backend_switches_per_min: int = 6
    verify_eps: float = 0.03       # improvement needed to keep an action
    allow_llm: bool = False
    # robustness features (WS2)
    adaptive_thresholds: bool = True   # F1: derive image thresholds from the run's own healthy frames
    persist_windows: int = 2           # F2: non-urgent diagnoses must repeat before acting
    trial_preprocessing: bool = True   # F4: image-level trial before committing a preprocess change
    multicriteria_verify: bool = True  # F5: a primary win cannot hide a health/lost regression
    natural_recovery_control: bool = True  # F6: do not credit an action if the trend already predicted it


@dataclass
class _Pending:
    decision_id: str
    metric: str                    # which WindowStats attribute to verify
    before: float
    direction: int                 # +1 want higher, -1 want lower
    config_before: dict            # snapshot to restore on rollback
    tool: str = ""                 # tool name (rebootstrap needs map-undo, not config-undo)
    diag: str = ""                 # diagnosis that produced this action (for the critic)
    guard_before: dict = field(default_factory=dict)  # F5 guard metrics at action time
    prior_slope: float = 0.0       # F6 metric trend over the two windows before acting


class Supervisor:
    def __init__(self, cfg: RuntimeConfig, params: Optional[SupervisorParams] = None,
                 mode: str = "supervised", llm=None, verify: bool = True):
        self.cfg = cfg
        self.p = params or SupervisorParams()
        self.mode = mode
        self.verify = verify
        self.llm = llm
        self.last_action_frame = -10**9
        self.pending: Optional[_Pending] = None
        # baseline reference config for clean scenes (used to restore after recovery)
        self._baseline_backend = cfg.backend
        self._baseline_preprocess = cfg.preprocess
        self.backend_switch_times: List[int] = []
        self.decisions = 0
        self.rolled_back = 0
        self._reboot_count = 0
        self._lost_streak = 0
        # F1: thresholds derived from this run's own healthy frames
        self._calib: List[dict] = []
        self._th = self._default_thresholds()
        # F2: consecutive-window persistence (failure episodes, not labels —
        # blur can present as dynamic-texture in one window and motion_blur in
        # the next; both are the same unresolved episode)
        self._diag_streak: dict = {}
        self._unhealthy_streak = 0
        # F6: previous window values for the metric we verify, per metric name
        self._metric_prev: dict = {}

    def _default_thresholds(self) -> dict:
        p = self.p
        return {"blur_low": p.blur_low, "dark": p.dark,
                "low_contrast": p.low_contrast, "few_features": p.few_features,
                "low_inlier_ratio": p.low_inlier_ratio,
                "low_inliers": p.min_landmarks}

    def _calibrate(self, w: WindowStats) -> None:
        """F1 — self-calibrated image thresholds.

        Fixed absolute thresholds tuned on one camera/exposure do not transfer.
        We record the first healthy windows of the current run and express the
        failure cut-offs as fractions of that reference, bounded so a genuinely
        dark or low-texture *dataset* cannot make the agent blind. The fixed
        values remain the cold-start fallback and the lower/upper bounds.
        """
        if not self.p.adaptive_thresholds:
            return
        if (w.health_score >= 0.6 and w.lost_frames <= 1
                and w.corrupt_ratio < 0.05 and w.feature_count > 0):
            self._calib.append({"blur": w.blur_var, "bright": w.brightness,
                                "contrast": w.contrast, "features": w.feature_count,
                                "ratio": w.inlier_ratio, "inliers": w.pnp_inliers})
            self._calib = self._calib[-5:]
        if not self._calib:
            return
        med = lambda k: float(np.median([c[k] for c in self._calib]))
        self._th = {
            "blur_low": float(np.clip(0.12 * med("blur"), 30.0, 120.0)),
            "dark": float(np.clip(0.60 * med("bright"), 45.0, 90.0)),
            "low_contrast": float(np.clip(0.55 * med("contrast"), 15.0, 32.0)),
            "few_features": int(np.clip(0.12 * med("features"), 60, 200)),
            # clean synthetic footage legitimately sits near 0.25-0.30 inlier
            # ratio; a fixed 0.35 cut-off fires on it. Calibrate the geometry
            # floor to 40 % of the clean reference, bounded to [0.12, 0.30].
            "low_inlier_ratio": float(np.clip(0.40 * med("ratio"), 0.12, 0.30)),
            # absolute floor: blur collapses inlier COUNT faster than the ratio
            "low_inliers": float(np.clip(0.30 * med("inliers"), 20.0, 100.0)),
        }

    def _diagnose_gated(self, w: WindowStats) -> Optional[tuple]:
        """F2 — persistence/hysteresis gate around the rule ladder.

        Image-only diagnoses must be seen in two consecutive windows before
        they may change the configuration; a single noisy window is not
        evidence. The gate resets as soon as the diagnosis changes or the scene
        is healthy, so a later episode still gets a fast first action after its
        confirmation window.
        """
        diag = self._diagnose(w)
        if diag is None:
            if w.health_score >= 0.6 or w.lost_frames == 0:
                self._diag_streak.clear()
                self._unhealthy_streak = 0
            return None
        diagnosis = diag[0]
        self._diag_streak[diagnosis] = self._diag_streak.get(diagnosis, 0) + 1
        if diagnosis in _PERSISTENT_DIAGNOSES:
            self._unhealthy_streak += 1
            if self._unhealthy_streak < max(1, self.p.persist_windows):
                return None
        else:
            self._unhealthy_streak = 0
        return diag

    def _trial_ok(self, tool: str, params: dict, image, detector) -> tuple:
        """F4 — run the proposed image transform on the latest frame and check
        that it actually changes the signal the diagnosis is about. This is a
        real trial, not a simulation: no config is committed if it fails."""
        if (not self.p.trial_preprocessing or tool != "apply_preprocessing"
                or image is None):
            return True, {}
        from aegis.health import image_quality
        from aegis.tools import apply_preprocess
        mode = str(params.get("mode", "none"))
        if mode not in ("clahe", "sharpen"):
            return True, {}
        before = image_quality(image, detector)
        after = image_quality(apply_preprocess(image, mode), detector)
        key = "contrast" if mode == "clahe" else "blur_var"
        b, a = float(before.get(key, 0.0)), float(after.get(key, 0.0))
        ok = b > 0 and a >= b * 1.05
        return ok, {"trial_mode": mode, "trial_metric": key,
                    "trial_before": round(b, 3), "trial_after": round(a, 3)}

    # -- verification ------------------------------------------------------ #
    def _verify(self, new: WindowStats, trace) -> str:
        """Measure the pending action on the new window. Returns the verdict."""
        if self.pending is None:
            return "none"
        if not self.verify:
            # ablation: no verify/rollback — leave the decision in place
            trace.set_outcome(self.pending.decision_id, {
                "window": [new.frame_lo, new.frame_hi],
                self.pending.metric: float(getattr(new, self.pending.metric, 0.0)),
                "verdict": "not_verified",
            })
            self.pending = None
            return "not_verified"
        after = float(getattr(new, self.pending.metric, 0.0))
        before = self.pending.before
        delta = (after - before) * self.pending.direction
        improved = delta >= self.p.verify_eps * max(abs(before), 1.0)

        # F5 — multi-criteria verification. A primary proxy (inlier ratio,
        # feature count) can rise while tracking actually gets worse; keep the
        # action only if the guard signals did not regress.
        guards_ok, guard_note = True, {}
        gb = self.pending.guard_before or {}
        if improved and self.p.multicriteria_verify and gb:
            health_drop = float(gb.get("health_score", new.health_score)) - new.health_score
            lost_rise = new.lost_frames - int(gb.get("lost_frames", new.lost_frames))
            guards_ok = health_drop <= 0.10 and lost_rise <= 1
            guard_note = {"health_before": round(gb.get("health_score", 0.0), 3),
                          "health_after": round(new.health_score, 3),
                          "lost_before": int(gb.get("lost_frames", 0)),
                          "lost_after": int(new.lost_frames)}

        # F6 — natural-recovery control. If the metric was already improving at
        # the same rate before the action, the next window cannot attribute the
        # gain to the tool. Keep the config but do not claim the win.
        natural = False
        if improved and guards_ok and self.p.natural_recovery_control:
            prior = self.pending.prior_slope
            threshold = self.p.verify_eps * max(abs(before), 1.0)
            if prior > threshold and delta <= 1.5 * prior:
                natural = True

        if improved and guards_ok:
            verdict = "inconclusive" if natural else "improved"
        else:
            # roll back the config change
            self._restore(self.pending.config_before)
            if self.pending.tool == "rebootstrap":
                # undo the destructive part too: ask the pipeline to restore
                # the map snapshot taken before the re-bootstrap
                self.cfg._restore_requested = True
            verdict = "rolled_back"
            self.rolled_back += 1

        outcome = {"window": [new.frame_lo, new.frame_hi],
                   self.pending.metric: after, "delta": round(delta, 4),
                   "verdict": verdict}
        if guard_note:
            outcome["guards"] = guard_note
        if natural:
            outcome["note"] = "pre-action trend explains the gain"
        trace.set_outcome(self.pending.decision_id, outcome)
        self.pending = None
        return verdict

    def _restore(self, snapshot: dict) -> None:
        self.cfg.backend = snapshot.get("backend", self.cfg.backend)
        self.cfg.preprocess = snapshot.get("preprocess", self.cfg.preprocess)
        self.cfg.ransac_thresh = snapshot.get("ransac_thresh", self.cfg.ransac_thresh)
        self.cfg.pnp_min_inliers = snapshot.get("pnp_min_inliers", self.cfg.pnp_min_inliers)
        self.cfg.max_features = snapshot.get("max_features", self.cfg.max_features)
        self.cfg.robust_estimator = snapshot.get("robust_estimator", self.cfg.robust_estimator)

    # -- diagnosis --------------------------------------------------------- #
    def _diagnose(self, w: WindowStats) -> Optional[tuple]:
        """Return (diagnosis, tool, params, verify_metric, direction) or None."""
        p = self.p
        th = self._th
        low_ratio = th.get("low_inlier_ratio", p.low_inlier_ratio)
        enough_landmarks = w.tracked_landmarks >= p.min_landmarks

        # 0) corrupted input: a DOMINANT fraction (>=50 %) of the window is
        # near-black, variance-free burst frames (dropped/corrupted feed). The
        # right move is to stop trusting the segment and WAIT — re-bootstrapping
        # here destroys a recoverable map. Partial bursts (30-50 %) fall
        # through to the persistence gate in single mode and to the multi-agent
        # safety officer's veto in multi mode (deliberate band separation).
        if w.lost_frames >= p.lost_streak_trigger and w.corrupt_ratio >= 0.5:
            return ("corrupted_input", "mark_segment_untrusted",
                    {"start": w.frame_lo - self.p.window, "end": w.frame_hi},
                    "health_score", +1)

        # 1) tracking lost — act only if loss PERSISTS across consecutive
        # windows (transient burst faults recover on their own; re-bootstrapping
        # mid-burst destroys a recoverable map, a measured regression we fixed).
        if w.lost_frames == 0:
            self._lost_streak = 0
            if w.health_score >= 0.5:
                # fully recovered: a later fresh episode starts clean
                self._reboot_count = 0
        if w.lost_frames >= p.lost_streak_trigger:
            self._lost_streak += 1
            if self._reboot_count >= 2 or self._lost_streak >= 5:
                # F7 — evidence-based allowlist: a recapture request is surfaced
                # only with the concrete counters that justified it. Otherwise
                # abstain instead of asking a human to re-record on thin evidence.
                evidence = {"lost_streak": self._lost_streak,
                            "reboot_count": self._reboot_count,
                            "corrupt_ratio": round(w.corrupt_ratio, 3),
                            "health_score": round(w.health_score, 3)}
                if (self._reboot_count >= 2 or self._lost_streak >= 3) \
                        and w.corrupt_ratio < 0.5:
                    return ("tracking_lost_unrecoverable", "request_recapture",
                            {"reason": "tracking lost after 2 re-bootstraps",
                             "evidence": evidence}, "health_score", +1)
                return ("tracking_lost_unrecoverable", "abstain",
                        {"reason": "recovery counters not yet met"}, "health_score", +1)
            if self._lost_streak >= 2:
                # Stale-loss guard: the streak counts the whole window, so a
                # fault that has already ended can still leave lost frames in
                # it. Re-bootstrapping a window whose health has recovered
                # destroys a map the tracker just rebuilt on its own.
                if w.health_score >= 0.55:
                    # recovered on its own: skip loss escalation, but still let
                    # an image cause (e.g. blur at fault onset) be named
                    return self._image_diagnosis(w, th, low_ratio, enough_landmarks)
                # Image-side cause first: persistent tracking loss during a
                # blur/light/texture fault is the fault's consequence, not a
                # broken map. Re-bootstrapping here would destroy a recoverable
                # map and mask the real diagnosis.
                cause = self._image_diagnosis(w, th, low_ratio, enough_landmarks)
                if cause is not None:
                    return cause
                self._reboot_count += 1
                # verify against MAP SIZE, not health_score: a re-built map is
                # only worth keeping if it beats the snapshot we destroyed
                return ("tracking_lost_persistent", "rebootstrap", {}, "tracked_landmarks", +1)
            # First lost window: do NOT return yet. A blur/light fault often
            # causes one lost frame before it causes persistent loss, and the
            # image-side evidence is the actionable diagnosis; fall through so
            # the persistence gate can confirm it.

        diag = self._image_diagnosis(w, th, low_ratio, enough_landmarks)
        if diag is not None:
            return diag

        # 6) still weak after USAC → tighten and raise feature cap
        if (w.inlier_ratio < low_ratio and enough_landmarks
                and self.cfg.robust_estimator.startswith("usac")):
            return ("weak_geometry", "retune_thresholds",
                    {"max_features": min(8000, self.cfg.max_features + 2000)},
                    "inlier_ratio", +1)

        return None

    def _image_diagnosis(self, w: WindowStats, th: dict, low_ratio: float,
                         enough_landmarks: bool) -> Optional[tuple]:
        """Image-quality causes, checked before escalating persistent loss."""
        # 2) motion blur: sharp-image metric collapsed and geometry degraded
        if w.blur_var < th["blur_low"] and (w.inlier_ratio < low_ratio
                                            or w.pnp_inliers < th["low_inliers"]):
            severe_blur = w.blur_var < 0.5 * th["blur_low"]
            if not (enough_landmarks or severe_blur):
                return None
            # Measured on the dev pilot: sharpen and the ORB->ALIKED family switch
            # both raised inlier ratio while final ATE got worse (the proxy trap),
            # so verification cannot validate either. Abstain and record why.
            if severe_blur and self._can_switch_backend(w.frame_hi) \
                    and _aliked_available() and self.cfg.backend != "aliked":
                return ("motion_blur", "abstain",
                        {"reason": "severe blur: family switch unverifiable online"},
                        "inlier_ratio", +1)
            return ("motion_blur", "abstain",
                    {"reason": "blur: sharpen passed inlier proxy but not ATE"},
                    "inlier_ratio", +1)

        # 3) low light: dim / low contrast with feature drop
        if (w.brightness < th["dark"] or w.contrast < th["low_contrast"]) and w.feature_count < 800:
            return ("low_light", "apply_preprocessing", {"mode": "clahe"}, "feature_count", +1)

        # 4) textureless: very few features even in a bright scene
        if w.feature_count < th["few_features"]:
            if self._can_switch_backend(w.frame_hi):
                return ("textureless", "set_feature_backend", {"backend": "sift"}, "feature_count", +1)
            return ("textureless", "mark_segment_untrusted",
                    {"start": w.frame_lo, "end": w.frame_hi}, "health_score", +1)

        # 5) many features but poor geometry → repetitive texture / dynamic objects
        # Only on frames that still track: a lost window has no PnP geometry, so
        # a low inlier ratio there says nothing about the scene's texture.
        if (w.lost_frames == 0 and w.feature_count > 600
                and w.inlier_ratio < low_ratio
                and self.cfg.robust_estimator == "ransac"):
            return ("dynamic_or_repetitive", "retune_thresholds",
                    {"robust_estimator": "usac", "ransac_thresh": max(1.5, self.cfg.ransac_thresh - 0.5)},
                    "inlier_ratio", +1)
        return None

    def _can_switch_backend(self, frame: int) -> bool:
        # cap switches per minute (assume ~30 fps → 1800 frames/min; use cadence-agnostic 60s ~ frame window)
        recent = [f for f in self.backend_switch_times if f > frame - 1800]
        return len(recent) < self.p.max_backend_switches_per_min

    # -- main entry: called by the pipeline every `cadence` frames --------- #
    def step(self, signals: List[HealthSignals], trace,
             image=None, detector=None) -> Optional[dict]:
        w = window_stats(signals)
        if self.mode == "baseline":
            return {"window": w}
        # verify the previous action against this window first
        self._verify(w, trace)
        self._calibrate(w)

        # if healthy, gently restore baseline preprocessing to avoid false actions
        if w.health_score >= 0.6 and self.cfg.preprocess != self._baseline_preprocess:
            self.cfg.preprocess = self._baseline_preprocess
        if w.health_score >= 0.65 and self.cfg.backend != self._baseline_backend \
                and self.cfg.backend != "aliked":
            # only revert away from a classical escalation, keep learned if chosen
            self.cfg.backend = self._baseline_backend

        if (w.frame_hi - self.last_action_frame) < self.p.cooldown:
            return {"window": w}

        diag = self._diagnose_gated(w)
        if diag is None:
            return {"window": w}
        diagnosis, tool, params, metric, direction = diag

        # F4 — trial the transform on the latest frame before committing it
        trial_ok, trial_note = self._trial_ok(tool, params, image, detector)
        if not trial_ok:
            rec = trace.add(
                frame=w.frame_hi, evidence=w.as_evidence(), diagnosis=diagnosis,
                decided_by="rules", action={"tool": "abstain",
                                            "args": {"reason": "preprocess trial showed no gain"}},
                config_before=self.cfg.snapshot(), config_after=self.cfg.snapshot(),
                outcome={"window": [w.frame_hi, w.frame_hi], "verdict": "abstained",
                         "trial": trial_note},
            )
            self.last_action_frame = w.frame_hi
            return {"window": w, "decision": rec}

        decided_by = "rules"
        # optional LLM planner only for ambiguous cases
        if self.p.allow_llm and self.llm is not None and diagnosis in ("dynamic_or_repetitive", "weak_geometry"):
            choice = self.llm.propose(w, tool, params, self.cfg)
            if choice and choice.get("tool") in _VALID_TOOLS:
                tool, params = choice["tool"], choice.get("args", {})
                decided_by = "bedrock"

        config_before = self.cfg.snapshot()
        result: ToolResult = invoke(tool, self.cfg, params, w.frame_hi)
        config_after = self.cfg.snapshot()

        rec = trace.add(
            frame=w.frame_hi, evidence=w.as_evidence(), diagnosis=diagnosis,
            decided_by=decided_by, action={"tool": tool, "args": params},
            config_before=config_before, config_after=config_after,
            outcome={"window": [w.frame_hi, w.frame_hi], "verdict": "pending"},
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
        return {"window": w, "decision": rec}


def _aliked_available() -> bool:
    try:
        from slam.core.features_utils import HAS_LIGHTGLUE
        return bool(HAS_LIGHTGLUE)
    except Exception:
        return False


_VALID_TOOLS = {"set_feature_backend", "apply_preprocessing", "retune_thresholds",
                "rebootstrap", "mark_segment_untrusted", "request_recapture",
                "abstain"}
_VERIFIABLE_TOOLS = {"set_feature_backend", "apply_preprocessing",
                     "retune_thresholds", "rebootstrap"}
# image-only diagnoses must persist before they may mutate the configuration
_PERSISTENT_DIAGNOSES = {"motion_blur", "low_light", "textureless",
                         "dynamic_or_repetitive", "weak_geometry"}
