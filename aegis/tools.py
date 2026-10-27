"""Agent tool registry (FR-4): the bounded set of actions the supervisor can take,
each backed by real OpenCV-5 operations applied to the running pipeline.

The supervisor never calls OpenCV directly; it selects a *tool* by name from
this registry. Every tool mutates a shared :class:`RuntimeConfig` that the
pipeline reads every frame, so a decision visibly changes downstream behaviour
(the Agentic-Vision requirement: perception evidence → later action).

Tools
  * set_feature_backend   — swap ORB / SIFT / ALIKED+LightGlue feature extractor
  * apply_preprocessing    — CLAHE / denoise / sharpen the incoming frames
  * retune_thresholds      — adjust RANSAC px, PnP min-inliers, feature cap, robust estimator
  * rebootstrap            — drop the map and re-initialise from a fresh good pair
  * mark_segment_untrusted — tag [start,end] frames as not trustworthy
  * request_recapture      — ask the human to re-record (needs approval; else untrusted)
  * abstain                — record that the evidence did not justify any action

Preprocessing (the image diagnostics/strengthening side of OpenCV 5) lives here
too so the tool and the pipeline agree on what each mode means.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Preprocessing (OpenCV 5 image operations)
# --------------------------------------------------------------------------- #
def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] in (3, 4) else img


def clahe(img: np.ndarray) -> np.ndarray:
    """Adaptive histogram equalisation — recovers contrast in dark scenes."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = cl.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
    """Non-loc means denoise — helps low-light sensor noise (h, hColor tuned small)."""
    return cv2.fastNlMeansDenoisingColored(img, None, h=5, hForColorComponents=5,
                                           templateWindowSize=7, searchWindowSize=21)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp masking — partially counters mild motion blur."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(img, 1.8, blurred, -0.8, 0)


PREPROCESSORS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": lambda img: img,
    "clahe": clahe,
    "denoise": denoise,
    "sharpen": sharpen,
}


def apply_preprocess(img: np.ndarray, mode: str) -> np.ndarray:
    fn = PREPROCESSORS.get(mode, PREPROCESSORS["none"])
    try:
        return fn(img)
    except Exception:
        return img


# --------------------------------------------------------------------------- #
#  Runtime configuration the pipeline reads every frame
# --------------------------------------------------------------------------- #
@dataclass
class RuntimeConfig:
    backend: str = "orb"           # {orb, sift, aliked}
    preprocess: str = "none"       # {none, clahe, denoise, sharpen}
    ransac_thresh: float = 2.5
    pnp_min_inliers: int = 30
    max_features: int = 4000
    min_conf: float = 0.7
    robust_estimator: str = "ransac"   # {ransac, usac, usac_magsac}
    # control signals consumed by the pipeline
    rebootstrap_requested: bool = False
    # trusted/untrusted segmentation
    untrusted: List[List[int]] = field(default_factory=list)
    _recapture: Optional[dict] = None
    # map snapshot + rollback control for the (destructive) re-bootstrap tool
    _snapshot: Optional[dict] = None
    _restore_requested: bool = False

    def snapshot(self) -> dict:
        return {
            "backend": self.backend,
            "preprocess": self.preprocess,
            "ransac_thresh": self.ransac_thresh,
            "pnp_min_inliers": self.pnp_min_inliers,
            "max_features": self.max_features,
            "min_conf": self.min_conf,
            "robust_estimator": self.robust_estimator,
        }

    def copy(self) -> "RuntimeConfig":
        return replace(self, untrusted=[list(seg) for seg in self.untrusted])


# --------------------------------------------------------------------------- #
#  Tool results + registry
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    ok: bool
    tool: str
    changed: bool
    params: dict
    note: str = ""


ToolFn = Callable[[RuntimeConfig, dict, int], ToolResult]


def _set_feature_backend(cfg: RuntimeConfig, params: dict, frame: int) -> ToolResult:
    backend = str(params.get("backend", cfg.backend)).lower()
    if backend not in {"orb", "sift", "aliked"}:
        return ToolResult(False, "set_feature_backend", False, params, f"unknown backend {backend}")
    if backend == "aliked":
        try:
            from slam.core.features_utils import HAS_LIGHTGLUE
            if not HAS_LIGHTGLUE:
                return ToolResult(False, "set_feature_backend", False, params,
                                  "aliked unavailable (no torch/lightglue)")
        except Exception:
            return ToolResult(False, "set_feature_backend", False, params, "aliked unavailable")
    changed = backend != cfg.backend
    cfg.backend = backend
    return ToolResult(True, "set_feature_backend", changed, {"backend": backend},
                      f"backend={'ALIKED+LightGlue' if backend=='aliked' else backend.upper()}")


def _apply_preprocessing(cfg: RuntimeConfig, params: dict, frame: int) -> ToolResult:
    mode = str(params.get("mode", "none")).lower()
    if mode not in PREPROCESSORS:
        return ToolResult(False, "apply_preprocessing", False, params, f"unknown mode {mode}")
    changed = mode != cfg.preprocess
    cfg.preprocess = mode
    return ToolResult(True, "apply_preprocessing", changed, {"mode": mode}, f"preprocess={mode}")


def _retune_thresholds(cfg: RuntimeConfig, params: dict, frame: int) -> ToolResult:
    changed = False
    applied = {}
    if "ransac_thresh" in params:
        v = float(params["ransac_thresh"])
        if abs(v - cfg.ransac_thresh) > 1e-6:
            cfg.ransac_thresh = v; changed = True
        applied["ransac_thresh"] = v
    if "pnp_min_inliers" in params:
        v = int(params["pnp_min_inliers"])
        if v != cfg.pnp_min_inliers:
            cfg.pnp_min_inliers = v; changed = True
        applied["pnp_min_inliers"] = v
    if "max_features" in params:
        v = int(params["max_features"])
        if v != cfg.max_features:
            cfg.max_features = v; changed = True
        applied["max_features"] = v
    if "robust_estimator" in params:
        v = str(params["robust_estimator"]).lower()
        if v in {"ransac", "usac", "usac_magsac"} and v != cfg.robust_estimator:
            cfg.robust_estimator = v; changed = True
        applied["robust_estimator"] = v
    return ToolResult(True, "retune_thresholds", changed, applied, f"tuned {applied}")


def _rebootstrap(cfg: RuntimeConfig, params: dict, frame: int) -> ToolResult:
    if cfg.rebootstrap_requested:
        return ToolResult(True, "rebootstrap", False, params, "already queued")
    cfg.rebootstrap_requested = True
    return ToolResult(True, "rebootstrap", True, {"at_frame": frame}, "queued re-bootstrap")


def _mark_segment_untrusted(cfg: RuntimeConfig, params: dict, frame: int) -> ToolResult:
    start = int(params.get("start", frame))
    end = int(params.get("end", start + int(params.get("window", 30))))
    seg = [max(0, start), end]
    # merge overlapping/adjacent segments
    merged = False
    for s in cfg.untrusted:
        if seg[0] <= s[1] + 1 and seg[1] >= s[0] - 1:
            s[0] = min(s[0], seg[0]); s[1] = max(s[1], seg[1]); merged = True; break
    if not merged:
        cfg.untrusted.append(seg)
    return ToolResult(True, "mark_segment_untrusted", True, {"segment": seg},
                      f"marked frames {seg[0]}-{seg[1]} untrusted")


def _request_recapture(cfg: RuntimeConfig, params: dict, frame: int) -> ToolResult:
    req = {"tool": "request_recapture", "frame": frame,
           "reason": str(params.get("reason", "low confidence segment")),
           "evidence": params.get("evidence", {})}
    cfg._recapture = req
    return ToolResult(True, "request_recapture", True, req, "awaiting human approval")


def _abstain(cfg: RuntimeConfig, params: dict, frame: int) -> ToolResult:
    """Deliberate no-op. Keeps an auditable record that the supervisor saw the
    evidence and chose not to touch the configuration."""
    reason = str(params.get("reason", "insufficient evidence"))
    return ToolResult(True, "abstain", False, {"reason": reason},
                      f"abstained: {reason}")


TOOLS: Dict[str, ToolFn] = {
    "set_feature_backend": _set_feature_backend,
    "apply_preprocessing": _apply_preprocessing,
    "retune_thresholds": _retune_thresholds,
    "rebootstrap": _rebootstrap,
    "mark_segment_untrusted": _mark_segment_untrusted,
    "request_recapture": _request_recapture,
    "abstain": _abstain,
}

# JSON-ish schema used by the (optional) LLM planner to constrain its choices.
TOOL_SCHEMA = {
    "set_feature_backend": {"backend": ["orb", "sift", "aliked"]},
    "apply_preprocessing": {"mode": ["none", "clahe", "denoise", "sharpen"]},
    "retune_thresholds": {"ransac_thresh": "float>0", "pnp_min_inliers": "int",
                          "max_features": "int", "robust_estimator": ["ransac", "usac", "usac_magsac"]},
    "rebootstrap": {},
    "mark_segment_untrusted": {"start": "int", "end": "int"},
    "request_recapture": {"reason": "str"},
    "abstain": {"reason": "str"},
}


def invoke(name: str, cfg: RuntimeConfig, params: dict, frame: int) -> ToolResult:
    fn = TOOLS.get(name)
    if fn is None:
        return ToolResult(False, name, False, params, "unknown tool")
    return fn(cfg, params or {}, frame)
