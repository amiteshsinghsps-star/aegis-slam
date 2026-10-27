"""Per-frame health-signal extraction (FR-2).

SLAM breaks silently; the agent can only react if the pipeline exposes *why* a
frame is hard. This module turns raw images and the reused OpenCV-5 geometry
stage outputs into a compact, comparable health snapshot.

Two families of signals:

  * Image-quality (computed here, pure OpenCV 5): blur via variance of the
    Laplacian, brightness (mean), contrast (std), texture energy, feature
    count.
  * Tracking-health (fed in by the pipeline from the reused ``slam.core``
    stages): PnP inlier count/ratio, mean reprojection error, tracked landmark
    count, frame-to-frame match count, keyframe flag, tracking-lost flag.

Signals are intentionally cheap so the supervisor can evaluate them every few
frames without dominating runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Image-quality primitives (OpenCV 5)
# --------------------------------------------------------------------------- #
def _to_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian — lower means blurrier. Motion blur collapses
    this quickly, which is the diagnostic the supervisor keys on."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def brightness_score(gray: np.ndarray) -> float:
    return float(gray.mean())


def contrast_score(gray: np.ndarray) -> float:
    return float(gray.std())


def texture_energy(gray: np.ndarray) -> float:
    """Mean gradient magnitude — a robust proxy for usable texture. Low texture
    (a blank wall) is a classic SLAM failure mode."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(mag.mean())


def image_quality(img: np.ndarray, detector=None) -> dict:
    """Compute the image-side quality signals. If a detector is supplied, also
    report how many features OpenCV could find."""
    gray = _to_gray(img)
    q = {
        "blur_var": blur_score(gray),
        "brightness": brightness_score(gray),
        "contrast": contrast_score(gray),
        "texture": texture_energy(gray),
    }
    if detector is not None:
        try:
            kp = detector.detect(gray if gray is not None else img, None)
            q["feature_count"] = int(len(kp))
        except Exception:
            q["feature_count"] = 0
    else:
        q["feature_count"] = -1
    return q


# --------------------------------------------------------------------------- #
#  Full health snapshot
# --------------------------------------------------------------------------- #
@dataclass
class HealthSignals:
    frame: int
    timestamp: float = 0.0

    # image quality
    blur_var: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    texture: float = 0.0
    feature_count: int = 0

    # tracking health (from reused geometry stages)
    pnp_inliers: int = 0
    pnp_candidates: int = 0
    mean_reproj_px: float = 0.0
    tracked_landmarks: int = 0
    frame_to_frame_matches: int = 0

    # state
    is_keyframe: bool = False
    tracking_lost: bool = False
    initialised: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        # compact evidence subset used in the decision trace
        return d

    @property
    def inlier_ratio(self) -> float:
        if self.pnp_candidates <= 0:
            return 0.0
        return float(self.pnp_inliers) / float(self.pnp_candidates)


def frame_confidence(s: "HealthSignals") -> float:
    """Per-frame reliability in [0, 1] from tracking evidence alone — cheap
    enough to attach to every frame, honest enough to gate output on."""
    if not s.initialised or s.tracking_lost:
        return 0.0
    return float(np.clip(
        0.6 * min(s.inlier_ratio / 0.5, 1.0)
        + 0.4 * min(s.tracked_landmarks / 120.0, 1.0), 0.0, 1.0))


def make_signals(frame_idx: int, img: np.ndarray, detector=None,
                 geometry: Optional[dict] = None) -> HealthSignals:
    """Assemble a HealthSignals from an image plus optional geometry metrics.

    ``geometry`` keys recognised: pnp_inliers, pnp_candidates, mean_reproj_px,
    tracked_landmarks, frame_to_frame_matches, is_keyframe, tracking_lost,
    initialised.
    """
    q = image_quality(img, detector)
    s = HealthSignals(
        frame=frame_idx,
        blur_var=q["blur_var"],
        brightness=q["brightness"],
        contrast=q["contrast"],
        texture=q["texture"],
        feature_count=q["feature_count"],
    )
    if geometry:
        s.pnp_inliers = int(geometry.get("pnp_inliers", 0))
        s.pnp_candidates = int(geometry.get("pnp_candidates", 0))
        s.mean_reproj_px = float(geometry.get("mean_reproj_px", 0.0))
        s.tracked_landmarks = int(geometry.get("tracked_landmarks", 0))
        s.frame_to_frame_matches = int(geometry.get("frame_to_frame_matches", 0))
        s.is_keyframe = bool(geometry.get("is_keyframe", False))
        s.tracking_lost = bool(geometry.get("tracking_lost", False))
        s.initialised = bool(geometry.get("initialised", False))
    return s
