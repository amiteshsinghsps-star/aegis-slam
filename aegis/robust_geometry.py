"""Robust geometry helpers built on OpenCV 5 (section 8 of the PRD).

The reused SimpleSLAM bootstrap uses plain RANSAC. AEGIS-SLAM exposes the
OpenCV-5 **USAC** estimator family as an agent-selectable robust backend:

  * ``usac``          — USAC_ACCURATE: unified sampling + ACS + MLESAC
                        refinement; finds more true inliers under repetitive
                        texture and moving objects (``dynamic_or_repetitive``).
  * ``usac_magsac``   — USAC_MAGSAC: *threshold-free* multi-consensus
                        sampling. It integrates over all consensus sizes, so
                        it needs no hand-tuned px threshold and is markedly
                        more stable when the noise scale itself is uncertain
                        — exactly the ``motion_blur`` regime, where epipolar
                        error spreads with the blur kernel.

``filter_matches`` accepts two (N,2) point arrays and returns the model + a
boolean inlier mask, choosing the estimator from the runtime config.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _method(estimator: str, kind: str):
    estimator = (estimator or "ransac").lower()
    if estimator in ("usac", "magsac", "usac_accurate"):
        return getattr(cv2, "USAC_ACCURATE", cv2.RANSAC if kind == "homography" else cv2.FM_RANSAC)
    if estimator in ("usac_magsac", "magsac_rts"):
        m = getattr(cv2, "USAC_MAGSAC", None)
        if m is not None:
            return m
        return cv2.RANSAC if kind == "homography" else cv2.FM_RANSAC
    return cv2.RANSAC if kind == "homography" else cv2.FM_RANSAC


def usac_support() -> dict:
    """What robust estimators this OpenCV build actually provides
    (surfaced by the service /version endpoint)."""
    return {
        "usac_accurate": hasattr(cv2, "USAC_ACCURATE"),
        "usac_magsac": hasattr(cv2, "USAC_MAGSAC"),
        "opencv": cv2.__version__,
    }


def find_fundamental(pts1: np.ndarray, pts2: np.ndarray, estimator: str = "ransac",
                     thresh: float = 2.5, conf: float = 0.99
                     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(pts1) < 8:
        return None, None
    m = _method(estimator, "fundamental")
    try:
        F, mask = cv2.findFundamentalMat(pts1, pts2, m, thresh, conf)
    except cv2.error:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, thresh, conf)
    return F, mask


def find_homography(pts1: np.ndarray, pts2: np.ndarray, estimator: str = "ransac",
                    thresh: float = 2.5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(pts1) < 4:
        return None, None
    m = _method(estimator, "homography")
    try:
        H, mask = cv2.findHomography(pts1, pts2, m, thresh)
    except cv2.error:
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, thresh)
    return H, mask


def filter_matches(kp1, kp2, matches, estimator: str = "ransac", thresh: float = 2.5) -> list:
    """Geometric outlier rejection for a list of cv2.DMatch, honoring the
    configured robust estimator. Falls back to fundamental-matrix gating."""
    if len(matches) < 8:
        return matches
    p1 = np.float32([kp1[m.queryIdx].pt for m in matches]).astype(np.float64)
    p2 = np.float32([kp2[m.trainIdx].pt for m in matches]).astype(np.float64)
    _, mask = find_fundamental(p1, p2, estimator=estimator, thresh=thresh)
    if mask is None:
        return []
    mask = mask.ravel().astype(bool)
    return [mm for mm, ok in zip(matches, mask) if ok]
