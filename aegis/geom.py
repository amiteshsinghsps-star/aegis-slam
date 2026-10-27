"""Geometry + trajectory-evaluation helpers for AEGIS-SLAM.

These are thin, dependency-light utilities built on NumPy/OpenCV so the
evaluation harness and the agent supervisor can reason about poses without
pulling in the whole visualization stack.

Pose convention (matches the reused ``slam.core`` code):
  * ``T_cw`` is a 4x4 *camera-from-world* matrix:  x_cam = R_cw @ X_w + t_cw.
  * The camera centre in world coordinates is      C_w   = -R_cw^T @ t_cw.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  Pose conversions
# --------------------------------------------------------------------------- #
def pose_inverse(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous rigid transform."""
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def camera_center(T_cw: np.ndarray) -> np.ndarray:
    """Camera centre in world coords from a camera-from-world pose."""
    T_cw = np.asarray(T_cw, dtype=np.float64)
    R, t = T_cw[:3, :3], T_cw[:3, 3]
    return -R.T @ t


def centers_from_poses(poses: Sequence[np.ndarray]) -> np.ndarray:
    """Stack per-pose camera centres into an (N,3) array."""
    if len(poses) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack([camera_center(T) for T in poses], axis=0)


def is_rotation(R: np.ndarray, tol: float = 1e-3) -> bool:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return False
    return (np.allclose(R @ R.T, np.eye(3), atol=tol)
            and abs(np.linalg.det(R) - 1.0) < tol)


def rotation_about_axis(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    """Rodrigues formula."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / (np.linalg.norm(k) + 1e-12)
    x, y, z = k
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


# --------------------------------------------------------------------------- #
#  Sim(3) alignment (Umeyama) + ATE
# --------------------------------------------------------------------------- #
def umeyama_alignment(src: np.ndarray, dst: np.ndarray,
                      with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares Sim(3)/SE(3) that maps ``src`` onto ``dst``.

    Returns (scale, R, t) with  dst ≈ scale * R @ src + t.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[1] == 3, "need matching (N,3)"

    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = (d_c.T @ s_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    var_s = (s_c ** 2).sum() / len(src)
    if with_scale:
        scale = float(np.trace(np.diag(D) @ S) / (var_s + 1e-12))
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def align_trajectories(est: np.ndarray, ref: np.ndarray,
                       with_scale: bool = True) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Align ``est`` to ``ref`` and return (aligned_est, scale, R, t)."""
    scale, R, t = umeyama_alignment(est, ref, with_scale=with_scale)
    est_aligned = scale * (R @ est.T).T + t
    return est_aligned, scale, R, t


def ate_rmse(est: np.ndarray, ref: np.ndarray,
             with_scale: bool = True) -> dict:
    """Absolute Trajectory Error after Sim(3)/SE(3) alignment.

    Monocular SLAM has no metric scale, so ``with_scale=True`` (Sim(3)) is the
    fair default. Returns rmse, mean, median, max and the recovered scale.
    """
    est = np.asarray(est, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = min(len(est), len(ref))
    if n < 3:
        return {"rmse": float("nan"), "mean": float("nan"), "median": float("nan"),
                "max": float("nan"), "scale": float("nan"), "n": n}
    est, ref = est[:n], ref[:n]
    est_aligned, scale, _, _ = align_trajectories(est, ref, with_scale=with_scale)
    err = np.linalg.norm(est_aligned - ref, axis=1)
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "scale": float(scale),
        "n": int(n),
    }
