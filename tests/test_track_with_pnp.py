# tests/test_track_with_pnp.py
"""Pose tracking via the real 2D-3D PnP entry point in slam.core.pnp_utils.

The legacy helper this file used to probe (track_with_pnp in the monocular
main) does not exist in the vendored tree; the equivalent public surface is
solve_pnp_ransac, which the AEGIS pipeline calls for per-frame tracking.
"""
import os

# --- Headless / GUI-off for tests ---
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from slam.core.pnp_utils import _pose_inverse, solve_pnp_ransac


def _K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float64)


def _Tcw(R=np.eye(3), t=np.zeros(3)):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _project(K, Tcw, pts_w):
    P = K @ Tcw[:3, :4]
    uvw = (P @ np.c_[pts_w, np.ones(len(pts_w))].T).T
    return (uvw[:, :2] / uvw[:, 2:3]).astype(np.float32)


def test_solve_pnp_ransac_recovers_pose():
    rng = np.random.default_rng(42)
    K = _K()

    N = 150
    pts_w = np.column_stack([
        rng.uniform(-2.0, 2.0, N),
        rng.uniform(-0.5, 0.5, N),
        rng.uniform(3.0, 8.0, N),
    ]).astype(np.float32)

    yaw = np.deg2rad(1.0)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                   [np.sin(yaw),  np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]])
    t_gt = np.array([0.02, -0.01, 0.12])
    Tcw_gt = _Tcw(Rz, t_gt)

    pts2d = _project(K, Tcw_gt, pts_w)
    pts2d += rng.normal(0.0, 0.5, size=pts2d.shape).astype(np.float32)

    Tcw_est, inliers = solve_pnp_ransac(pts_w, pts2d, K, ransac_px=3.0)
    assert Tcw_est is not None, "PnP should succeed"
    assert inliers.dtype == bool and len(inliers) == N
    assert inliers.sum() >= 30

    t_err = np.linalg.norm(Tcw_est[:3, 3] - t_gt)
    assert t_err < 0.05, f"translation error too high: {t_err:.3f} m"

    dR = Tcw_gt[:3, :3].T @ Tcw_est[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
    assert ang < 2.0, f"rotation error too high: {ang:.2f} deg"

    # inlier correspondences must actually reproject inside the threshold
    P = K @ Tcw_est[:3, :4]
    uv = (P @ np.c_[pts_w, np.ones(N)].T).T
    uv = uv[:, :2] / uv[:, 2:3]
    errs = np.linalg.norm(uv - pts2d, axis=1)
    assert np.median(errs[inliers]) < 3.0


def test_solve_pnp_ransac_with_outliers():
    """RANSAC must survive 30 % grossly wrong correspondences."""
    rng = np.random.default_rng(1)
    K = _K()

    N = 120
    pts_w = np.column_stack([
        rng.uniform(-2.0, 2.0, N),
        rng.uniform(-0.5, 0.5, N),
        rng.uniform(3.0, 8.0, N),
    ]).astype(np.float32)

    Tcw_gt = _Tcw(t=np.array([0.05, 0.0, 0.2]))
    pts2d = _project(K, Tcw_gt, pts_w) + rng.normal(0.0, 0.4, (N, 2))

    n_bad = N // 4
    pts2d[:n_bad] += rng.uniform(-80.0, 80.0, size=(n_bad, 2))

    Tcw_est, inliers = solve_pnp_ransac(pts_w, pts2d, K, ransac_px=3.0)
    assert Tcw_est is not None
    assert np.linalg.norm(Tcw_est[:3, 3] - Tcw_gt[:3, 3]) < 0.05
    assert inliers[:n_bad].sum() < n_bad // 2, "most outliers flagged as inliers"


def test_solve_pnp_ransac_not_enough_data():
    K = _K()
    pts3d = np.zeros((3, 3), dtype=np.float32)
    pts2d = np.zeros((3, 2), dtype=np.float32)

    Tcw_est, inliers = solve_pnp_ransac(pts3d, pts2d, K, ransac_px=3.0)
    assert Tcw_est is None
    assert isinstance(inliers, np.ndarray)
    assert len(inliers) == 0


def test_pose_inverse_roundtrip():
    T = _Tcw(np.eye(3), np.array([1.0, -2.0, 3.0]))
    Tinv = _pose_inverse(T)
    assert np.allclose(T @ Tinv, np.eye(4), atol=1e-9)
