# tests/test_multi_view_triangulation-minimal.py
"""Minimal checks for the 2-view triangulation helpers in slam.core.triangulation_utils.

The vendored tree exposes triangulate_between_kfs_2view plus small geometric
helpers; these tests pin the helper semantics that the full pipeline relies on.
"""
import numpy as np
import cv2
import pytest

from slam.core.triangulation_utils import (
    _P_from_K_Tcw,
    _RCt_from_Tcw,
    _angle_parallax_deg,
    _project_px,
)

K = np.array([[500.0, 0.0, 320.0],
              [0.0, 500.0, 320.0],
              [0.0, 0.0, 1.0]])


def _Tcw_from_C(R, C):
    """Build camera-from-world pose from rotation R and camera centre C."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ C
    return T


def test_pose_decomposition_roundtrip():
    rng = np.random.default_rng(0)
    yaw = 0.35
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                  [np.sin(yaw), np.cos(yaw), 0.0],
                  [0.0, 0.0, 1.0]])
    C = np.array([1.0, -0.5, 2.0])
    Tcw = _Tcw_from_C(R, C)

    R_hat, C_hat, t_hat = _RCt_from_Tcw(Tcw)
    assert np.allclose(R_hat, R, atol=1e-9)
    assert np.allclose(C_hat, C, atol=1e-9)
    assert np.allclose(t_hat, -R @ C, atol=1e-9)


def test_projection_matrix_matches_manual_projection():
    Tcw = np.eye(4)
    Tcw[:3, 3] = [0.0, 0.0, -1.0]  # camera 1 m back along z
    P = _P_from_K_Tcw(K, Tcw)
    X_w = np.array([2.0, 1.5, 8.0])
    uv_h = P @ np.append(X_w, 1.0)
    uv = uv_h[:2] / uv_h[2]

    R, t = Tcw[:3, :3], Tcw[:3, 3]
    uv_direct = _project_px(K, R, t, X_w)
    assert uv_direct is not None
    assert np.allclose(uv, uv_direct, atol=1e-6)


def test_behind_camera_projection_returns_none():
    R, C, t = _RCt_from_Tcw(np.eye(4))
    assert _project_px(K, R, t, np.array([0.0, 0.0, -1.0])) is None


def test_parallax_zero_for_pure_rotation_positive_for_baseline():
    X_w = np.array([0.0, 0.0, 5.0])

    # camera yaw about Y: a point on the optical axis keeps a ray angle of
    # exactly the rotation when only the camera rotates (pure rotation ⇒ no
    # parallax between views in the world frame, but the cam-frame ray turns)
    R1 = np.eye(3)
    th = np.deg2rad(5.0)
    R2 = np.array([[np.cos(th), 0.0, np.sin(th)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(th), 0.0, np.cos(th)]])
    uv1 = _project_px(K, R1, np.zeros(3), X_w)
    uv2_cam = _project_px(K, np.eye(3), np.zeros(3), R2.T @ X_w)
    assert _angle_parallax_deg(K, uv1, uv1) == pytest.approx(0.0, abs=1e-3)
    assert _angle_parallax_deg(K, uv1, uv2_cam) == pytest.approx(5.0, abs=1e-2)

    # 1 m baseline along X at 5 m depth: parallax ~ atan-free ray angle > 10 deg
    T2 = np.eye(4)
    T2[:3, 3] = [-1.0, 0.0, 0.0]  # camera centre at world x = +1
    R2b, C2b, t2b = _RCt_from_Tcw(T2)
    uv2 = _project_px(K, R2b, t2b, X_w)
    par = _angle_parallax_deg(K, uv1, uv2)
    assert par > 10.0
