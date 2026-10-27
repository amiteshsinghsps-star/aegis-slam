import numpy as np
import pytest

from aegis.geom import (umeyama_alignment, align_trajectories,
                             ate_rmse, pose_inverse, camera_center,
                             rotation_about_axis)


def _random_poses(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * np.array([1.0, 0.5, 3.0])


def test_umeyama_identity():
    pts = _random_poses()
    s, R, t = umeyama_alignment(pts, pts, with_scale=True)
    assert s == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(R, np.eye(3), atol=1e-9)
    assert np.allclose(t, np.zeros(3), atol=1e-9)


def test_umeyama_roundtrip_scale_rotate_translate():
    rng = np.random.default_rng(1)
    src = _random_poses(50, seed=1)
    s_true = 2.5
    R_true = rotation_about_axis(np.array([0.1, 0.9, 0.3]), 0.6)
    t_true = np.array([1.0, -2.0, 0.5])
    dst = s_true * (R_true @ src.T).T + t_true
    s, R, t = umeyama_alignment(src, dst, with_scale=True)
    assert s == pytest.approx(s_true, rel=1e-6)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(t, t_true, atol=1e-6)


def test_ate_rmse_recovers_known_transform():
    est = _random_poses(60, seed=2)
    R = rotation_about_axis(np.array([0, 0, 1.0]), 0.3)
    ref = (R @ est.T).T + np.array([0.4, 0.1, -0.2])
    out = ate_rmse(est, ref)
    assert out["rmse"] < 1e-6
    assert out["n"] == 60


def test_ate_rmse_positive_for_noise():
    est = _random_poses(60, seed=3)
    ref = est + 0.05 * np.random.default_rng(4).normal(size=est.shape)
    out = ate_rmse(est, ref)
    assert out["rmse"] > 0.01


def test_pose_inverse_roundtrip():
    rng = np.random.default_rng(5)
    R = rotation_about_axis(rng.normal(size=3), rng.uniform(-1, 1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.normal(size=3)
    assert np.allclose(pose_inverse(pose_inverse(T)), T, atol=1e-10)


def test_camera_center_consistency():
    # For T_cw = [R|t], center C = -R^T t, and R C + t = 0
    R = rotation_about_axis(np.array([1.0, 0, 0]), 0.7)
    t = np.array([0.2, -0.5, 1.3])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    C = camera_center(T)
    assert np.allclose(R @ C + t, np.zeros(3), atol=1e-10)
