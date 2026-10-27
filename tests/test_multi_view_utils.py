"""pytest suite for the real 2-view triangulation pipeline in
`slam.core.triangulation_utils.triangulate_between_kfs_2view`.

Covers the three behaviours the AEGIS pipeline relies on:
  1. numerical accuracy of cv2.triangulatePoints-based reconstruction,
  2. the quality gates (parallax / depth window / reprojection),
  3. landmark insertion into the project's own `Map` with observations.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from slam.core.landmark_utils import Map
from slam.core.keyframe_utils import Keyframe
from slam.core.triangulation_utils import triangulate_between_kfs_2view

K = np.array([[800.0, 0.0, 320.0],
              [0.0, 800.0, 240.0],
              [0.0, 0.0, 1.0]])

LOG = logging.getLogger("test_triangulation")
LOG.addHandler(logging.NullHandler())
LOG.propagate = False


def _Tcw_from_C(C: np.ndarray, R: np.ndarray | None = None) -> np.ndarray:
    """Camera-from-world 4x4 pose for camera centre C (identity rotation default)."""
    R = np.eye(3) if R is None else R
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ C
    return T


def _project(Tcw: np.ndarray, pts_w: np.ndarray) -> np.ndarray:
    P = K @ Tcw[:3, :]
    h = (P @ np.c_[pts_w, np.ones(len(pts_w))].T).T
    return (h[:, :2] / h[:, 2:3]).astype(np.float32)


def _make_scene(n_pts: int = 50, noise_px: float = 0.3, seed: int = 7):
    """Toy forward-facing scene observed by two cameras 1 m apart on +X."""
    rng = np.random.default_rng(seed)
    pts_w = np.column_stack([
        rng.uniform(-1.0, 1.0, n_pts),
        rng.uniform(0.2, 1.0, n_pts),
        rng.uniform(4.0, 6.0, n_pts),
    ])
    T1 = _Tcw_from_C(np.zeros(3))
    T2 = _Tcw_from_C(np.array([1.0, 0.0, 0.0]))

    uv1 = _project(T1, pts_w) + rng.normal(0.0, noise_px, (n_pts, 2))
    uv2 = _project(T2, pts_w) + rng.normal(0.0, noise_px, (n_pts, 2))

    descs = rng.integers(0, 256, size=(n_pts, 32), dtype=np.uint8)
    kps1 = [cv2.KeyPoint(float(u), float(v), 1) for u, v in uv1]
    kps2 = [cv2.KeyPoint(float(u), float(v), 1) for u, v in uv2]
    return pts_w, (T1, T2), (kps1, kps2), descs


def _keyframe(idx: int, frame_idx: int, kps, pose, desc) -> Keyframe:
    return Keyframe(idx=idx, frame_idx=frame_idx, path="",
                    kps=kps, desc=desc, pose=pose, thumb=b"")


def _args(**over):
    base = dict(use_lightglue=False, ransac_thresh=3.0,
                min_depth=0.1, max_depth=10.0)
    base.update(over)
    return SimpleNamespace(**base)


def _run(pts_w, poses, kps_pair, descs, args=None, world_map=None,
         parallax_min_deg: float = 2.0):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    kf1 = _keyframe(0, 0, kps_pair[0], poses[0], descs)
    kf2 = _keyframe(1, 1, kps_pair[1], poses[1], descs.copy())
    world_map = Map() if world_map is None else world_map
    ids = triangulate_between_kfs_2view(
        args or _args(), K, world_map, kf1, kf2, matcher, LOG,
        use_parallax_gate=True, parallax_min_deg=parallax_min_deg,
    )
    return ids, world_map


def test_triangles_points_to_centimetre_accuracy():
    pts_w, poses, kps, descs = _make_scene()
    ids, world_map = _run(pts_w, poses, kps, descs)

    assert len(ids) >= 0.8 * len(pts_w), "gates rejected too many good points"
    errs = [np.linalg.norm(world_map.points[pid].position - pts_w[pid])
            for pid in ids]
    rms = float(np.sqrt(np.mean(np.square(errs))))
    assert rms < 5e-2, f"RMS localisation error too high: {rms:.4f} m"


def test_landmarks_carry_two_observations_each():
    pts_w, poses, kps, descs = _make_scene(n_pts=30)
    ids, world_map = _run(pts_w, poses, kps, descs)
    assert ids
    for pid in ids:
        obs = world_map.points[pid].observations
        assert len(obs) == 2
        assert {o[0] for o in obs} == {0, 1}


def test_parallax_gate_blocks_reconstruction_when_demanded():
    """Typical scene parallax here is ~9-14°; demanding 20° must gate it all."""
    pts_w, poses, kps, descs = _make_scene(n_pts=30)
    ids_loose, _ = _run(pts_w, poses, kps, descs, parallax_min_deg=2.0)
    ids_strict, world_map = _run(pts_w, poses, kps, descs, parallax_min_deg=20.0)
    assert len(ids_loose) > 5
    assert len(ids_strict) < 0.2 * len(ids_loose)
    assert len(world_map.points) == len(ids_strict)


def test_depth_window_rejects_far_scene():
    """Same scene, second camera 12 m away: points leave the [0.1, 10] m
    depth window, so nothing may be inserted."""
    pts_w = np.column_stack([np.linspace(-1.0, 1.0, 30),
                             np.linspace(0.2, 1.0, 30),
                             np.full(30, 5.0)])
    rng = np.random.default_rng(3)
    T1 = _Tcw_from_C(np.zeros(3))
    T2 = _Tcw_from_C(np.array([0.0, 0.0, 12.0]))
    uv1 = _project(T1, pts_w) + rng.normal(0.0, 0.3, (30, 2))
    uv2 = _project(T2, pts_w) + rng.normal(0.0, 0.3, (30, 2))
    descs = rng.integers(0, 256, size=(30, 32), dtype=np.uint8)
    kps1 = [cv2.KeyPoint(float(u), float(v), 1) for u, v in uv1]
    kps2 = [cv2.KeyPoint(float(u), float(v), 1) for u, v in uv2]

    ids, world_map = _run(pts_w, (T1, T2), (kps1, kps2), descs)
    assert ids == []
    assert len(world_map.points) == 0
