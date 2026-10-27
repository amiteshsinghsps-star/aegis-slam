# -*- coding: utf-8 -*-
"""
ba_utils.py
===========

ALWAYS PASS POSE as T_wc (world→camera) (camera-from-world) pose convention to pyceres.

Light-weight Bundle-Adjustment helpers built on **pyceres**.
Implements
  • two_view_ba(...)
  • pose_only_ba(...)
  • local_bundle_adjustment(...)
and keeps the older run_bundle_adjustment (full BA) for
back-compatibility.

A *blue-print* for global_bundle_adjustment is included but not wired.
"""
from __future__ import annotations
import cv2
import numpy as np
import pyceres
from pycolmap import cost_functions, CameraModelId
import math
from scipy.spatial.transform import Rotation as R

import logging
logger = logging.getLogger("ba")


# TODO: USES T_cw (camera-from-world) convention for storing poses in the map, because of PyCERES, REMEMBER TO CONVERT, use below functions
from slam.core.pose_utils import _pose_inverse, _pose_to_quat_trans, _quat_trans_to_pose

# TODO : UNCOMMENT below block IF YOU USE USES World-from-camera (camera-to-world) pose convention T_wc (camera→world) for storing poses in the map.
# from slam.core.pose_utils import (
#     _pose_inverse,           # T ↔ T⁻¹
#     _pose_to_quat_trans as _cw_to_quat_trans,
#     _quat_trans_to_pose as _quat_trans_to_cw,
# )
# def _pose_to_quat_trans(T_wc):
#     """
#     Convert **camera→world** pose T_wc to the (q_cw, t_cw) parameterisation
#     expected by COLMAP/pyceres residuals.
#     """
#     return _cw_to_quat_trans(_pose_inverse(T_wc))


# def _quat_trans_to_pose(q_cw, t_cw):
#     """
#     Convert optimised (q_cw, t_cw) back to a **camera→world** SE3 matrix.
#     """
#     return _pose_inverse(_quat_trans_to_cw(q_cw, t_cw))

# --------------------------------------------------------------------- #
#  Shared helper to add one reprojection residual
# --------------------------------------------------------------------- #
# --- ba_utils.py: entry points now use kfs instead of a separate keypoints list ---
#
# NOTE (2026-10 code review): the pycolmap.cost_functions ↔ pyceres bridge is
# dead in this environment (pyceres 2.6 / pycolmap 4.2.1 — see colmap/colmap
# #4273): residuals are silently evaluated as 0.0, so BA "converges" instantly
# without moving anything.  We therefore provide a pure-Python cost with
# analytical Jacobians; pyceres' Python Evaluate() callback is verified to
# drive the solver correctly.
#
# Camera block layout: one 7-dof vector [qx, qy, qz, qw, tx, ty, tz] of the
# world→camera pose T_cw (scipy 'xyzw' quaternion).  Layout validated against
# pycolmap ReprojErrorCost numerics on this machine.
def _pack_cam7(Tcw: np.ndarray) -> np.ndarray:
    quat, trans = _pose_to_quat_trans(Tcw, ordering="xyzw")
    return np.ascontiguousarray(np.concatenate([quat, trans]), dtype=np.float64)


def _unpack_cam7(cam7: np.ndarray) -> np.ndarray:
    q = np.asarray(cam7[:4], dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    return _quat_trans_to_pose(q, np.asarray(cam7[4:7], dtype=float),
                               ordering="xyzw")


class _ReprojCost(pyceres.CostFunction):
    """Pinhole reprojection residual (2) over [point3d(3), cam7(7), intr4(4)]."""

    def __init__(self, uv_obs: np.ndarray):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([3, 7, 4])
        self.obs = np.asarray(uv_obs, dtype=np.float64).ravel()

    def Evaluate(self, parameters, residual, jacobian):
        X = np.asarray(parameters[0], dtype=np.float64).ravel()
        cam = np.asarray(parameters[1], dtype=np.float64).ravel()
        fx, fy, cx, cy = np.asarray(parameters[2], dtype=np.float64).ravel()
        q, tvec = cam[:4], cam[4:7]

        def project(Xv, camv, fx_, fy_, cx_, cy_):
            qv = camv[:4]; tv = camv[4:7]
            nn = float(np.dot(qv, qv))
            if nn <= 0.0:
                return None
            s = 1.0 / np.sqrt(nn)
            x_, y_, z_, w_ = qv[0] * s, qv[1] * s, qv[2] * s, qv[3] * s
            Rm = np.array([
                [1 - 2*(y_*y_ + z_*z_),     2*(x_*y_ - z_*w_),      2*(x_*z_ + y_*w_)],
                [    2*(x_*y_ + z_*w_),  1 - 2*(x_*x_ + z_*z_),     2*(y_*z_ - x_*w_)],
                [    2*(x_*z_ - y_*w_),      2*(y_*z_ + x_*w_),  1 - 2*(x_*x_ + y_*y_)],
            ])
            Xc = Rm @ Xv + tv
            if not np.isfinite(Xc[2]) or Xc[2] <= 1e-8:
                return None
            return (fx_ * Xc[0] / Xc[2] + cx_, fy_ * Xc[1] / Xc[2] + cy_), Xc, Rm

        pr = project(X, cam, fx, fy, cx, cy)
        if pr is None:
            return False
        (u, v), Xc, R = pr
        residual[0] = u - self.obs[0]
        residual[1] = v - self.obs[1]

        if jacobian is not None:
            X_, Y_, Z_ = Xc
            du_dXc = np.array([fx / Z_, 0.0, -fx * X_ / (Z_ * Z_)])
            dv_dXc = np.array([0.0, fy / Z_, -fy * Y_ / (Z_ * Z_)])
            # point block: dXc/dX = R
            Jp = np.array([du_dXc @ R[:, 0], du_dXc @ R[:, 1], du_dXc @ R[:, 2],
                           dv_dXc @ R[:, 0], dv_dXc @ R[:, 1], dv_dXc @ R[:, 2]]
                          ).reshape(2, 3)
            # camera block: analytic for translation, central differences for quat
            Jc = np.zeros((2, 7))
            Jc[0, 4] = du_dXc[0]; Jc[0, 5] = du_dXc[1]; Jc[0, 6] = du_dXc[2]
            Jc[1, 4] = dv_dXc[0]; Jc[1, 5] = dv_dXc[1]; Jc[1, 6] = dv_dXc[2]
            h = 1e-6
            for j in range(4):
                cp = cam.copy(); cp[j] += h
                cm = cam.copy(); cm[j] -= h
                pp = project(X, cp, fx, fy, cx, cy)
                pm = project(X, cm, fx, fy, cx, cy)
                if pp is None or pm is None:
                    return False
                Jc[0, j] = (pp[0][0] - pm[0][0]) / (2.0 * h)
                Jc[1, j] = (pp[0][1] - pm[0][1]) / (2.0 * h)
            # intrinsics block
            Ji = np.array([[X_ / Z_, 0.0, 1.0, 0.0],
                           [0.0, Y_ / Z_, 0.0, 1.0]])
            for i, blk in enumerate((Jp, Jc, Ji)):
                jb = jacobian[i]
                if jb.ndim == 1:                    # pyceres hands flat row-major buffers
                    jb[:] = blk.ravel()
                else:
                    jb[:, :] = blk
        return True


def _sync_pose(kf, new_Tcw: np.ndarray, world_map=None, idx: int = None) -> None:
    """Publish an optimised T_cw pose.  The runtime Keyframe holds the *same*
    array object as world_map.poses[k] (Map.add_pose copies the caller's
    matrix once), so writing in-place keeps every view consistent; containers
    whose Keyframe translates conventions rebind instead and sync themselves."""
    if (world_map is not None and idx is not None
            and idx < len(world_map.poses)
            and world_map.poses[idx] is kf.pose):
        kf.pose[:] = new_Tcw
    else:
        kf.pose = new_Tcw


def _add_reproj_edge(problem, loss_fn,
                     kp_uv: tuple[float, float],
                     cam_param, point_param, intr_param):
    """Create a pyceres residual block for one observation."""
    problem.add_residual_block(
        _ReprojCost(np.asarray(kp_uv, np.float64)),
        loss_fn,
        [point_param, cam_param, intr_param]
    )

# --------------------------------------------------------------------- #
#  1) Two-view BA  (bootstrap refinement)
# --------------------------------------------------------------------- #
# --- ba_utils.py: entry points now use kfs instead of a separate keypoints list ---
def two_view_ba(world_map, K, kfs, max_iters: int = 20):
    """
    Refine the two initial camera poses + all bootstrap landmarks.
    Expects KF indices 0 and 1 to exist.
    """
    assert len(world_map.poses) >= 2, "two_view_ba expects at least 2 poses"
    _core_ba(world_map, K, kfs,
             opt_kf_idx=[0, 1],
             fix_kf_idx=[],
             max_iters=max_iters,
             info_tag="[2-view BA]")

# --------------------------------------------------------------------- #
#  2) Pose-only BA   (current frame refinement)
# --------------------------------------------------------------------- #
def pose_only_ba(world_map, K, kfs,
                 kf_idx: int, max_iters: int = 8,
                 huber_thr: float = 2.0):
    """
    Optimise only one **keyframe** pose while keeping all 3-D points fixed.
    """
    import pyceres
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    intr = np.array([fx, fy, cx, cy], np.float64)

    # read pose from keyframe (authoritative), fall back to map if needed
    Tcw = kfs[kf_idx].pose
    cam = _pack_cam7(Tcw)

    problem = pyceres.Problem()
    problem.add_parameter_block(cam, 7)

    problem.add_parameter_block(intr, 4)
    problem.set_parameter_block_constant(intr)

    loss_fn = pyceres.HuberLoss(huber_thr)

    # add residuals for observations in this KF
    added = 0
    for mp in world_map.points.values():
        # point is constant in pose-only BA
        problem.add_parameter_block(mp.position, 3)
        problem.set_parameter_block_constant(mp.position)
        for f_idx, kp_idx, _desc in mp.observations:
            if f_idx != kf_idx: 
                continue
            u, v = kfs[f_idx].kps[kp_idx].pt
            _add_reproj_edge(problem, loss_fn, (u, v), cam, mp.position, intr)
            added += 1

    if added < 10:
        logger.warning("[Pose-only BA] skipped – not enough residuals")
        return

    opts = pyceres.SolverOptions()
    opts.max_num_iterations = max_iters
    opts.logging_type = pyceres.LoggingType.SILENT
    summary = pyceres.SolverSummary()
    pyceres.solve(opts, problem, summary)

    # write back to the keyframe (authoritative pose holder); runtime Keyframes
    # share the very array objects stored in world_map.poses, so in-place
    # writes keep the map consistent without a second, convention-dependent write
    new_Tcw = _unpack_cam7(cam)
    _sync_pose(kfs[kf_idx], new_Tcw, world_map, kf_idx)
    logger.debug("[Pose-only BA] iters=%s residuals=%d", getattr(summary, "num_successful_steps", 0), added)


# --------------------------------------------------------------------- #
#  3) Local BA  (sliding window)
# --------------------------------------------------------------------- #
def local_bundle_adjustment(world_map, K, kfs,
                            center_kf_idx: int,
                            window_size: int = 6,
                            max_points  : int = 10000,
                            max_iters   : int = 15):
    """
    Optimise a sliding window of keyframes around center_kf_idx plus
    all landmarks they observe. Older KFs are fixed (gauge).
    """
    first_opt = max(1, center_kf_idx - window_size + 1)
    opt_kf    = list(range(first_opt, center_kf_idx + 1))
    fix_kf    = list(range(0, first_opt))
    logger.debug("[Local BA window] | opt_kf=%s fix_kf=%s center=%d", opt_kf, fix_kf, center_kf_idx)

    _core_ba(world_map, K, kfs,
             opt_kf_idx = opt_kf,
             fix_kf_idx = fix_kf,
             max_points = max_points,
             max_iters  = max_iters,
             info_tag   = f"[Local BA @ KF {center_kf_idx}]")

# --------------------------------------------------------------------- #
#  4) Global BA  (all keyframes + points)
# --------------------------------------------------------------------- #
def global_bundle_adjustment(
    world_map, K, kfs,
    *,
    fix_first: bool = True,
    max_points: int | None = 30000,
    max_iters: int = 30
):
    """
    Run a full bundle adjustment over *all* keyframes and all landmarks
    observed by them. By default we fix the very first keyframe to anchor
    the gauge (otherwise the solution drifts by a similarity).

    Parameters
    ----------
    world_map : Map
        Your map container (points + poses). Poses are **T_cw**.
    K : (3,3) ndarray
        Intrinsic matrix of the (pinhole) camera.
    kfs : list[Keyframe]
        The keyframe list. Each KF stores pose in **T_cw**.
    fix_first : bool
        If True, fix KF-0 pose to anchor gauge freedom.
    max_points : int | None
        Optional cap on the number of 3-D points optimised (for speed).
    max_iters : int
        Ceres iterations.

    Returns
    -------
    None  (poses and points are updated in-place)
    """
    if len(kfs) < 2:
        logger.warning("[Global BA] skipped – need at least 2 keyframes")
        return

    opt_kf_idx = list(range(len(kfs)))
    fix_kf_idx = [0] if (fix_first and len(kfs) > 0) else []

    _core_ba(
        world_map, K, kfs,
        opt_kf_idx=opt_kf_idx,
        fix_kf_idx=fix_kf_idx,
        max_points=max_points,
        max_iters=max_iters,
        info_tag=f"[Global BA] | KFs={len(kfs)} pts≤{max_points}]"
    )

# --------------------------------------------------------------------- #
#  Shared low-level BA engine
# --------------------------------------------------------------------- #
def _core_ba(world_map, K, kfs,
             *,
             opt_kf_idx: list[int],
             fix_kf_idx: list[int],
             max_points: int | None = None,
             max_iters : int = 20,
             info_tag  : str = ""):
    """
    Sparse BA over a subset of poses (keyframes) + points.
    Poses come from kfs[idx].pose; residual UVs from kfs[idx].kps[kp_idx].pt
    """
    import pyceres
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    intr = np.array([fx, fy, cx, cy], np.float64)

    problem = pyceres.Problem()
    loss_fn = pyceres.HuberLoss(2.0)

    problem.add_parameter_block(intr, 4)
    problem.set_parameter_block_constant(intr)

    # pose blocks (opt + fixed) from Keyframe poses — one 7-dof CAM block each
    cam_params = {}
    for k in opt_kf_idx:
        cam = _pack_cam7(kfs[k].pose)
        cam_params[k] = cam
        problem.add_parameter_block(cam, 7)

    for k in fix_kf_idx:
        cam = _pack_cam7(kfs[k].pose)
        cam_params[k] = cam
        problem.add_parameter_block(cam, 7)
        problem.set_parameter_block_constant(cam)

    # point blocks + residuals
    added_pts = 0
    added_res = 0
    for mp in world_map.points.values():
        # keep only points seen by at least one optimisable KF
        if not any(f in opt_kf_idx for f, _, _ in mp.observations):
            continue
        if max_points and added_pts >= max_points:
            continue

        problem.add_parameter_block(mp.position, 3)
        added_pts += 1

        for f_idx, kp_idx, _desc in mp.observations:
            if (f_idx not in opt_kf_idx) and (f_idx not in fix_kf_idx):
                continue
            u, v = kfs[f_idx].kps[kp_idx].pt
            _add_reproj_edge(problem, loss_fn,
                             (u, v),
                             cam_params[f_idx],
                             mp.position,
                             intr)
            added_res += 1

    if added_res < 10:
        logger.warning("%s skipped – not enough residuals", info_tag)
        return

    opts = pyceres.SolverOptions()
    opts.max_num_iterations = max_iters
    # opts.minimizer_progress_to_stdout = True # make true to debug
    opts.minimizer_progress_to_stdout = False
    opts.logging_type = pyceres.LoggingType.SILENT
    summary = pyceres.SolverSummary()
    pyceres.solve(opts, problem, summary)

    # write poses back → Keyframes + Map (in-place: aliases stay live)
    for k in opt_kf_idx:
        new_Tcw = _unpack_cam7(cam_params[k])
        _sync_pose(kfs[k], new_Tcw, world_map, k)

    iters = getattr(summary, "iterations_used",
             getattr(summary, "num_successful_steps",
             getattr(summary, "num_iterations", None)))
    logger.info("%s iters=%s  chi2=%.2f  residuals=%d",
                info_tag, str(iters), float(getattr(summary, "final_cost", float("nan"))), added_res)