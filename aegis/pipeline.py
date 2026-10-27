"""AEGIS-SLAM supervised pipeline.

This reuses the OpenCV-5 SLAM core from ``slam.core`` (two-view bootstrap, PnP
tracking, keyframes, triangulation, local bundle adjustment) and wraps it in the
agent supervisor. It is a deliberate, self-contained re-implementation of the
monocular loop (rather than editing ``slam/monocular/main_revamped.py``) so the
reused-versus-new boundary stays clean and the supervisor can act on the loop.

Modes (section 11 conditions):
  * baseline_classical — fixed OpenCV features, no agent
  * baseline_learned   — fixed ALIKED+LightGlue, no agent
  * supervised         — full agent (observe→diagnose→act→verify→rollback)
  * ablation           — agent without verify/rollback

Outputs per run: trajectory (.npz), decision trace (.json), health timeline,
trusted/untrusted segments, ATE + coverage metrics, a 2D trajectory PNG, and
evidence frames for re-capture requests.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from slam.core.pose_utils import _pose_inverse, _pose_rt_to_homogenous
from slam.core.features_utils import (
    init_feature_pipeline, feature_extractor, feature_matcher, filter_matches_ransac)
from slam.core.keyframe_utils import Keyframe, select_keyframe, make_thumb
from slam.core.landmark_utils import Map
from slam.core.two_view_bootstrap import (
    InitParams, pts_from_matches, evaluate_two_view_bootstrap_with_masks, bootstrap_two_view_map)
from slam.core.pnp_utils import (
    predict_pose_const_vel, reproject_and_match_2d3d, solve_pnp_ransac)
from slam.core.triangulation_utils import triangulate_between_kfs_2view
from aegis import robust_geometry

try:
    from slam.core.ba_utils import local_bundle_adjustment
    _HAS_BA = True
except Exception:  # pragma: no cover
    local_bundle_adjustment = None
    _HAS_BA = False

from aegis.health import HealthSignals, make_signals, frame_confidence
from aegis.tools import RuntimeConfig, apply_preprocess, invoke
from aegis.supervisor import Supervisor, SupervisorParams
from aegis.multi import MultiSupervisor
from aegis.trace import DecisionTrace, json_safe
from aegis.approval import ApprovalGate
from aegis.geom import camera_center, ate_rmse, align_trajectories

log = logging.getLogger("aegis")


# --------------------------------------------------------------------------- #
#  Lightweight args namespace consumed by the reused slam.core functions
# --------------------------------------------------------------------------- #
def _backend_family(backend: str) -> str:
    """Descriptor families are not interchangeable: a binary (ORB) map point
    cannot be matched against float (SIFT) or learned (ALIKED) descriptors."""
    if backend == "aliked":
        return "learned"
    if backend in ("orb", "akaze"):
        return "binary"
    return "float"


@dataclass
class _Args:
    use_lightglue: bool = False
    detector: str = "orb"
    matcher: str = "bf"
    max_features: int = 4000
    min_conf: float = 0.7
    ransac_thresh: float = 2.5
    pnp_min_inliers: int = 30
    proj_radius: float = 10.0
    min_depth: float = 0.40
    max_depth: float = 100.0
    mvt_rep_err: float = 2.0
    kf_max_disp: float = 45.0
    kf_min_inliers: float = 150.0
    kf_min_ratio: float = 0.35
    kf_min_rot_deg: float = 8.0
    kf_cooldown: int = 5
    kf_thumb_hw: List[int] = field(default_factory=lambda: [640, 360])
    local_ba_min_new_points: int = 60
    local_ba_window: int = 10
    local_ba_max_points: int = 5000
    local_ba_max_iters: int = 12

    def sync_from_cfg(self, cfg: RuntimeConfig):
        self.use_lightglue = (cfg.backend == "aliked")
        self.detector = cfg.backend if cfg.backend in ("orb", "sift") else "orb"
        self.max_features = cfg.max_features
        self.min_conf = cfg.min_conf
        self.ransac_thresh = cfg.ransac_thresh
        self.pnp_min_inliers = cfg.pnp_min_inliers


# --------------------------------------------------------------------------- #
#  Frame source
# --------------------------------------------------------------------------- #
@dataclass
class FrameSource:
    frames: List           # list of str paths OR in-memory ndarray frames
    K: np.ndarray
    gt_centers: Optional[np.ndarray]   # (N,3) camera centres (world), may be None
    meta: dict


def _load_groundtruth_centers(gt_path: str) -> np.ndarray:
    ts, centers = [], []
    with open(gt_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            ts.append(float(v[0]))
            centers.append([float(v[1]), float(v[2]), float(v[3])])
    return np.asarray(centers, dtype=np.float64)


def load_source(path: str, K: Optional[np.ndarray] = None) -> FrameSource:
    """Load a synthetic dataset dir, an mp4, or a directory of images."""
    if os.path.isdir(path):
        intr_fp = os.path.join(path, "intrinsics.json")
        rgb_dir = os.path.join(path, "rgb")
        if os.path.isdir(rgb_dir):
            names = sorted(os.listdir(rgb_dir))
            frames = [os.path.join(rgb_dir, n) for n in names if n.lower().endswith((".png", ".jpg"))]
        else:
            frames = [os.path.join(path, n) for n in sorted(os.listdir(path))
                      if n.lower().endswith((".png", ".jpg"))]
        if K is None and os.path.isfile(intr_fp):
            with open(intr_fp) as f:
                K = np.asarray(json.load(f)["K"], dtype=np.float64)
        gt_fp = os.path.join(path, "groundtruth.txt")
        gt = _load_groundtruth_centers(gt_fp) if os.path.isfile(gt_fp) else None
        if K is None:
            raise ValueError("No intrinsics: provide intrinsics.json or pass K")
        meta = {"type": "dir", "path": path}
        if os.path.isfile(intr_fp):
            with open(intr_fp) as f:
                ic = json.load(f)
            for key in ("seed", "fps", "model"):
                if key in ic:
                    meta[key] = ic[key]
        return FrameSource(frames, K, gt, meta)

    if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        max_frames = int(os.environ.get("AEGIS_MAX_FRAMES", "900"))
        max_w = int(os.environ.get("AEGIS_MAX_WIDTH", "960"))
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            if fr.shape[1] > max_w:                      # bound memory: ~max_frames * max_w * h * 3 bytes
                s = max_w / float(fr.shape[1])
                fr = cv2.resize(fr, (max_w, int(round(fr.shape[0] * s))), interpolation=cv2.INTER_AREA)
            frames.append(fr)
        cap.release()
        if not frames:
            raise ValueError("could not decode any frames from the video")
        if K is None:
            h, w = frames[0].shape[:2]
            fx = 1.2 * w
            K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)
        return FrameSource(frames, K, None, {"type": "video", "path": path})

    raise FileNotFoundError(path)


def _read(frame):
    if isinstance(frame, np.ndarray):
        return frame.copy()
    img = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(frame)
    return img


def _to_u8_bgr(img):
    if img is None:
        return None
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and float(img.max()) <= 1.01:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def _reproj_mean(K, Tcw, pts3d, pts2d, mask):
    if pts3d is None or len(pts3d) == 0 or mask is None or not np.any(mask):
        return 0.0
    p3 = np.asarray(pts3d, dtype=np.float64)[mask]
    p2 = np.asarray(pts2d, dtype=np.float64)[mask]
    Xc = (Tcw[:3, :3] @ p3.T).T + Tcw[:3, 3]
    z = Xc[:, 2]
    ok = z > 1e-6
    if not np.any(ok):
        return 0.0
    uv = (K @ (Xc[ok] / z[ok, None]).T).T[:, :2]
    return float(np.mean(np.linalg.norm(uv - p2[ok], axis=1)))


def run_pipeline(source: FrameSource, out_dir: str, mode: str = "supervised",
                 cfg: Optional[RuntimeConfig] = None,
                 sup_params: Optional[SupervisorParams] = None,
                 fault=None, save_images: bool = True,
                 trace: Optional[DecisionTrace] = None,
                 gate: Optional[ApprovalGate] = None,
                 progress_cb=None, llm=None) -> dict:
    """Run one pipeline. ``fault`` is an optional FaultPlan applied per frame.

    ``trace``/``gate`` may be injected so the web service can tail the decision
    trace live and hold re-capture requests for a human. ``progress_cb(frame,
    total)`` is called once per processed frame.
    """
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.perf_counter()
    cfg = cfg or RuntimeConfig()
    sup_params = sup_params or SupervisorParams()

    baseline = mode.startswith("baseline")
    agent = not baseline
    verify = (mode != "ablation")
    multi = mode == "supervised_multi"
    if mode == "baseline_learned":
        cfg.backend = "aliked"
    elif mode == "baseline_classical":
        cfg.backend = "orb"
        cfg.preprocess = "none"

    args = _Args()
    args.sync_from_cfg(cfg)

    # build the initial feature pipeline
    detector, matcher = init_feature_pipeline(args)
    backend_key = (cfg.backend, cfg.max_features)

    K = source.K
    frames = source.frames
    total = len(frames)

    world_map = Map()
    kfs: List[Keyframe] = []
    last_kf_frame_no = -999
    initialised = False
    Tcw_cur = np.eye(4)

    bs_has_ref = False
    bs_kps_ref = bs_des_ref = bs_img_ref = None
    bs_frame_ref = -1

    family = _backend_family(cfg.backend)
    family_hold: Optional[dict] = None   # map built under the other family

    trace = trace or DecisionTrace()
    gate = gate or ApprovalGate(default_timeout_s=0.5, auto="skip")
    supervisor = ((MultiSupervisor if multi else Supervisor)(
        cfg, sup_params, mode=("baseline" if baseline else "supervised"),
        llm=llm, verify=verify) if agent else None)
    if llm is not None:
        sup_params.allow_llm = True

    signals: List[HealthSignals] = []
    est_traj: List[np.ndarray] = []
    est_frames: List[int] = []
    per_frame_health: List[dict] = []
    confidences: List[float] = []
    recapture_evidence: List[str] = []

    def maybe_rebuild_backend():
        nonlocal detector, matcher, backend_key, args, kp_prev, des_prev, bs_has_ref
        nonlocal family, family_hold, initialised, last_kf_frame_no
        args.sync_from_cfg(cfg)
        new_key = (cfg.backend, cfg.max_features)
        if new_key == backend_key:
            return
        new_family = _backend_family(cfg.backend)
        try:
            detector, matcher = init_feature_pipeline(args)
            backend_key = new_key
        except Exception as e:
            log.warning("backend rebuild failed (%s); keeping previous", e)
            return
        # Cached descriptors belong to the previous backend. Re-extract them
        # so the matcher never sees e.g. ORB's 32-byte rows where LightGlue
        # expects 128-float ALIKED rows (AssertionError), and drop the
        # bootstrap reference built with the old descriptor family.
        kp_prev, des_prev = feature_extractor(args, img_prev, detector)
        bs_has_ref = False
        if new_family == family:
            return
        if family_hold is not None and family_hold["family"] == new_family:
            # Switching back (e.g. verification rolled the config back): the
            # old-family map was stashed at switch time, so reinstate it
            # instead of rebuilding from scratch.
            snap = family_hold["map"]
            family_hold = None
            _restore_snapshot(world_map, kfs, snap)
            del est_traj[snap["est_len"]:]
            del est_frames[snap["est_len"]:]
            last_kf_frame_no = snap["last_kf_frame_no"]
            # a snapshot taken before bootstrap has fewer than two poses; do
            # not pretend the map is initialised in that case
            initialised, bs_has_ref = len(world_map.poses) >= 2, False
            log.info("[Agent] backend family %s restored; map snapshot reinstated", new_family)
        else:
            family_hold = {"family": family,
                           "map": _take_snapshot(world_map, kfs, last_kf_frame_no, len(est_traj))}
            _do_rebootstrap(world_map, kfs)
            initialised = False
            bs_has_ref = False
            log.info("[Agent] backend family %s -> %s; map reset for re-bootstrap",
                     family, new_family)
        family = new_family

    img_prev = _to_u8_bgr(_read(frames[0]))
    if fault is not None:
        img_prev = fault.apply(img_prev, 0)
    kp_prev, des_prev = feature_extractor(args, img_prev, detector)

    for fi in range(1, total):
        maybe_rebuild_backend()
        img_cur = _to_u8_bgr(_read(frames[fi]))
        if fault is not None:
            img_cur = fault.apply(img_cur, fi)
        # preprocessing decision (agent-controlled) applied to the raw frame
        img_cur_proc = apply_preprocess(img_cur, cfg.preprocess)
        kp_cur, des_cur = feature_extractor(args, img_cur_proc, detector)

        new_ids: List[int] = []
        tracking_lost = False
        pnp_inl = pnp_cand = 0
        reproj = 0.0

        # geometric frame-to-frame matches with the configured robust estimator
        if cfg.robust_estimator.startswith("usac") and len(kp_prev) > 8 and len(kp_cur) > 8 and des_prev is not None:
            raw = feature_matcher(args, kp_prev, kp_cur, des_prev, des_cur, matcher)
            prev_cur_matches = robust_geometry.filter_matches(
                kp_prev, kp_cur, raw, estimator="usac", thresh=cfg.ransac_thresh)
        else:
            raw = feature_matcher(args, kp_prev, kp_cur, des_prev, des_cur, matcher)
            prev_cur_matches = filter_matches_ransac(kp_prev, kp_cur, raw, cfg.ransac_thresh)

        # ------------------------- BOOTSTRAP ------------------------- #
        if not initialised:
            if not bs_has_ref:
                bs_has_ref = True
                bs_kps_ref, bs_des_ref, bs_img_ref, bs_frame_ref = kp_cur, des_cur, img_cur_proc, fi
                img_prev, kp_prev, des_prev = img_cur_proc, kp_cur, des_cur
                continue
            raw_bs = feature_matcher(args, bs_kps_ref, kp_cur, bs_des_ref, des_cur, matcher)
            matches_bs = filter_matches_ransac(bs_kps_ref, kp_cur, raw_bs, cfg.ransac_thresh)
            if len(matches_bs) < 80 or (fi - bs_frame_ref) > 30:
                bs_kps_ref, bs_des_ref, bs_img_ref, bs_frame_ref = kp_cur, des_cur, img_cur_proc, fi
                img_prev, kp_prev, des_prev = img_cur_proc, kp_cur, des_cur
                continue
            pts_ref, pts_cur = pts_from_matches(bs_kps_ref, kp_cur, matches_bs)
            params = InitParams(ransac_px=cfg.ransac_thresh, min_posdepth=0.90,
                                min_parallax_deg=0.5, score_ratio_H=0.45)
            decision = evaluate_two_view_bootstrap_with_masks(K, pts_ref, pts_cur, params)
            if decision is None:
                img_prev, kp_prev, des_prev = img_cur_proc, kp_cur, des_cur
                continue
            ok, T0, T1 = bootstrap_two_view_map(
                K, bs_kps_ref, bs_des_ref, kp_cur, des_cur, matches_bs, args,
                world_map, params=params, decision=decision)
            if not ok:
                img_prev, kp_prev, des_prev = img_cur_proc, kp_cur, des_cur
                continue
            world_map.add_pose(T0, is_keyframe=True)
            world_map.add_pose(T1, is_keyframe=True)
            kf0 = Keyframe(0, bs_frame_ref, "", bs_kps_ref, bs_des_ref, T0, make_thumb(bs_img_ref, tuple(args.kf_thumb_hw)))
            kf1 = Keyframe(1, fi, "", kp_cur, des_cur, T1, make_thumb(img_cur_proc, tuple(args.kf_thumb_hw)))
            kfs.extend([kf0, kf1])
            last_kf_frame_no = fi
            est_traj.append(camera_center(T0)); est_traj.append(camera_center(T1))
            est_frames.extend([bs_frame_ref, fi])
            initialised = True
            bs_has_ref = False
            Tcw_cur = T1.copy()
            log.info("[Init] Bootstrapped at frame %d (map pts=%d)", fi, len(world_map))
            img_prev, kp_prev, des_prev = img_cur_proc, kp_cur, des_cur
            continue

        # ------------------------- PnP TRACKING ------------------------- #
        if len(world_map.poses) >= 2:
            Tcw_pred = predict_pose_const_vel(world_map.poses[-2], world_map.poses[-1])
        else:
            Tcw_pred = Tcw_cur
        H, W = img_cur_proc.shape[:2]
        m23d = reproject_and_match_2d3d(world_map, K, Tcw_pred, kp_cur, des_cur, W, H,
                                        radius_px=args.proj_radius, max_hamm=64)
        pnp_cand = int(len(m23d.pts3d))
        is_kf = False
        if pnp_cand >= cfg.pnp_min_inliers:
            Tcw_est, inl = solve_pnp_ransac(m23d.pts3d, m23d.pts2d, K,
                                            ransac_px=cfg.ransac_thresh, Tcw_init=Tcw_pred,
                                            iters=300, conf=0.999)
            pnp_inl = int(inl.sum()) if inl.size else 0
            if Tcw_est is not None and pnp_inl >= cfg.pnp_min_inliers:
                Tcw_cur = Tcw_est
                world_map.add_pose(Tcw_cur, is_keyframe=False)
                est_traj.append(camera_center(Tcw_cur))
                est_frames.append(fi)
                reproj = _reproj_mean(K, Tcw_cur, m23d.pts3d, m23d.pts2d, inl)
            else:
                tracking_lost = True
        else:
            tracking_lost = True

        # ------------------------- 2D-2D FALLBACK ------------------------- #
        if tracking_lost:
            if len(prev_cur_matches) >= 5:
                p0 = np.float32([kp_prev[m.queryIdx].pt for m in prev_cur_matches])
                p1 = np.float32([kp_cur[m.trainIdx].pt for m in prev_cur_matches])
                if cfg.robust_estimator.startswith("usac"):
                    _, inlE = robust_geometry.find_fundamental(p0, p1, estimator="usac", thresh=cfg.ransac_thresh)
                    E, inlE2 = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, cfg.ransac_thresh)
                    if E is None or E.shape != (3, 3):
                        fallback_inliers = 0
                    elif inlE is not None and int(inlE.sum()) >= 5:
                        maskE = inlE.ravel().astype(np.uint8) * 255
                        ninl, R, t, inlPose = cv2.recoverPose(E, p0, p1, K, mask=maskE.astype(maskE.dtype))
                        fallback_inliers = int(ninl)
                    else:
                        ninl, R, t, inlPose = cv2.recoverPose(E, p0, p1, K, mask=inlE2)
                        fallback_inliers = int(ninl)
                else:
                    E, inlE = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, cfg.ransac_thresh)
                    if E is not None and inlE is not None and int(inlE.sum()) >= 5:
                        fallback_inliers, R, t, inlPose = cv2.recoverPose(E, p0, p1, K, mask=inlE)
                        fallback_inliers = int(fallback_inliers)
                    else:
                        fallback_inliers = 0
                if fallback_inliers >= 5:
                    T_rel = _pose_rt_to_homogenous(R, t)
                    if len(world_map.poses) >= 2:
                        T_rp = world_map.poses[-1] @ _pose_inverse(world_map.poses[-2])
                        T_rel[:3, 3] = t.ravel() * float(np.linalg.norm(T_rp[:3, 3]))
                    else:
                        T_rel[:3, 3] = 0.0
                    Tcw_cur = T_rel @ world_map.poses[-1]
                    world_map.add_pose(Tcw_cur, is_keyframe=False)
                    est_traj.append(camera_center(Tcw_cur))
                    est_frames.append(fi)
                    tracking_lost = False

        # ------------------------- KEYFRAME + MAP GROWTH ------------------------- #
        prev_len = len(kfs)
        kfs, last_kf_frame_no = select_keyframe(
            args, frames, fi - 1, img_cur_proc, kp_cur, des_cur, Tcw_cur, matcher, kfs, last_kf_frame_no)
        is_kf = (len(kfs) > prev_len)
        if is_kf and len(kfs) >= 2:
            new_ids = triangulate_between_kfs_2view(
                args, K, world_map, kfs[-2], kfs[-1], matcher, log,
                use_parallax_gate=True, parallax_min_deg=2.0, reproj_px_max=cfg.ransac_thresh)
            if new_ids and _HAS_BA and len(new_ids) >= args.local_ba_min_new_points:
                try:
                    local_bundle_adjustment(world_map, K, kfs, center_kf_idx=kfs[-1].idx,
                                            window_size=args.local_ba_window,
                                            max_points=args.local_ba_max_points,
                                            max_iters=args.local_ba_max_iters)
                except Exception as e:
                    log.debug("[BA] local BA skipped: %s", e)

        # ------------------------- HEALTH + SUPERVISOR ------------------------- #
        geo = {
            "pnp_inliers": pnp_inl, "pnp_candidates": pnp_cand, "mean_reproj_px": reproj,
            "tracked_landmarks": len(world_map), "frame_to_frame_matches": len(prev_cur_matches),
            "is_keyframe": is_kf, "tracking_lost": tracking_lost, "initialised": True,
        }
        sig = make_signals(fi, img_cur_proc, detector, geo)
        signals.append(sig)
        d = sig.to_dict()
        d["confidence"] = frame_confidence(sig)
        confidences.append(d["confidence"])
        per_frame_health.append(d)

        if supervisor is not None and (fi % sup_params.cadence == 0) and len(signals) >= sup_params.window:
            window = signals[-sup_params.window:]
            supervisor.step(window, trace, image=img_cur_proc, detector=detector)
            # consume control signals
            if cfg._restore_requested and cfg._snapshot is not None:
                snap = cfg._snapshot
                _restore_snapshot(world_map, kfs, snap)
                del est_traj[snap["est_len"]:]
                del est_frames[snap["est_len"]:]
                last_kf_frame_no = snap["last_kf_frame_no"]
                initialised, bs_has_ref = True, True
                cfg._restore_requested = False
                log.info("[Agent] Rolled back re-bootstrap — map restored at frame %d", fi)
            if cfg.rebootstrap_requested:
                cfg.rebootstrap_requested = False
                cfg._snapshot = _take_snapshot(world_map, kfs, last_kf_frame_no, len(est_traj))
                _do_rebootstrap(world_map, kfs)
                initialised = False
                bs_has_ref = False
                log.info("[Agent] Re-bootstrap at frame %d (map snapshot taken)", fi)
            if cfg._recapture is not None:
                req = cfg._recapture
                cfg._recapture = None
                approval = gate.request(req["frame"], req["reason"], req["evidence"])
                status = approval.wait()  # offline: resolves via auto=skip/timeout
                if status == "approved":
                    fp = os.path.join(out_dir, f"recapture_{req['frame']}.jpg")
                    cv2.imwrite(fp, img_cur_proc)
                    recapture_evidence.append(fp)
                else:
                    invoke("mark_segment_untrusted", cfg,
                           {"start": req["frame"] - sup_params.window, "end": req["frame"]}, req["frame"])
                trace.add(frame=fi, evidence=sig.to_dict(), diagnosis=req["reason"],
                          decided_by="rules", action={"tool": "request_recapture",
                          "args": {"approval": status,
                                   "approved_by": approval.decided_by,
                                   "resolved_at": approval.resolved_at}},
                          config_before={},
                          config_after=cfg.snapshot(), outcome={"verdict": status})

        img_prev, kp_prev, des_prev = img_cur_proc, kp_cur, des_cur
        if progress_cb is not None:
            try:
                progress_cb(fi, total)
            except Exception:
                pass
    return _finalize(source, world_map, kfs, est_traj, cfg, trace, out_dir,
                     per_frame_health, recapture_evidence, mode, save_images,
                     confidences, est_frames, t_start=t_start)


def _do_rebootstrap(world_map: Map, kfs: List[Keyframe]) -> None:
    world_map.points.clear()
    world_map.poses.clear()
    world_map.keyframe_indices.clear()
    world_map._next_pid = 0
    kfs.clear()


def _take_snapshot(world_map: Map, kfs: List[Keyframe],
                   last_kf_frame_no: int, est_len: int) -> dict:
    """Shallow container snapshot: re-bootstrap only clears the containers, and
    new points/poses created afterwards are fresh objects, so restoring the
    original dicts/lists genuinely undoes the destructive action."""
    return {
        "points": dict(world_map.points),
        "poses": list(world_map.poses),
        "kf_indices": list(world_map.keyframe_indices),
        "next_pid": getattr(world_map, "_next_pid", 0),
        "kfs": list(kfs),
        "last_kf_frame_no": int(last_kf_frame_no),
        "est_len": int(est_len),
    }


def _restore_snapshot(world_map: Map, kfs: List[Keyframe], snap: dict) -> None:
    world_map.points.clear()
    world_map.points.update(snap["points"])
    world_map.poses[:] = snap["poses"]
    world_map.keyframe_indices[:] = snap["kf_indices"]
    world_map._next_pid = snap["next_pid"]
    kfs[:] = snap["kfs"]


def _trusted(mask_untrusted: List[List[int]], n: int) -> np.ndarray:
    t = np.ones(n, dtype=bool)
    for lo, hi in mask_untrusted:
        t[max(0, lo):min(n, hi + 1)] = False
    return t


def _finalize(source, world_map, kfs, est_traj, cfg, trace, out_dir,
              per_frame_health, recapture_evidence, mode, save_images,
              confidences=None, est_frames=None, t_start=None) -> dict:
    est = np.asarray(est_traj, dtype=np.float64) if est_traj else np.empty((0, 3))
    n_frames = len(source.frames)
    trusted = _trusted(cfg.untrusted, n_frames)
    coverage = float(trusted.mean())

    est_aligned = None
    a = {"rmse": float("nan"), "median": float("nan"), "max": float("nan"),
         "scale": float("nan"), "n": len(est)}
    ate_all = {"rmse": float("nan"), "n": 0}
    ate_trusted = {"rmse": float("nan"), "n": 0}
    frame_err = np.empty(0, dtype=np.float64)
    if source.gt_centers is not None and len(est) >= 3:
        # pair each estimated pose with the ground truth of ITS OWN frame —
        # the trajectory starts at bootstrap, index alignment would skew ATE
        gt = np.asarray(source.gt_centers, dtype=np.float64)
        if est_frames and len(est_frames) == len(est):
            idx = [f for f in est_frames if 0 <= f < len(gt)]
            ref = gt[idx]
        else:
            ref = gt[:len(est)]
            idx = list(range(len(est)))
        a = ate_rmse(est, ref)
        if len(est) == len(ref) and len(est) >= 3:
            est_aligned = align_trajectories(est, ref)[0]   # Sim(3): same alignment the ATE uses
            frame_err = np.linalg.norm(est_aligned - ref, axis=1)
            # Primary metric: every frame from the first pose onward. Frames
            # without an estimate are filled with the last known pose, so a
            # trajectory that drops tracking is penalised instead of silently
            # shrinking the evaluation set.
            if idx:
                start = max(0, int(idx[0]))
                full = np.empty((len(gt), 3), dtype=np.float64)
                last = est_aligned[0]
                k = 0
                for f in range(start, len(gt)):
                    if k < len(idx) and idx[k] == f:
                        last = est_aligned[k]
                        k += 1
                    full[f] = last
                ate_all = ate_rmse(full[start:], gt[start:])
                ate_all["n"] = len(gt) - start
            # Trusted-frame ATE: the same alignment restricted to frames the
            # agent did not mark untrusted. Always reported next to coverage;
            # a low trusted ATE with low coverage is not a win.
            keep = [k for k, f in enumerate(idx)
                    if 0 <= f < n_frames and trusted[f]]
            if len(keep) >= 3:
                ate_trusted = ate_rmse(est_aligned[keep], ref[keep])
                ate_trusted["n"] = len(keep)

    conf = np.asarray(confidences, dtype=np.float64) if confidences else np.empty(0, np.float64)

    summary = {
        "mode": mode,
        "frames": n_frames,
        "wall_seconds": (round(time.perf_counter() - t_start, 2)
                         if t_start else None),
        "seed": (source.meta or {}).get("seed"),
        "source_fps": (source.meta or {}).get("fps"),
        "initialised": len(world_map.poses) >= 2,
        "num_keyframes": len(kfs),
        "num_landmarks": len(world_map),
        "poses_estimated": len(est),
        "coverage_trusted": coverage,
        "coverage_estimated": (len(est) / n_frames) if n_frames else 0.0,
        "untrusted_segments": cfg.untrusted,
        "ate": a,
        "ate_all": ate_all,
        "ate_trusted": ate_trusted,
        "recapture_requests": len(recapture_evidence),
        "confidence": ({"mean": float(conf.mean()), "p10": float(np.percentile(conf, 10)),
                        "min": float(conf.min())} if conf.size else None),
        "trace_summary": trace.summary(),
        "final_config": cfg.snapshot(),
    }
    trace.dump(os.path.join(out_dir, "trace.json"))
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(json_safe(summary), f, indent=2)
    np.savez(os.path.join(out_dir, "trajectory.npz"),
             est=est, est_aligned=(est_aligned if est_aligned is not None else est),
             gt=(source.gt_centers if source.gt_centers is not None else np.empty((0, 3))),
             untrusted=np.asarray(cfg.untrusted, dtype=np.int64),
             conf=conf,
             est_frames=np.asarray(est_frames if est_frames else [], dtype=np.int64),
             frame_err=frame_err)
    with open(os.path.join(out_dir, "health_timeline.json"), "w") as f:
        json.dump(json_safe(per_frame_health), f)

    _save_traj_png(est_aligned if est_aligned is not None else est, source.gt_centers, cfg.untrusted,
                   os.path.join(out_dir, "trajectory.png"),
                   conf if conf.size == len(est) else None)
    if save_images and recapture_evidence:
        pass

    log.info("[Done] mode=%s poses=%d landmarks=%d coverage=%.2f ATE=%.3f decisions=%d",
             mode, len(est), len(world_map), coverage,
             a.get("rmse", float("nan")), len(trace.records))
    return summary


def _save_traj_png(est, gt, untrusted, path, conf=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        if gt is not None and len(gt):
            ax.plot(gt[:, 0], gt[:, 2], "-", color="gray", label="ground truth")
        if len(est):
            if conf is not None and len(conf) == len(est):
                sc = ax.scatter(est[:, 0], est[:, 2], c=conf, cmap="RdYlGn",
                                vmin=0.0, vmax=1.0, s=14, zorder=3)
                ax.plot(est[:, 0], est[:, 2], "-", color="tab:blue", lw=0.8, alpha=0.5)
                fig.colorbar(sc, ax=ax, label="per-frame confidence")
            else:
                ax.plot(est[:, 0], est[:, 2], "o-", color="tab:blue", ms=2, label="estimated")
        if len(est) and untrusted:
            t = _trusted(untrusted, len(est))
            ax.plot(est[~t, 0], est[~t, 2], "x", color="tab:red", ms=6, label="untrusted")
        ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_title("AEGIS-SLAM trajectory (x-z, colored by confidence)")
        ax.legend(); ax.axis("equal")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.debug("traj png skipped: %s", e)
