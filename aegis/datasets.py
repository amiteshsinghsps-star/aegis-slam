"""Real-dataset adapters: TUM RGB-D -> AEGIS dataset layout.

The pipeline needs three things a research dataset does not ship in that form:
an ordered image list, pinhole intrinsics, and one ground-truth camera centre
per frame. TUM RGB-D has ``rgb.txt`` (RGB timestamps), ``groundtruth.txt`` at
100 Hz, and ``calib.txt`` with a radial-tangential distortion model.

This adapter writes a portable directory::

    out/rgb/frame_00000.png          (undistorted, because slam.core is pinhole-only)
    out/intrinsics.json              K, fps, adapter provenance
    out/groundtruth.txt              one "<timestamp> tx ty tz" line per frame

so ``aegis.run``, ``aegis.evaluate`` and the web service consume real footage
with no dataset-specific code. Frames whose nearest GT stamp is farther than
``tol`` seconds are dropped rather than paired with the wrong pose.

CLI::

    python -m aegis.datasets --src C:/tum/rgbd_dataset_freiburg1_xyz \
        --out C:/tum/prepared/fr1_xyz --max-frames 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

TUM_INDEX_URL = "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download"

# Official calibration from the TUM download page, used when calib.txt is
# absent. d = (k1, k2, p1, p2, k3).
TUM_INTRINSICS = {
    "fr1": dict(fx=517.306504, fy=516.469215, cx=318.643040, cy=255.313989,
                d=(0.262383, -0.953104, -0.005358, 0.002628, 1.163314)),
    "fr2": dict(fx=520.908620, fy=521.007327, cx=325.141442, cy=249.701764,
                d=(0.231222, -0.784899, -0.003257, -0.000105, 0.917205)),
    "fr3": dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6, d=(0.0, 0.0, 0.0, 0.0, 0.0)),
}


def _numeric_lines(path: str) -> List[List[float]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(v) for v in line.split()])
            except ValueError:
                continue
    return rows


def parse_rgb_index(path: str) -> Tuple[np.ndarray, List[str]]:
    """``rgb.txt`` -> (timestamps, relative image paths), file order preserved."""
    ts, names = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ts.append(float(parts[0]))
            names.append(parts[1])
    return np.asarray(ts, dtype=np.float64), names


def parse_groundtruth(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """TUM ground truth -> (timestamps, (N,3) camera centres).

    TUM stores ``timestamp tx ty tz qx qy qz qw``; tx/ty/tz are the camera
    position in world coordinates, which is exactly the centre the pipeline
    scores against.
    """
    rows = _numeric_lines(path)
    if not rows:
        raise ValueError(f"empty ground truth: {path}")
    arr = np.asarray([r[:4] for r in rows], dtype=np.float64)
    return arr[:, 0], arr[:, 1:4]


def nearest_indices(query: np.ndarray, reference: np.ndarray,
                    tol: float = 0.02) -> np.ndarray:
    """Index of the nearest reference stamp for each query, -1 beyond ``tol``."""
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.size == 0:
        return np.full(query.shape, -1, dtype=np.int64)
    right = np.searchsorted(reference, query)
    out = np.empty(query.shape, dtype=np.int64)
    for i, (q, r) in enumerate(zip(query, right)):
        best, best_dt = -1, tol
        for j in (r - 1, r):
            if 0 <= j < reference.size:
                dt = abs(reference[j] - q)
                if dt <= best_dt:
                    best, best_dt = int(j), dt
        out[i] = best
    return out


def _camera_family(seq_dir: str) -> Optional[str]:
    name = os.path.basename(os.path.normpath(seq_dir)).lower()
    for fam in ("freiburg1", "freiburg2", "freiburg3"):
        if fam in name:
            return fam.replace("freiburg", "fr")
    return None


def tum_intrinsics(seq_dir: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """(K, distortion, provenance) from calib.txt, else the official table."""
    calib = os.path.join(seq_dir, "calib.txt")
    if os.path.isfile(calib):
        rows = _numeric_lines(calib)
        if rows and len(rows[0]) >= 4:
            fx, fy, cx, cy = rows[0][:4]
            d = rows[0][4:9]
            d = list(d) + [0.0] * (5 - len(d))
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            return K, np.asarray(d, dtype=np.float64), "calib.txt"
    fam = _camera_family(seq_dir)
    if fam not in TUM_INTRINSICS:
        raise ValueError(
            f"no calib.txt and unknown camera family for {seq_dir!r}; "
            "pass --fx/--fy/--cx/--cy or rename the directory to include freiburg1/2/3")
    c = TUM_INTRINSICS[fam]
    K = np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]],
                 dtype=np.float64)
    return K, np.asarray(c["d"], dtype=np.float64), f"official {fam} table"


def prepare_tum(src_dir: str, out_dir: str, max_frames: Optional[int] = None,
                stride: int = 1, tol: float = 0.02, undistort: bool = True,
                K_override: Optional[Sequence[float]] = None) -> dict:
    """Materialise a TUM sequence as an AEGIS dataset directory."""
    rgb_txt = os.path.join(src_dir, "rgb.txt")
    gt_txt = os.path.join(src_dir, "groundtruth.txt")
    for fp in (rgb_txt, gt_txt):
        if not os.path.isfile(fp):
            raise FileNotFoundError(fp)
    ts, names = parse_rgb_index(rgb_txt)
    gt_ts, gt_xyz = parse_groundtruth(gt_txt)
    if K_override is not None:
        fx, fy, cx, cy = [float(v) for v in K_override[:4]]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist, source = np.zeros(5), "override"
    else:
        K, dist, source = tum_intrinsics(src_dir)

    order = np.arange(len(names))[::max(1, int(stride))]
    if max_frames:
        order = order[:int(max_frames)]
    matched_all = nearest_indices(ts[order], gt_ts, tol=tol)
    keep = matched_all >= 0
    dropped = int((~keep).sum())
    if not np.any(keep):
        raise ValueError(f"no RGB frame has ground truth within {tol:g}s")
    order, matched = order[keep], matched_all[keep]

    rgb_out = os.path.join(out_dir, "rgb")
    os.makedirs(rgb_out, exist_ok=True)
    D = dist.reshape(-1, 1) if undistort and float(np.abs(dist).max()) > 1e-9 else None
    kept_ts = []
    for i, (src_i, gt_i) in enumerate(zip(order, matched)):
        img = cv2.imread(os.path.join(src_dir, names[src_i]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(names[src_i])
        if D is not None:
            img = cv2.undistort(img, K, D)
        cv2.imwrite(os.path.join(rgb_out, f"frame_{i:05d}.png"), img)
        kept_ts.append(float(ts[src_i]))

    kept_ts = np.asarray(kept_ts, dtype=np.float64)
    fps = float(1.0 / np.median(np.diff(kept_ts))) if len(kept_ts) > 1 else 30.0
    with open(os.path.join(out_dir, "groundtruth.txt"), "w", encoding="utf-8") as f:
        for t, c in zip(kept_ts, gt_xyz[matched]):
            f.write(f"{t:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
    seq = os.path.basename(os.path.normpath(src_dir))
    ic = {
        "K": K.tolist(), "fps": round(fps, 3), "model": "pinhole",
        "adapter": "tum", "sequence": seq, "undistorted": bool(D is not None),
        "distortion": [float(v) for v in dist], "intrinsics_from": source,
        "source": TUM_INDEX_URL,
    }
    with open(os.path.join(out_dir, "intrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(ic, f, indent=2)
    return {
        "sequence": seq, "frames": int(len(order)),
        "dropped_no_gt": dropped,
        "fps": round(fps, 2), "K": K.tolist(),
        "undistorted": bool(D is not None), "intrinsics_from": source,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser("AEGIS dataset adapter")
    ap.add_argument("--kind", default="tum", choices=["tum"])
    ap.add_argument("--src", required=True, help="extracted TUM sequence directory")
    ap.add_argument("--out", required=True, help="output AEGIS dataset directory")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="max |rgb stamp - gt stamp| in seconds (TUM associate default)")
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    args = ap.parse_args(argv)

    override = None
    if args.fx is not None:
        if None in (args.fy, args.cx, args.cy):
            ap.error("--fx needs --fy --cx --cy")
        override = [args.fx, args.fy, args.cx, args.cy]
    os.makedirs(args.out, exist_ok=True)
    info = prepare_tum(args.src, args.out, max_frames=args.max_frames,
                       stride=args.stride, tol=args.tol,
                       undistort=not args.no_undistort, K_override=override)
    print(f"[datasets] {info['sequence']}: {info['frames']} frames "
          f"({info['dropped_no_gt']} dropped, no GT within {args.tol:g}s), "
          f"fps={info['fps']}, undistorted={info['undistorted']}")
    print(f"[datasets] wrote {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
