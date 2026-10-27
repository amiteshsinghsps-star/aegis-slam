"""Per-component timing benchmark (PRD v2 U3).

Times the OpenCV-5 calls the AEGIS pipeline actually makes, on real frames
from a dataset, and reports p50/p95 plus throughput:

    python -m aegis.bench [--data Dataset/synthetic] [--frames 120] [--reps 3]

Run this on the Graviton host to replace "<1 ms" style claims with measured
numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import cv2
import numpy as np


def _pct(vals: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(vals, dtype=float), q)) if vals else 0.0


def bench(data_dir: str, n_frames: int, reps: int, out: str | None) -> dict:
    from aegis.health import make_signals
    from aegis.robust_geometry import find_fundamental
    from slam.core.pnp_utils import solve_pnp_ransac

    manifest = json.load(open(os.path.join(data_dir, "manifest.json"))) \
        if os.path.exists(os.path.join(data_dir, "manifest.json")) else {}
    intr = os.path.join(data_dir, "intrinsics.json")
    K = np.eye(3)
    if os.path.exists(intr):
        c = json.load(open(intr))
        if "K" in c:
            K = np.array(c["K"], float)
        elif "fx" in c:
            K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], float)

    rgb_dir = os.path.join(data_dir, "rgb")
    names = sorted(f for f in os.listdir(rgb_dir) if f.lower().endswith((".png", ".jpg")))
    names = names[:max(2, n_frames)]
    imgs = [cv2.imread(os.path.join(rgb_dir, n)) for n in names]
    imgs = [i for i in imgs if i is not None]
    if len(imgs) < 2:
        raise SystemExit(f"need >=2 readable frames under {rgb_dir}")

    orb = cv2.ORB_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    T0 = np.eye(4)
    T1 = np.eye(4); T1[:3, :3] = cv2.Rodrigues(np.array([0.01, 0.02, 0.03]))[0]; T1[:3, 3] = [0.05, 0.0, 0.02]
    rng = np.random.default_rng(0)
    pts3d = rng.uniform(-0.5, 0.5, size=(200, 3)) + np.array([0, 0, 4.0])
    Xc = T0[:3, :3] @ pts3d.T + T0[:3, 3:]
    uv0 = (K @ (Xc / Xc[2:]))[:2].T
    Xc1 = T1[:3, :3] @ pts3d.T + T1[:3, 3:]
    uv1 = (K @ (Xc1 / Xc1[2:]))[:2].T
    kp0 = [cv2.KeyPoint(float(u), float(v), 15) for u, v in uv0]
    kp1 = [cv2.KeyPoint(float(u), float(v), 15) for u, v in uv1]
    # fake but well-behaved descriptors so the matcher produces a usable graph
    d0 = rng.integers(0, 256, size=(len(kp0), 32), dtype=np.uint8)
    d1 = d0.copy()
    ms = [cv2.DMatch(i, i, 0, 0.0) for i in range(len(kp0))]
    p0 = np.float32([kp0[m.queryIdx].pt for m in ms])
    p1 = np.float32([kp1[m.trainIdx].pt for m in ms])

    t = {}
    def timeit(label, fn, items):
        for _ in range(reps):
            for it in items:
                s = time.perf_counter()
                fn(it)
                t.setdefault(label, []).append((time.perf_counter() - s) * 1000.0)

    for f in imgs:
        timeit("health_signals", lambda x: make_signals(0, x, orb, {}), [f])
        timeit("orb_detect_compute", lambda x: orb.detectAndCompute(x, None), [f])
        timeit("clahe", lambda x: cv2.createCLAHE(2.0, (8, 8)).apply(
            cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)), [f])
    half = imgs[:max(1, len(imgs) // 2)]
    for f in half:
        timeit("nl_means", lambda x: cv2.fastNlMeansDenoising(
            cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), None, 10, 7, 21), [f])

    timeit("bf_match_knn", lambda _: bf.knnMatch(d0, d1, k=2), [0])
    timeit("findFundamentalMat_ransac",
           lambda _: cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 3.0, 0.99), [0])
    timeit("find_fundamental_usac",
           lambda _: find_fundamental(p0, p1, estimator="usac", thresh=3.0), [0])
    timeit("findEssentialMat",
           lambda _: cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, 3.0), [0])
    timeit("solvePnPRansac",
           lambda _: solve_pnp_ransac(pts3d, np.float32([kp1[m.trainIdx].pt for m in ms]),
                                      K, 3.0, Tcw_init=T0), [0])

    rows = {k: {"p50_ms": round(_pct(v, 50), 3), "p95_ms": round(_pct(v, 95), 3), "n": len(v)}
            for k, v in sorted(t.items())}
    frames_per_sec = 1000.0 / max(sum(r["p50_ms"] for r in rows.values()), 1e-6)
    report = {
        "dataset": data_dir, "frames_used": len(imgs), "reps": reps,
        "arch": os.uname().machine if hasattr(os, "uname") else os.environ.get("PROCESSOR_ARCHITECTURE", ""),
        "opencv": cv2.__version__,
        "components_ms": rows,
        "sequential_sum_fps_est": round(frames_per_sec, 2),
    }
    for k, v in rows.items():
        print(f"{k:32s} p50={v['p50_ms']:8.3f} ms  p95={v['p95_ms']:8.3f} ms  (n={v['n']})")
    print(f"\nsum-of-components fps estimate: {report['sequential_sum_fps_est']}")
    if out:
        json.dump(report, open(out, "w"), indent=2)
        print(f"wrote {out}")
    return report


def main():
    ap = argparse.ArgumentParser(prog="AEGIS-SLAM bench")
    ap.add_argument("--data", default="Dataset/synthetic")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="eval/bench.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    bench(a.data, a.frames, a.reps, a.out)


if __name__ == "__main__":
    main()
