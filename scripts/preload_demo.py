"""Record read-only demo runs for the judge-access fallback (PRD U2).

Generates (if missing) the synthetic dataset and runs three supervised jobs —
clean, motion_blur, frame_drop — into docs/demo/{clean,blur,framedrop} with
evidence frames so the UI presets and /preloaded endpoints work without
starting a job.

    python -m scripts.preload_demo [--frames 90]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aegis.pipeline import load_source, run_pipeline
from aegis.faults import FaultPlan, FaultSegment

DEMO_DIR = "docs/demo"
SCENARIOS = {
    "clean": None,
    "blur": FaultPlan([FaultSegment("motion_blur", 0, 0, severity=1.0)]),
    "framedrop": FaultPlan([FaultSegment("frame_drop", 0, 0, severity=1.0)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Dataset/synthetic")
    ap.add_argument("--frames", type=int, default=90)
    a = ap.parse_args()

    if not os.path.isdir(a.data):
        print("generating synthetic dataset…")
        subprocess.run([sys.executable, "-m", "aegis.synth", "--out", a.data,
                        "--frames", "200"], check=True)

    src = load_source(a.data)
    src.frames = src.frames[:a.frames]
    if src.gt_centers is not None:
        src.gt_centers = src.gt_centers[:a.frames]
    n = len(src.frames)
    lo, hi = int(n * 0.35), int(n * 0.65)

    os.makedirs(DEMO_DIR, exist_ok=True)
    for name, fault in SCENARIOS.items():
        if fault is not None:
            fault.segments = [FaultSegment(fault.segments[0].kind, lo, hi,
                                           severity=fault.segments[0].severity)]
        out = os.path.join(DEMO_DIR, name)
        shutil.rmtree(out, ignore_errors=True)
        print(f"running demo: {name} ({a.frames} frames)…")
        res = run_pipeline(src, out, mode="supervised", fault=fault, save_images=True)
        ate = (res.get("ate") or {}).get("rmse")
        print(f"  {name}: ate_rmse={ate} coverage={res['coverage_trusted']} "
              f"decisions={res['trace_summary']['num_decisions']}")
    idx = {"runs": [d for d in sorted(os.listdir(DEMO_DIR))
                    if os.path.isfile(os.path.join(DEMO_DIR, d, "result.json"))]}
    with open(os.path.join(DEMO_DIR, "index.json"), "w") as f:
        json.dump(idx, f, indent=2)
    print(f"preloaded {len(idx['runs'])} demo runs under {DEMO_DIR}/")


if __name__ == "__main__":
    main()
