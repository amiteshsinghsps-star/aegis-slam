"""Command-line entry point for AEGIS-SLAM.

Examples
  # supervised run on the built-in synthetic dataset
  python -m aegis.run --data Dataset/synthetic --mode supervised --out runs/demo

  # custom uploaded video, classical baseline
  python -m aegis.run --data Dataset/custom.mp4 --mode baseline_classical

  # a specific fault scenario under the agent
  python -m aegis.run --data Dataset/synthetic --mode supervised --fault motion_blur
"""
from __future__ import annotations

import argparse
import json
import os

from aegis.pipeline import load_source, run_pipeline
from aegis.tools import RuntimeConfig
from aegis.supervisor import SupervisorParams
from aegis.faults import FaultPlan, FaultSegment


def main():
    ap = argparse.ArgumentParser("AEGIS-SLAM run")
    ap.add_argument("--data", default="Dataset/synthetic")
    ap.add_argument("--mode", default="supervised",
                    choices=["baseline_classical", "baseline_learned", "supervised", "supervised_multi", "ablation"])
    ap.add_argument("--out", default="runs/cli")
    ap.add_argument("--frames", type=int, default=0, help="0 = all")
    ap.add_argument("--backend", default="orb", choices=["orb", "sift", "aliked"])
    ap.add_argument("--estimator", default="ransac",
                    choices=["ransac", "usac", "usac_magsac"],
                    help="robust geometry estimator for match gating (OpenCV-5 USAC family)")
    ap.add_argument("--fault", default=None,
                    choices=["motion_blur", "low_light", "texture_loss", "frame_drop"])
    ap.add_argument("--fault_lo", type=float, default=0.35)
    ap.add_argument("--fault_hi", type=float, default=0.65)
    ap.add_argument("--severity", type=float, default=1.0,
                    help="fault level 0.05..1.0: blur kernel length, darkness gamma "
                         "and texture-removal radius scale with it")
    ap.add_argument("--cadence", type=int, default=10)
    ap.add_argument("--llm", action="store_true",
                    help="consult the optional Bedrock planner for ambiguous diagnoses")
    ap.add_argument("--llm-model", default=None,
                    help="Bedrock model id (env AEGIS_BEDROCK_MODEL; "
                         "default anthropic.claude-3-haiku-20240307-v1:0 — "
                         "verify model availability in your region)")
    ap.add_argument("--llm-region", default=None)
    args = ap.parse_args()

    src = load_source(args.data)
    if args.frames:
        src.frames = src.frames[:args.frames]
        if src.gt_centers is not None:
            src.gt_centers = src.gt_centers[:args.frames]

    cfg = RuntimeConfig(backend=args.backend, robust_estimator=args.estimator)
    sup = SupervisorParams(cadence=args.cadence)

    fault = None
    if args.fault:
        n = len(src.frames)
        sev = min(1.0, max(0.05, args.severity))
        fault = FaultPlan([FaultSegment(args.fault, int(n * args.fault_lo),
                                        int(n * args.fault_hi), severity=sev)])

    llm = None
    if args.llm:
        from aegis.llm import BedrockPlanner
        llm = BedrockPlanner(model_id=args.llm_model
                             or os.environ.get("AEGIS_BEDROCK_MODEL")
                             or "anthropic.claude-3-haiku-20240307-v1:0",
                             region=args.llm_region, enabled=True)

    res = run_pipeline(src, args.out, mode=args.mode, cfg=cfg, sup_params=sup,
                       fault=fault, llm=llm)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
