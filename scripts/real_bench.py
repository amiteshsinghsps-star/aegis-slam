"""Real-footage benchmark: TUM RGB-D sequences, clean and fault-injected.

``aegis.evaluate`` measures the agent on synthetic scenes, where ground truth
is exact and the fault model is the same code that injects it. This harness
runs the same conditions on real handheld video prepared by
``python -m aegis.datasets`` so the report can quote third-party numbers, and
it injects the same fault kernels mid-sequence so the supervisor's value can be
measured against the classical baseline on imagery it was never tuned on.

One run per (sequence, fault, condition) cell: real sequences are the scenes,
so there is no seed expansion. ATE is Umeyama Sim(3) on camera centres, the
same metric as the synthetic grid.

Usage::

    python -m aegis.datasets --src C:/tum/rgbd_dataset_freiburg1_xyz \
        --out C:/tum/prepared/fr1_xyz --max-frames 300
    python scripts/real_bench.py --data C:/tum/prepared/fr1_xyz \
        --data C:/tum/prepared/fr1_desk --frames 300 --jobs 2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import traceback
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.evaluate import _load_timeline, _recovery_frames  # noqa: E402
from aegis.faults import FaultPlan, FaultSegment  # noqa: E402
from aegis.pipeline import load_source, run_pipeline  # noqa: E402
from aegis.supervisor import SupervisorParams  # noqa: E402

FAULTS = ("clean", "motion_blur", "low_light", "texture_loss")


def _record(seq: str, fault: str, cond: str, res: dict, out_dir: str) -> dict:
    n = int(res.get("frames") or 0)
    recover = None
    if fault != "clean" and n > 0:
        recover = _recovery_frames(_load_timeline(out_dir), int(n * 0.35),
                                   int(n * 0.65), n)
    return {
        "sequence": seq, "fault": fault, "condition": cond,
        "ate": res["ate"]["rmse"], "ate_n": res["ate"]["n"],
        "tracking": res["poses_estimated"] / max(1, n),
        "coverage": res["coverage_trusted"],
        "decisions": res["trace_summary"]["num_decisions"],
        "rolled_back": res["trace_summary"]["by_verdict"].get("rolled_back", 0),
        "recover": recover, "wall_seconds": res["wall_seconds"],
        "initialised": res["initialised"],
    }


def _job(spec) -> tuple:
    data, seq, fault_kind, severity, frames, cond, out_root = spec
    tag = f"{seq}__{fault_kind}__{cond}"
    try:
        src = load_source(data)
        src.frames = src.frames[:frames]
        if src.gt_centers is not None:
            src.gt_centers = src.gt_centers[:frames]
        n = len(src.frames)
        plan = None
        if fault_kind != "clean":
            plan = FaultPlan([FaultSegment(fault_kind, int(n * 0.35), int(n * 0.65),
                                           severity=severity)])
        out = os.path.join(out_root, tag)
        res = run_pipeline(src, out, mode=cond, fault=plan,
                           sup_params=SupervisorParams(), save_images=False)
        return _record(seq, fault_kind, cond, res, out), None
    except BaseException as e:  # a cell that crashes must not sink the table
        try:
            os.makedirs(os.path.join(out_root, tag), exist_ok=True)
            with open(os.path.join(out_root, tag, "ERROR.txt"), "w", encoding="utf-8") as f:
                traceback.print_exc(file=f)
        except Exception:
            pass
        return None, f"{tag} FAILED ({type(e).__name__}): {e}"


def _load_record(seq: str, fault: str, cond: str, out_dir: str) -> Optional[dict]:
    fp = os.path.join(out_dir, "result.json")
    if not os.path.isfile(fp):
        return None
    try:
        with open(fp, encoding="utf-8") as f:
            return _record(seq, fault, cond, json.load(f), out_dir)
    except Exception:
        return None


def _agg(runs: List[dict], keys) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for r in runs:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    rows = []
    for key, rs in sorted(groups.items()):
        ates = [r["ate"] for r in rs if r["ate"] == r["ate"]]
        rec = [r["recover"] for r in rs if r["recover"] is not None]
        rows.append({
            **dict(zip(keys, key)),
            "ate_mean": round(statistics.mean(ates), 4) if ates else None,
            "ate_std": round(statistics.pstdev(ates), 4) if len(ates) > 1 else None,
            "tracking_mean": round(statistics.mean(r["tracking"] for r in rs), 3),
            "coverage_mean": round(statistics.mean(r["coverage"] for r in rs), 3),
            "recover_mean": round(statistics.mean(rec), 1) if rec else None,
            "not_recovered": sum(1 for r in rs if r["recover"] is None and r["fault"] != "clean"),
            "decisions_mean": round(statistics.mean(r["decisions"] for r in rs), 1),
            "rolled_back_mean": round(statistics.mean(r["rolled_back"] for r in rs), 1),
        })
    return rows


def _relative(cell: dict, base: Optional[dict]) -> Optional[float]:
    if not cell or not base or not cell.get("ate_mean") or not base.get("ate_mean"):
        return None
    return round((cell["ate_mean"] - base["ate_mean"]) / base["ate_mean"] * 100.0, 1)


def to_md(table: List[dict], cross: List[dict], frames: int, severity: float) -> str:
    lines = [
        "# AEGIS-SLAM - real-footage benchmark (TUM RGB-D)",
        "",
        f"_Real handheld RGB used monocularly, first {frames} frames per sequence, "
        f"distortion removed at preparation. Faults injected on frames "
        f"35-65% at severity {severity:g}. ATE = Umeyama Sim(3) RMSE on camera "
        f"centres, the same metric as the synthetic grid. One run per cell._",
        "",
        "| Sequence | Fault | Condition | ATE (m) | vs clean | Tracking | Coverage | Recover(frames) | Decisions | Rollbacks |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        row = {**r,
               "vs_clean": ("-" if r["fault"] == "clean" or r["vs_clean"] is None
                            else f"{r['vs_clean']:+.1f}%"),
               "recover_mean": "-" if r["recover_mean"] is None else r["recover_mean"],
               "ate_mean": "-" if r["ate_mean"] is None else r["ate_mean"]}
        lines.append("| {sequence} | {fault} | {condition} | {ate_mean} | {vs_clean} | "
                     "{tracking_mean} | {coverage_mean} | {recover_mean} | {decisions_mean} | "
                     "{rolled_back_mean} |".format(**row))
    lines += [
        "",
        "## Across sequences",
        "",
        "_Mean over the sequences and the paired per-sequence change against the "
        "classical baseline for the same sequence and fault (negative = lower ATE)._",
        "",
        "| Fault | Condition | Mean ATE (m) | Paired ATE vs baseline (%) | Tracking | Decisions | Rollbacks |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in cross:
        lines.append("| {fault} | {condition} | {ate_mean} | {delta_vs_baseline} | "
                     "{tracking_mean} | {decisions_mean} | {rolled_back_mean} |".format(**r))
    return "\n".join(lines) + "\n"


def _plot(table: List[dict], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        seqs = sorted({r["sequence"] for r in table})
        faults = [f for f in FAULTS if any(r["fault"] == f for r in table)]
        conds = sorted({r["condition"] for r in table})
        fig, ax = plt.subplots(figsize=(1.9 * max(3, len(faults) * len(seqs)), 4))
        width = 0.8 / max(1, len(conds))
        labels, x = [], 0
        for seq in seqs:
            for fault in faults:
                base_x = x
                for ci, cond in enumerate(conds):
                    cell = next((r for r in table if r["sequence"] == seq and r["fault"] == fault
                                 and r["condition"] == cond), None)
                    if cell and cell["ate_mean"] is not None:
                        ax.bar(base_x + ci * width, cell["ate_mean"], width,
                               label=cond if x == 0 else None)
                labels.append(f"{seq}\n{fault}")
                x += 1
        ax.set_xticks(np.arange(len(labels)) + width * (len(conds) - 1) / 2)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Sim(3) ATE RMSE (m)")
        ax.set_title("TUM RGB-D, clean vs injected faults (lower is better)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
    except Exception as e:  # a missing figure must not fail the benchmark
        print("[real] figure skipped:", e)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser("AEGIS real-footage benchmark")
    ap.add_argument("--data", action="append", required=True,
                    help="prepared dataset dirs (repeat); label = basename")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--conditions", default="baseline_classical,supervised")
    ap.add_argument("--faults", default=",".join(FAULTS))
    ap.add_argument("--severity", type=float, default=0.6,
                    help="fault strength as a fraction of the synthetic nominal "
                         "(real frames are 640x480, so 1.0 is harsh)")
    ap.add_argument("--out", default="eval/real")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--resume", action="store_true",
                    help="keep cells that already have runs/<tag>/result.json and only run the rest")
    args = ap.parse_args(argv)

    conditions = [c for c in args.conditions.split(",") if c]
    faults = [f for f in args.faults.split(",") if f]
    data = list(dict.fromkeys(args.data))
    seqs = [(os.path.basename(os.path.normpath(d)), d) for d in data]

    os.makedirs(args.out, exist_ok=True)
    run_root = os.path.join(args.out, "runs")
    specs = [(d, seq, fault, args.severity, args.frames, cond, run_root)
             for seq, d in seqs for fault in faults for cond in conditions]

    runs = []
    if args.resume:
        for seq, _ in seqs:
            for fault in faults:
                for cond in conditions:
                    tag = f"{seq}__{fault}__{cond}"
                    rec = _load_record(seq, fault, cond, os.path.join(run_root, tag))
                    if rec is not None:
                        runs.append(rec)
        done = {f"{r['sequence']}__{r['fault']}__{r['condition']}" for r in runs}
        specs = [s for s in specs if f"{s[1]}__{s[2]}__{s[5]}" not in done]
        print(f"[real] resume: {len(runs)} cell(s) reused, {len(specs)} to run")

    if args.jobs > 1 and len(specs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            outcomes = list(pool.map(_job, specs))
    else:
        outcomes = [_job(s) for s in specs]

    for r, err in outcomes:
        if r is None:
            print("[real]", err)
        else:
            runs.append(r)
            print(f"[real] {r['sequence']}/{r['fault']}/{r['condition']}: "
                  f"ATE={r['ate']:.3f} track={r['tracking']:.2f} "
                  f"cov={r['coverage']:.2f} dec={r['decisions']} rec={r['recover']}")

    table = _agg(runs, keys=("sequence", "fault", "condition"))
    by_key = {(r["sequence"], r["fault"], r["condition"]): r for r in table}
    for r in table:
        clean = by_key.get((r["sequence"], r["fault"], "baseline_classical"))
        own_clean = by_key.get((r["sequence"], "clean", r["condition"]))
        r["vs_clean"] = (_relative(r, own_clean) if r["fault"] != "clean" else 0.0)
        r["paired_vs_baseline"] = _relative(r, clean)

    cross = []
    for fault in faults:
        for cond in conditions:
            cells = [by_key[(seq, fault, cond)] for seq, _ in seqs
                     if (seq, fault, cond) in by_key]
            if not cells:
                continue
            deltas = [c["paired_vs_baseline"] for c in cells
                      if c.get("paired_vs_baseline") is not None]
            cross.append({
                "fault": fault, "condition": cond,
                "ate_mean": round(statistics.mean(c["ate_mean"] for c in cells
                                                  if c["ate_mean"] is not None), 4)
                if any(c["ate_mean"] is not None for c in cells) else None,
                "delta_vs_baseline": (f"{statistics.mean(deltas):+.1f}%"
                                      if deltas else "n/a"),
                "tracking_mean": round(statistics.mean(c["tracking_mean"] for c in cells), 3),
                "decisions_mean": round(statistics.mean(c["decisions_mean"] for c in cells), 1),
                "rolled_back_mean": round(statistics.mean(c["rolled_back_mean"] for c in cells), 1),
            })

    payload = {
        "dataset": "TUM RGB-D", "frames": args.frames, "severity": args.severity,
        "conditions": conditions, "faults": faults, "sequences": [s for s, _ in seqs],
        "table": table, "across_sequences": cross, "runs": runs,
    }
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as f:
        f.write(to_md(table, cross, args.frames, args.severity))
    _plot(table, os.path.join(args.out, "real_bench.png"))

    print("\nAcross sequences (paired vs classical baseline):")
    for r in cross:
        print(f"  {r['fault']:<13} {r['condition']:<19} ATE={r['ate_mean']} "
              f"delta={r['delta_vs_baseline']} track={r['tracking_mean']} "
              f"dec={r['decisions_mean']} rb={r['rolled_back_mean']}")
    print("Wrote", os.path.join(args.out, "results.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
