"""Evaluation harness (FR-9, section 11).

Runs the pipeline under fault-injected scenarios across conditions and produces
the comparison table the technical report cites: ATE, coverage, mean health in
the fault window, frames-to-recover, agent decisions, and false interventions
on clean video. Fixed seeds; mean ± spread over runs.

Usage:
    python -m aegis.evaluate --data Dataset/synthetic --frames 120 --runs 1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from aegis.pipeline import load_source, run_pipeline
from aegis.tools import RuntimeConfig
from aegis.supervisor import SupervisorParams
from aegis.faults import standard_scenarios, FaultPlan


CONDITIONS = ["baseline_classical", "supervised", "supervised_multi", "ablation", "baseline_learned"]


def _load_timeline(out_dir: str) -> Dict[int, dict]:
    fp = os.path.join(out_dir, "health_timeline.json")
    if not os.path.isfile(fp):
        return {}
    with open(fp) as f:
        rows = json.load(f)
    return {int(r["frame"]): r for r in rows}


def _mean_in_window(tl: Dict[int, dict], lo: int, hi: int, key: str) -> Optional[float]:
    vals = [tl[f].get(key, 0) for f in tl if lo <= f <= hi]
    return float(np.mean(vals)) if vals else None


def _recovery_frames(tl: Dict[int, dict], start: int, end: int, n_frames: int) -> Optional[int]:
    """Frames after fault start until tracking is stable (not lost) for 3 frames."""
    stable = 0
    for f in range(start, min(n_frames, start + 80)):
        row = tl.get(f)
        if row is None:
            continue
        if not row.get("tracking_lost", False) and row.get("pnp_inliers", 0) > 0:
            stable += 1
            if stable >= 3:
                return f - start
        else:
            stable = 0
    return None  # never recovered in the search horizon


@dataclass
class ScenarioResult:
    scenario: str
    condition: str
    seed: int
    ate: float                     # tracked-frame ATE (diagnostic, not the primary)
    ate_all: float                 # all frames from first pose, hold-last for gaps
    ate_trusted: float             # ATE restricted to trusted frames
    coverage: float                # trusted fraction
    coverage_estimated: float      # frames with a pose / all frames
    health_in_fault: Optional[float]
    recover: Optional[int]
    censored: bool                 # True when no recovery inside the observation horizon
    decisions: int
    backend_switches: int
    rolled_back: int
    recapture: int
    verdicts: Dict[str, int]
    wall_seconds: float
    fps: float
    poses_estimated: int
    n_frames: int


def run_once(data: str, frames: int, condition: str, fault: Optional[FaultPlan],
              out_root: str, tag: str, seed: int = 0) -> ScenarioResult:
    src = load_source(data)
    src.frames = src.frames[:frames]
    if src.gt_centers is not None:
        src.gt_centers = src.gt_centers[:frames]
    out_dir = os.path.join(out_root, tag)
    sup = SupervisorParams()
    res = run_pipeline(src, out_dir, mode=condition, fault=fault, sup_params=sup, save_images=False)
    tl = _load_timeline(out_dir)
    health_in_fault = recover = None
    censored = False
    if fault and fault.segments:
        lo, hi = fault.segments[0].start, fault.segments[0].end
        health_in_fault = _mean_in_window(tl, lo, hi, "pnp_inliers")  # track quality proxy
        recover = _recovery_frames(tl, lo, hi, len(src.frames))
        censored = recover is None
    ts = res["trace_summary"]
    wall = float(res.get("wall_seconds") or 0.0)
    return ScenarioResult(
        scenario=tag.split("__")[0], condition=condition, seed=seed,
        ate=res["ate"]["rmse"],
        ate_all=(res.get("ate_all") or {}).get("rmse", float("nan")),
        ate_trusted=(res.get("ate_trusted") or {}).get("rmse", float("nan")),
        coverage=res["coverage_trusted"],
        coverage_estimated=res.get("coverage_estimated", 0.0),
        health_in_fault=health_in_fault, recover=recover, censored=censored,
        decisions=ts["num_decisions"],
        backend_switches=ts["num_backend_switches"],
        rolled_back=ts["by_verdict"].get("rolled_back", 0),
        recapture=ts.get("by_diagnosis", {}).get("tracking_lost_unrecoverable", 0),
        verdicts=dict(ts.get("by_verdict", {})),
        wall_seconds=wall,
        fps=(len(src.frames) / wall) if wall > 0 else float("nan"),
        poses_estimated=res["poses_estimated"],
        n_frames=len(src.frames),
    )


def _finite_mean(xs: List[float]) -> Optional[float]:
    vals = [x for x in xs if x == x and x not in (float("inf"), float("-inf"))]
    return statistics.mean(vals) if vals else None


def _finite_std(xs: List[float]) -> Optional[float]:
    vals = [x for x in xs if x == x and x not in (float("inf"), float("-inf"))]
    return statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None)


def aggregate(runs: List[ScenarioResult]) -> dict:
    groups: Dict[tuple, List[ScenarioResult]] = {}
    for r in runs:
        groups.setdefault((r.scenario, r.condition), []).append(r)
    table = []
    for (sc, cond), rs in groups.items():
        recovs = [x.recover for x in rs if x.recover is not None]
        ate_all_mean = _finite_mean([x.ate_all for x in rs])
        ate_tr_mean = _finite_mean([x.ate_trusted for x in rs])
        table.append({
            "scenario": sc, "condition": cond, "n": len(rs),
            "ate_mean": _round(_finite_mean([x.ate for x in rs]), 4),
            "ate_std": _round(_finite_std([x.ate for x in rs]), 4),
            "ate_all_mean": _round(ate_all_mean, 4),
            "ate_all_std": _round(_finite_std([x.ate_all for x in rs]), 4),
            "ate_trusted_mean": _round(ate_tr_mean, 4),
            "ate_trusted_std": _round(_finite_std([x.ate_trusted for x in rs]), 4),
            "coverage_mean": _round(_finite_mean([x.coverage for x in rs]), 3),
            "coverage_estimated_mean": _round(_finite_mean([x.coverage_estimated for x in rs]), 3),
            "recover_mean": _round(statistics.mean(recovs), 1) if recovs else None,
            "recovered_n": len(recovs),
            "not_recovered": sum(1 for x in rs if x.recover is None),
            "censored": sum(1 for x in rs if x.censored),
            "decisions_mean": _round(_finite_mean([x.decisions for x in rs]), 1),
            "backend_switches_mean": _round(_finite_mean([x.backend_switches for x in rs]), 1),
            "rolled_back_mean": _round(_finite_mean([x.rolled_back for x in rs]), 1),
            "recapture_mean": _round(_finite_mean([x.recapture for x in rs]), 1),
            "wall_s_mean": _round(_finite_mean([x.wall_seconds for x in rs]), 2),
            "fps_mean": _round(_finite_mean([x.fps for x in rs]), 1),
            "verdicts": _sum_verdicts(rs),
        })
    return table


def _round(v: Optional[float], nd: int):
    return round(v, nd) if v is not None else None


def _sum_verdicts(rs: List[ScenarioResult]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rs:
        for k, v in (r.verdicts or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


def to_markdown(table: List[dict], n_frames: int, runs: int = 1, severity: float = 1.0,
                seeds: Optional[List[int]] = None) -> str:
    sev = ("" if abs(severity - 1.0) < 1e-9
           else f" Fault level scaled to **severity {severity:g}** "
                f"(1.0 = the nominal kernel/gamma/radius in `aegis/faults.py`).")
    seed_txt = (f"seeds {seeds[0]}..{seeds[-1]}" if seeds and len(seeds) > 1
                else f"seed {seeds[0]}" if seeds else f"seeds 0..{runs-1}")
    lines = ["# AEGIS-SLAM — Evaluation results",
             "",
             f"_Synthetic sequences, {n_frames} frames/clip, faults injected mid-sequence. "
             f"Each cell is the mean over **{runs} distinct scenes ({seed_txt})**. "
             f"`ATE all` is the primary metric: every frame from the first pose, with "
             f"missing poses filled by the last one. `ATE trusted` uses only frames the "
             f"agent did not distrust, so it must be read with coverage._{sev}",
             "",
             "| Scenario | Condition | ATE all (m) | ATE trusted (m) | Trusted cov | Estimated cov | "
             "Recover (frames) | Recovered n | Not recovered | Decisions | Switches | Rollbacks | Wall (s) | FPS |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        rec = ("—" if r.get("recover_mean") is None
               else f"{r['recover_mean']} (n={r.get('recovered_n', 0)})")
        lines.append(
            f"| {r['scenario']} | {r['condition']} | "
            f"{r['ate_all_mean']} ± {r['ate_all_std']} | "
            f"{r['ate_trusted_mean']} ± {r['ate_trusted_std']} | "
            f"{r['coverage_mean']} | {r['coverage_estimated_mean']} | "
            f"{rec} | {r.get('recovered_n', 0)} | {r['not_recovered']} | "
            f"{r['decisions_mean']} | {r['backend_switches_mean']} | "
            f"{r['rolled_back_mean']} | {r['wall_s_mean']} | {r['fps_mean']} |")
    lines += ["", "`Recover (frames)` is **conditional on recovery**: it averages only "
              "the scenes that recovered. `Not recovered` counts right-censored "
              "scenes that never stabilised inside the observation window — a "
              "small conditional mean next to a non-zero censored count is a "
              "failure, not a speed-up."]
    return "\n".join(lines) + "\n"


def clean_false_intervention_check(data: str, frames: int, out_root: str,
                                   runs: int, seeds: Optional[List[int]] = None,
                                   conditions=("supervised", "supervised_multi")) -> dict:
    seeds = seeds if seeds else list(range(runs))
    rows = []
    for i in seeds:
        d = ensure_seed_dataset(data, i, frames) if i != 0 else data
        src = load_source(d)
        src.frames = src.frames[:frames]
        if src.gt_centers is not None:
            src.gt_centers = src.gt_centers[:frames]
        for cond in conditions:
            out = os.path.join(out_root, f"clean__{cond}__s{i}")
            res = run_pipeline(src, out, mode=cond, fault=None, save_images=False)
            ts = res["trace_summary"]
            rows.append({
                "seed": i,
                "condition": cond,
                "decisions": ts["num_decisions"],
                "by_diagnosis": ts.get("by_diagnosis", {}),
                "by_verdict": ts.get("by_verdict", {}),
                "ate_all": (res.get("ate_all") or {}).get("rmse"),
                "final_config_is_baseline": (
                    res.get("final_config") == RuntimeConfig().snapshot()),
                "wall_seconds": res.get("wall_seconds"),
            })
    decisions = [r["decisions"] for r in rows]
    sup = [r["decisions"] for r in rows if r["condition"] == "supervised"]
    return {"rows": rows,
            "clean_decisions_per_run": sup,
            "mean": round(statistics.mean(decisions), 2) if decisions else None,
            "total_decisions": int(sum(decisions)),
            "seeds": seeds,
            "conditions": list(conditions)}


def _eval_job(job):
    """Picklable worker for serial and --jobs parallel paths."""
    if len(job) == 7:
        data, frames, cond, plan, out_root, tag, seed = job
    else:
        data, frames, cond, plan, out_root, tag = job
        seed = int(tag.rsplit("s", 1)[-1]) if "s" in tag else 0
    try:
        return run_once(data, frames, cond, plan, out_root, tag, seed=seed), tag
    except Exception as e:
        return None, f"{tag} FAILED: {e}"


def ensure_seed_dataset(base_data: str, seed: int, n_frames: int) -> str:
    """One distinct synthetic scene per seed, generated once and reused.

    ``--runs`` must vary the *scene* (seed), not just re-execute a
    deterministic job — otherwise 'mean ± std' would be fake precision.
    """
    from aegis.synth import make_dataset
    d = os.path.join(os.path.dirname(base_data.rstrip("/\\")) or ".",
                     f"{os.path.basename(base_data.rstrip('/\\'))}_seed{seed}")
    if seed == 0 and os.path.abspath(d) != os.path.abspath(base_data) \
            and not os.path.isdir(d):
        d = base_data  # seed 0 == the original built dataset
    if not os.path.isfile(os.path.join(d, "rgb", "image_00000.png")):
        os.makedirs(os.path.join(d, "rgb"), exist_ok=True)
        make_dataset(d, n_frames=n_frames, seed=seed, write_mp4=False)
    return d


def main():
    ap = argparse.ArgumentParser("AEGIS-SLAM evaluation")
    ap.add_argument("--data", default="Dataset/synthetic")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--runs", type=int, default=1,
                    help="fallback when --seeds is not given: use seeds 0..N-1")
    ap.add_argument("--seeds", default=None,
                    help="comma list of synthetic scene seeds, e.g. 0,1,2,3,4 "
                         "(dev) or 10,11,...,17 (locked test). Every condition "
                         "runs on the same scene for paired comparisons")
    ap.add_argument("--conditions", default="baseline_classical,supervised,ablation")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--scenarios", default="all",
                    help="comma list: motion_blur,low_light,texture_loss,frame_drop (or 'all')")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="scales each scenario's nominal fault level (0.05..1.0); "
                         "1.0 is the published grid")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run jobs in N parallel worker processes")
    ap.add_argument("--skip_learned", action="store_true")
    ap.add_argument("--skip_clean", action="store_true",
                    help="omit the clean-video false-intervention check (used by the severity sweep)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seeds = ([int(x) for x in args.seeds.split(",") if x.strip()]
             if args.seeds else list(range(args.runs)))
    if not seeds:
        raise SystemExit("no seeds selected")
    conditions = [c for c in args.conditions.split(",") if c]
    if args.skip_learned and "baseline_learned" in conditions:
        conditions.remove("baseline_learned")

    scenarios = standard_scenarios(args.frames, severity=args.severity)
    if args.scenarios.strip().lower() != "all":
        wanted = {s.strip() for s in args.scenarios.split(",")}
        scenarios = [p for p in scenarios if p.segments[0].kind in wanted]
    all_results: List[ScenarioResult] = []
    jobs = []
    # one distinct synthetic scene per seed (generated once up front so
    # parallel workers never race on dataset creation)
    seed_data = {s: (args.data if s == 0 else ensure_seed_dataset(args.data, s, args.frames))
                 for s in seeds}
    for si, plan in enumerate(scenarios):
        scen_name = plan.segments[0].kind
        for seed in seeds:
            for cond in conditions:
                tag = f"{scen_name}__{cond}__s{seed}"
                jobs.append((seed_data[seed], args.frames, cond, plan, args.out,
                             tag, seed))

    if args.jobs > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            outcomes = list(pool.map(_eval_job, jobs))
    else:
        outcomes = [_eval_job(j) for j in jobs]
    failures = [note for r, note in outcomes if r is None]
    for r, note in outcomes:
        if r is None:
            print(f"[eval] {note}")
        else:
            all_results.append(r)
            print(f"[eval] {r.scenario}__{r.condition}: ATE={r.ate:.3f} cov={r.coverage:.2f} "
                  f"recover={r.recover} dec={r.decisions}")

    table = aggregate(all_results)
    clean = ({"clean_decisions_per_run": [], "mean": None, "rows": []}
             if args.skip_clean else
             clean_false_intervention_check(args.data, args.frames, args.out,
                                            runs=len(seeds), seeds=seeds))

    payload = {"frames": args.frames, "runs": len(seeds), "seeds": seeds,
               "severity": args.severity,
               "conditions": conditions,
               "scenarios": [p.segments[0].kind for p in scenarios],
               "table": table,
               "rows": [r.__dict__ for r in all_results],
               "failures": failures,
               "false_intervention_clean": clean}
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(table, args.frames, runs=len(seeds),
                            severity=args.severity, seeds=seeds))
        if not args.skip_clean:
            f.write("\n## False interventions on clean video\n\n")
            f.write("| Seed | Condition | Decisions | Diagnoses | Verdicts | ATE all (m) | Final config = default |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for row in clean.get("rows", []):
                ate = row.get("ate_all")
                ate_txt = f"{ate:.4f}" if isinstance(ate, (int, float)) else "—"
                f.write(f"| {row['seed']} | {row.get('condition', 'supervised')} | "
                        f"{row['decisions']} | {row['by_diagnosis']} | "
                        f"{row['by_verdict']} | {ate_txt} | "
                        f"{row['final_config_is_baseline']} |\n")
            f.write(f"\n**{clean['total_decisions']} decisions across "
                    f"{len(clean.get('rows', []))} clean runs** "
                    f"(mean {clean['mean']} per 90-frame run).\n")
        if failures:
            f.write("\n## Failed runs\n\n")
            for note in failures:
                f.write(f"* {note}\n")
            f.write("\nFailed cells are excluded from the table above and counted here; "
                    "they are not silently replaced.\n")
    print("\nWrote", os.path.join(args.out, "results.md"))
    print(to_markdown(table, args.frames, severity=args.severity, seeds=seeds))
    print("Clean decisions:", clean.get("clean_decisions_per_run"))


if __name__ == "__main__":
    main()
