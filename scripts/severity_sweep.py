"""U4 — severity sweep: does the supervisor's benefit grow with the fault?

One table row per (scenario, severity): baseline vs supervised ATE on the same
scenes, paired per seed, so the delta is not a between-scene comparison. A
severity level multiplies the nominal fault in `aegis/faults.py` (blur kernel
length, darkness gamma + noise, texture radius, blackout duration), so the x
axis is "how broken is the input", not "how many frames are broken".

    python scripts/severity_sweep.py --frames 90 --runs 2 --jobs 4

Writes eval/severity/results.json, eval/severity/results.md and
eval/severity/severity_sweep.png.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aegis.evaluate import ScenarioResult, _eval_job, ensure_seed_dataset
from aegis.faults import standard_scenarios


def _cell(rows: List[ScenarioResult]) -> dict:
    ates = [r.ate for r in rows if r.ate == r.ate]
    rec = [r.recover for r in rows if r.recover is not None]
    return {
        "n": len(rows),
        "ate_mean": round(statistics.mean(ates), 4) if ates else None,
        "ate_std": round(statistics.pstdev(ates), 4) if len(ates) > 1 else 0.0,
        "coverage_mean": round(statistics.mean(r.coverage for r in rows), 3),
        "recover_mean": round(statistics.mean(rec), 1) if rec else None,
        "not_recovered": sum(1 for r in rows if r.recover is None),
        "decisions_mean": round(statistics.mean(r.decisions for r in rows), 1),
        "rolled_back_mean": round(statistics.mean(r.rolled_back for r in rows), 1),
        "ates": [round(a, 4) for a in ates],
    }


def run(severities: List[float], kinds: List[str], conditions: List[str],
        data: str, frames: int, runs: int, out: str, jobs: int) -> dict:
    seed_data = [data if s == 0 else ensure_seed_dataset(data, s, frames)
                 for s in range(runs)]
    work: List[tuple] = []
    keys: List[Tuple[float, str, str, int]] = []
    for sev in severities:
        plans = [p for p in standard_scenarios(frames, severity=sev)
                 if p.segments[0].kind in kinds]
        for plan in plans:
            kind = plan.segments[0].kind
            for r_i in range(runs):
                for cond in conditions:
                    tag = f"s{sev:g}__{kind}__{cond}__r{r_i}"
                    work.append((seed_data[r_i], frames, cond, plan, out, tag))
                    keys.append((sev, kind, cond, r_i))

    print(f"[sweep] {len(work)} runs over {len(severities)} severity levels "
          f"x {len(kinds)} scenarios x {len(conditions)} conditions x {runs} scenes")
    got: Dict[Tuple[float, str, str], Dict[int, ScenarioResult]] = {}
    if jobs > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outcomes = list(pool.map(_eval_job, work))
    else:
        outcomes = [_eval_job(w) for w in work]
    for (res, note), key in zip(outcomes, keys):
        if res is None:
            print(f"[sweep] FAILED {note}")
            continue
        got.setdefault(key[:3], {})[key[3]] = res
        print(f"[sweep] sev={key[0]:g} {key[1]}__{key[2]} r{key[3]}: "
              f"ATE={res.ate:.3f} cov={res.coverage:.2f} recover={res.recover} "
              f"dec={res.decisions}")

    table = []
    for sev in severities:
        for kind in kinds:
            row = {"severity": sev, "scenario": kind}
            for cond in conditions:
                by_seed = got.get((sev, kind, cond), {})
                row[cond] = _cell(list(by_seed.values())) if by_seed else None
            base, sup = row.get("baseline_classical"), row.get("supervised")
            if base and sup:
                b, s = got.get((sev, kind, "baseline_classical"), {}), \
                       got.get((sev, kind, "supervised"), {})
                # ΔATE% = (supervised − baseline) / baseline, so + = worse
                deltas = [(s[i].ate - b[i].ate) / b[i].ate * 100.0
                          for i in sorted(set(b) & set(s)) if b[i].ate]
                row["ate_change_pct"] = round(statistics.mean(deltas), 1) if deltas else None
                row["pairs"] = len(deltas)
                row["pairs_won"] = sum(1 for d in deltas if d < 0)
            table.append(row)
    return {"frames": frames, "runs": runs, "conditions": conditions,
            "severities": severities, "scenarios": kinds, "table": table}


def to_md(payload: dict, conditions: List[str]) -> str:
    base = conditions[0] if conditions else "baseline_classical"
    lines = ["# AEGIS-SLAM — severity sweep (U4)",
             "",
             f"_{payload['runs']} distinct scenes per cell, {payload['frames']} frames/clip. "
             "`ΔATE%` is (supervised − baseline) / baseline averaged over **paired seeds** "
             "(same scene, both conditions); **positive = worse** (the supervisor's pose "
             "error increased). `Won` counts pairs where supervised ATE is lower._",
             "",
             "| Severity | Scenario | Baseline ATE (m) | Supervised ATE (m) | ΔATE% | Won | "
             "Baseline cov | Supervised cov | Baseline recover | Supervised recover | Decisions |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in payload["table"]:
        b, s = r.get(base), r.get("supervised")
        fmt = lambda v, n=4: ("—" if v is None else f"{v:.{n}f}")
        lines.append(
            f"| {r['severity']:g} | {r['scenario']} | "
            f"{fmt(b['ate_mean'] if b else None)} ± {fmt(b['ate_std'] if b else None)} | "
            f"{fmt(s['ate_mean'] if s else None)} ± {fmt(s['ate_std'] if s else None)} | "
            f"{r.get('ate_change_pct', '—')} | {r.get('pairs_won', '—')} | "
            f"{fmt(b['coverage_mean'] if b else None, 3)} | "
            f"{fmt(s['coverage_mean'] if s else None, 3)} | "
            f"{b['recover_mean'] if b else '—'} | {s['recover_mean'] if s else '—'} | "
            f"{s['decisions_mean'] if s else '—'} |")
    return "\n".join(lines) + "\n"


def plot(payload: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds, sevs = payload["scenarios"], payload["severities"]
    fig, axes = plt.subplots(1, len(kinds) + 1, figsize=(4.4 * (len(kinds) + 1), 3.6))
    axes = list(axes)
    for ax, kind in zip(axes[:-1], kinds):
        for cond, style in (("baseline_classical", "o--"), ("supervised", "s-")):
            xs, ys, err = [], [], []
            for r in payload["table"]:
                if r["scenario"] == kind and r.get(cond):
                    xs.append(r["severity"])
                    ys.append(r[cond]["ate_mean"])
                    err.append(r[cond]["ate_std"])
            ax.errorbar(xs, ys, yerr=err, marker=style[0], ls=style[1], capsize=3, label=cond)
        ax.set_title(f"{kind}: ATE vs fault severity", fontsize=10)
        ax.set_xlabel("severity (1.0 = nominal)")
        ax.set_ylabel("ATE RMSE (m, lower is better)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    ax = axes[-1]
    for kind, color in zip(kinds, ("tab:blue", "tab:orange")):
        xs = [r["severity"] for r in payload["table"] if r["scenario"] == kind]
        ys = [r.get("ate_change_pct") for r in payload["table"] if r["scenario"] == kind]
        ax.plot(xs, ys, "o-", color=color, label=f"{kind} ΔATE%")
    ax.axhline(0, color="k", lw=.8)
    ax.set_title("paired change (+ = worse)", fontsize=10)
    ax.set_xlabel("severity (1.0 = nominal)")
    ax.set_ylabel("ΔATE% vs baseline (lower is better)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("AEGIS-SLAM severity sweep")
    ap.add_argument("--data", default="Dataset/synthetic")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--runs", type=int, default=2, help="distinct scenes per cell")
    ap.add_argument("--severities", default="0.35,0.7,1.0")
    ap.add_argument("--scenarios", default="motion_blur,low_light",
                    help="graded scenarios only; frame_drop is a blackout and "
                         "texture_loss already has its own nominal level")
    ap.add_argument("--conditions", default="baseline_classical,supervised")
    ap.add_argument("--out", default="eval/severity")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sevs = [float(x) for x in args.severities.split(",") if x.strip()]
    kinds = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    conds = [x.strip() for x in args.conditions.split(",") if x.strip()]

    payload = run(sevs, kinds, conds, args.data, args.frames, args.runs, args.out, args.jobs)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    md = to_md(payload, conds)
    with open(os.path.join(args.out, "results.md"), "w", encoding="utf-8") as f:
        f.write(md)
    png = os.path.join(args.out, "severity_sweep.png")
    try:
        plot(payload, png)
    except Exception as e:                              # figure is nice-to-have
        png = None
        print(f"[sweep] figure skipped: {e}")
    for r in payload["table"]:
        b, s = (r.get("baseline_classical") or {}), (r.get("supervised") or {})
        print(f"sev={r['severity']:g} {r['scenario']}: baseline {b.get('ate_mean')} "
              f"-> supervised {s.get('ate_mean')}  ΔATE%={r.get('ate_change_pct')} "
              f"won {r.get('pairs_won')}/{r.get('pairs')}")
    print("wrote", os.path.join(args.out, "results.json"),
          os.path.join(args.out, "results.md"), png or "(no figure)")


if __name__ == "__main__":
    main()
