"""Trust / risk-coverage metrics (WS3).

For every evaluated run this script joins the saved per-frame confidence and
health evidence with the Sim(3)-aligned per-frame error, then reports:

  * AUROC of confidence, inlier ratio and landmark count for predicting a
    high-error frame (the label is error above the run's pooled 75th percentile
    and at least 5 cm, so the label is not just noise),
  * risk-coverage curves and AURC (lower is better) when frames are retained in
    descending score order,
  * trusted-ATE at matched coverage: the agent condition gated to its own
    trusted coverage versus the classical baseline gated to the *same*
    coverage (``baseline_gated``),
  * silent-failure rate: high-error frames that confidence still called safe
    (confidence >= 0.5).

Usage:
    python scripts/trust.py --eval eval --out eval/trust.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_run(run_dir: str) -> Optional[dict]:
    traj = os.path.join(run_dir, "trajectory.npz")
    health = os.path.join(run_dir, "health_timeline.json")
    result = os.path.join(run_dir, "result.json")
    if not all(os.path.isfile(p) for p in (traj, health, result)):
        return None
    z = np.load(traj)
    if "frame_err" not in z.files or "est_frames" not in z.files:
        return None
    err = np.asarray(z["frame_err"], dtype=np.float64)
    frames = np.asarray(z["est_frames"], dtype=np.int64)
    conf = np.asarray(z["conf"], dtype=np.float64)
    if err.size == 0 or err.size != frames.size:
        return None
    with open(health, encoding="utf-8") as f:
        rows = {int(r["frame"]): r for r in json.load(f)}
    with open(result, encoding="utf-8") as f:
        res = json.load(f)
    ratio, land, reproj = [], [], []
    for fr in frames:
        r = rows.get(int(fr), {})
        cand = float(r.get("pnp_candidates", 0) or 0)
        ratio.append(float(r.get("pnp_inliers", 0) or 0) / cand if cand else 0.0)
        land.append(float(r.get("tracked_landmarks", 0) or 0))
        reproj.append(float(r.get("mean_reproj_px", 0) or 0))
    if conf.size != frames.size:
        conf = np.asarray([rows.get(int(fr), {}).get("confidence", 0.0) for fr in frames],
                          dtype=np.float64)
    tag = os.path.basename(run_dir)
    parts = tag.split("__")
    seed = None
    if len(parts) >= 3 and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
    return {"frames": frames, "err": err, "conf": conf,
            "ratio": np.asarray(ratio), "land": np.asarray(land),
            "reproj": np.asarray(reproj),
            "coverage_trusted": float(res.get("coverage_trusted", 1.0)),
            "coverage_estimated": float(res.get("coverage_estimated", 1.0)),
            "mode": res.get("mode"), "scenario": parts[0],
            "condition": parts[1] if len(parts) > 1 else None,
            "seed": seed}


def _auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    pos, neg = scores[labels], scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return None
    # rank-based AUC with ties counted as 0.5
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        idx = np.where(allv == v)[0]
        if idx.size > 1:
            ranks[idx] = ranks[idx].mean()
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _risk_curve(err: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="mergesort")
    e = err[order]
    frac = np.arange(1, e.size + 1) / e.size
    risk = np.cumsum(e) / np.arange(1, e.size + 1)
    return frac, risk


def _aurc(frac: np.ndarray, risk: np.ndarray) -> float:
    return float(np.trapezoid(risk, frac) if hasattr(np, "trapezoid")
                 else np.trapz(risk, frac))


def _gated_mean(err: np.ndarray, score: np.ndarray, target_cov: float) -> Optional[float]:
    n = max(1, min(err.size, int(round(target_cov * err.size))))
    order = np.argsort(-score, kind="mergesort")
    return float(err[order[:n]].mean())


def _high_error_label(err: np.ndarray) -> np.ndarray:
    thresh = max(0.05, float(np.percentile(err, 75)))
    return err >= thresh


def _summarise(runs: List[dict]) -> dict:
    if not runs:
        return {"n_runs": 0}
    err = np.concatenate([r["err"] for r in runs])
    conf = np.concatenate([r["conf"] for r in runs])
    ratio = np.concatenate([r["ratio"] for r in runs])
    land = np.concatenate([r["land"] for r in runs])
    labels = np.concatenate([_high_error_label(r["err"]) for r in runs])
    frac_c, risk_c = _risk_curve(err, conf)
    frac_r, risk_r = _risk_curve(err, ratio)
    frac_l, risk_l = _risk_curve(err, land)
    high = labels
    silent = float(np.mean(conf[high] >= 0.5)) if high.any() else None
    return {
        "n_runs": len(runs), "n_frames": int(err.size),
        "auroc_confidence": _round(_auc(conf, labels)),
        "auroc_inlier_ratio": _round(_auc(ratio, labels)),
        "auroc_landmarks": _round(_auc(land, labels)),
        "aurc_confidence": _round(_aurc(frac_c, risk_c), 4),
        "aurc_inlier_ratio": _round(_aurc(frac_r, risk_r), 4),
        "aurc_landmarks": _round(_aurc(frac_l, risk_l), 4),
        "mean_error": _round(float(err.mean()), 4),
        "high_error_rate": _round(float(labels.mean()), 3),
        "silent_failure_rate": _round(silent, 3),
        "curve_coverage": [round(float(x), 3) for x in frac_c[::max(1, len(frac_c)//10)]],
        "curve_risk_confidence": [round(float(x), 4) for x in risk_c[::max(1, len(risk_c)//10)]],
    }


def _round(v, nd=3):
    return round(float(v), nd) if v is not None else None


def main() -> None:
    ap = argparse.ArgumentParser("AEGIS-SLAM trust metrics")
    ap.add_argument("--eval", default="eval")
    ap.add_argument("--out", default="eval/trust.json")
    ap.add_argument("--markdown", default="eval/trust.md")
    args = ap.parse_args()

    runs: List[dict] = []
    for name in sorted(os.listdir(args.eval)):
        d = os.path.join(args.eval, name)
        if not os.path.isdir(d):
            continue
        r = _load_run(d)
        if r is not None:
            r["tag"] = name
            runs.append(r)
    if not runs:
        raise SystemExit("no runs with frame_err found; rerun aegis.evaluate after the upgrade")

    by: Dict[Tuple[str, str], List[dict]] = {}
    for r in runs:
        by.setdefault((r["scenario"], r["mode"]), []).append(r)
    out = {"sign_convention": "lower AURC is better; AUROC > 0.5 means the score ranks high-error frames higher",
           "by_cell": {}, "overall": {}}
    for (sc, mode), rs in by.items():
        out["by_cell"].setdefault(sc, {})[mode] = _summarise(rs)
    for mode in sorted({r["mode"] for r in runs}):
        out["overall"][mode] = _summarise([r for r in runs if r["mode"] == mode])

    # matched-coverage comparison against the classical baseline, paired by seed
    matched = []
    for sc in sorted({r["scenario"] for r in runs}):
        for mode in sorted({r["mode"] for r in runs if r["mode"] != "baseline_classical"}):
            base = {(r["seed"]): r for r in runs
                    if r["scenario"] == sc and r["mode"] == "baseline_classical"}
            for r in runs:
                if r["scenario"] != sc or r["mode"] != mode:
                    continue
                b = base.get(r["seed"])
                if b is None:
                    continue
                target = r["coverage_trusted"]
                cg = _gated_mean(r["err"], r["conf"], target)
                bg = _gated_mean(b["err"], b["conf"], target)
                if cg is not None and bg is not None and bg > 0:
                    matched.append({"scenario": sc, "mode": mode, "seed": r["seed"],
                                    "coverage": round(target, 3),
                                    "agent_gated": round(cg, 4),
                                    "baseline_gated": round(bg, 4),
                                    "delta_pct": round((cg - bg) / bg * 100.0, 2)})
    if matched:
        ds = [m["delta_pct"] for m in matched]
        out["matched_coverage"] = {
            "n_pairs": len(ds),
            "mean_delta_pct": round(float(np.mean(ds)), 2),
            "agent_gated_mean": round(float(np.mean([m["agent_gated"] for m in matched])), 4),
            "baseline_gated_mean": round(float(np.mean([m["baseline_gated"] for m in matched])), 4),
            "pairs": matched,
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = ["# Trust / risk–coverage metrics",
             "",
             "_High-error label = framed error ≥ max(5 cm, run 75th percentile). "
             "AUROC > 0.5 means the score ranks high-error frames higher; lower AURC is better. "
             "Silent-failure rate = high-error frames whose confidence was still ≥ 0.5._",
             "",
             "| Cell | Condition | AUROC conf | AUROC inliers | AUROC landmarks | AURC conf | Silent failure |",
             "|---|---|---|---|---|---|---|"]
    for sc, modes in out["by_cell"].items():
        for mode, s in modes.items():
            if not s.get("n_runs"):
                continue
            lines.append(f"| {sc} | {mode} | {s['auroc_confidence']} | "
                         f"{s['auroc_inlier_ratio']} | {s['auroc_landmarks']} | "
                         f"{s['aurc_confidence']} | {s['silent_failure_rate']} |")
    if matched:
        m = out["matched_coverage"]
        lines += ["", "## Trusted ATE at matched coverage",
                  f"Agent-gated {m['agent_gated_mean']:.4f} m vs baseline-gated "
                  f"{m['baseline_gated_mean']:.4f} m across {m['n_pairs']} paired cells "
                  f"(Δ {m['mean_delta_pct']:+.2f} %, positive = worse)."]
    with open(args.markdown, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
