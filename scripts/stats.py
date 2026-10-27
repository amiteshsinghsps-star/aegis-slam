"""Paired statistics for the AEGIS-SLAM benchmark (WS1).

Reads ``eval/results.json`` (written by ``python -m aegis.evaluate --seeds ...``)
and computes, for every condition against ``baseline_classical`` on the same
(scenario, seed):

  * mean paired ΔATE% with a 10,000-resample bootstrap 95% CI,
  * a two-sided Wilcoxon signed-rank p-value (exact sign-flip when SciPy is
    absent and n is small; normal approximation otherwise),
  * wins/ties/losses, where a win means the condition's ATE is lower.

Sign convention, used everywhere in the docs: **ΔATE% = (condition − baseline)
/ baseline × 100, so positive = worse.**

Usage:
    python scripts/stats.py --results eval/results.json --out eval/stats.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from typing import Dict, List, Optional, Tuple


def _wilcoxon(xs: List[float]) -> Tuple[Optional[float], str]:
    """Two-sided Wilcoxon signed-rank test on paired differences."""
    nz = [x for x in xs if abs(x) > 1e-12]
    n = len(nz)
    if n == 0:
        return 1.0, "no-nonzero"
    try:
        from scipy.stats import wilcoxon  # type: ignore
        return float(wilcoxon(nz, zero_method="wilcox", alternative="two-sided").pvalue), "scipy"
    except Exception:
        pass
    # exact sign-flip permutation on the mean (2^n) when feasible
    if n <= 18:
        obs = abs(statistics.mean(nz))
        rng = random.Random(12345)
        hits = 0
        total = 1 << n
        for mask in range(total):
            s = sum((-v if (mask >> i) & 1 else v) for i, v in enumerate(nz))
            if abs(s / n) >= obs - 1e-15:
                hits += 1
        return hits / total, f"signflip-exact n={n}"
    # normal approximation on the signed-rank sum
    ranks = sorted(range(n), key=lambda i: abs(nz[i]))
    rank = {i: r + 1 for r, i in enumerate(ranks)}
    w_plus = sum(rank[i] for i in range(n) if nz[i] > 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mu) / sigma if sigma else 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return min(1.0, p), "normal-approx"


def _bootstrap_ci(xs: List[float], n_boot: int = 10000,
                  alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    if len(xs) < 2:
        return None, None
    rng = random.Random(20261026)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def _pairs(rows: List[dict], scenario: str, cond: str) -> List[dict]:
    base = {(r["seed"]): r for r in rows
            if r["scenario"] == scenario and r["condition"] == "baseline_classical"}
    out = []
    for r in rows:
        if r["scenario"] != scenario or r["condition"] != cond:
            continue
        b = base.get(r["seed"])
        if b is None:
            continue
        ba, ca = b.get("ate_all"), r.get("ate_all")
        if not isinstance(ba, (int, float)) or not isinstance(ca, (int, float)):
            continue
        if not (ba == ba and ca == ca) or ba <= 0:
            continue
        out.append({"seed": r["seed"], "base": ba, "cond": ca,
                    "delta_pct": (ca - ba) / ba * 100.0,
                    "delta_trusted_pct": (
                        (r["ate_trusted"] - b["ate_trusted"]) / b["ate_trusted"] * 100.0
                        if isinstance(b.get("ate_trusted"), (int, float))
                        and isinstance(r.get("ate_trusted"), (int, float))
                        and b["ate_trusted"] > 0 else None),
                    "coverage": r.get("coverage"),
                    "base_coverage": b.get("coverage")})
    return out


def _summarise(pairs: List[dict]) -> dict:
    if not pairs:
        return {"n": 0}
    ds = [p["delta_pct"] for p in pairs]
    mean = statistics.mean(ds)
    lo, hi = _bootstrap_ci(ds)
    pval, method = _wilcoxon(ds)
    wins = sum(1 for d in ds if d < -1e-9)
    losses = sum(1 for d in ds if d > 1e-9)
    ties = len(ds) - wins - losses
    return {
        "n": len(ds),
        "mean_delta_pct": round(mean, 2),
        "median_delta_pct": round(statistics.median(ds), 2),
        "ci95_low": round(lo, 2) if lo is not None else None,
        "ci95_high": round(hi, 2) if hi is not None else None,
        "wilcoxon_p": round(pval, 5) if pval is not None else None,
        "wilcoxon_method": method,
        "wins": wins, "ties": ties, "losses": losses,
        "mean_ate_base": round(statistics.mean(p["base"] for p in pairs), 4),
        "mean_ate_cond": round(statistics.mean(p["cond"] for p in pairs), 4),
        "mean_coverage_cond": round(statistics.mean(
            p["coverage"] for p in pairs if isinstance(p["coverage"], (int, float))), 3)
        if any(isinstance(p["coverage"], (int, float)) for p in pairs) else None,
        "pairs": [{"seed": p["seed"], "base": round(p["base"], 4),
                   "cond": round(p["cond"], 4),
                   "delta_pct": round(p["delta_pct"], 2)} for p in pairs],
    }


def main() -> None:
    ap = argparse.ArgumentParser("AEGIS-SLAM paired statistics")
    ap.add_argument("--results", default="eval/results.json")
    ap.add_argument("--out", default="eval/stats.json")
    ap.add_argument("--markdown", default="eval/stats.md")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("rows") or []
    if not rows:
        raise SystemExit("results.json has no per-run rows; rerun aegis.evaluate with --seeds")
    scenarios = payload.get("scenarios") or sorted({r["scenario"] for r in rows})
    conditions = [c for c in payload.get("conditions", [])
                  if c != "baseline_classical"] or sorted(
                      {r["condition"] for r in rows if r["condition"] != "baseline_classical"})

    out: Dict[str, dict] = {"sign_convention": "positive ΔATE% = worse",
                            "source": args.results,
                            "seeds": payload.get("seeds"),
                            "by_condition": {}, "by_scenario": {}}
    for cond in conditions:
        all_pairs = []
        for sc in scenarios:
            pairs = _pairs(rows, sc, cond)
            if not pairs:
                continue
            out["by_scenario"].setdefault(sc, {})[cond] = _summarise(pairs)
            all_pairs.extend(pairs)
        out["by_condition"][cond] = _summarise(all_pairs)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = ["# Paired statistics (ΔATE% = condition − baseline; positive = worse)",
             "",
             f"_Seeds: {payload.get('seeds')}. Paired on the same synthetic scene. "
             "CI = 10,000-resample bootstrap; p = two-sided Wilcoxon signed-rank._",
             "",
             "| Condition | n | ΔATE% mean [95% CI] | Median | W / T / L | Wilcoxon p | ATE base → cond (m) |",
             "|---|---|---|---|---|---|---|"]
    for cond, s in out["by_condition"].items():
        if not s.get("n"):
            continue
        ci = (f"[{s['ci95_low']}, {s['ci95_high']}]"
              if s["ci95_low"] is not None else "—")
        lines.append(f"| {cond} | {s['n']} | {s['mean_delta_pct']:+.2f} {ci} | "
                     f"{s['median_delta_pct']:+.2f} | {s['wins']} / {s['ties']} / {s['losses']} | "
                     f"{s['wilcoxon_p']} | {s['mean_ate_base']:.4f} → {s['mean_ate_cond']:.4f} |")
    lines += ["", "## Per scenario", "",
              "| Scenario | Condition | n | ΔATE% mean [95% CI] | W / T / L | p |",
              "|---|---|---|---|---|---|"]
    for sc, conds in out["by_scenario"].items():
        for cond, s in conds.items():
            if not s.get("n"):
                continue
            ci = (f"[{s['ci95_low']}, {s['ci95_high']}]"
                  if s["ci95_low"] is not None else "—")
            lines.append(f"| {sc} | {cond} | {s['n']} | {s['mean_delta_pct']:+.2f} {ci} | "
                         f"{s['wins']} / {s['ties']} / {s['losses']} | {s['wilcoxon_p']} |")
    with open(args.markdown, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
