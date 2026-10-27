"""Annotated decision-trace walkthrough figure (PRD U5 / rule A2).

Renders one run as a single figure: per-frame health proxy + confidence over
time, fault window, diagnosis/action markers, and keep/rollback verdicts —
the visual proof that an OpenCV-5 observation changed a later decision.

    python -m scripts.trace_figure runs/demo --out docs/diagrams/trace_walkthrough.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rd = a.run_dir
    out = a.out or os.path.join(rd, "trace_walkthrough.png")
    tl = json.load(open(os.path.join(rd, "health_timeline.json")))
    tr = json.load(open(os.path.join(rd, "trace.json")))
    res = json.load(open(os.path.join(rd, "result.json")))
    decisions = tr.get("decisions", [])

    frames = [r["frame"] for r in tl]
    inlr = [(r.get("pnp_inliers", 0) / r["pnp_candidates"]) if r.get("pnp_candidates") else 0.0
            for r in tl]
    blur = [r.get("blur_var", 0.0) for r in tl]
    conf = [r.get("confidence", None) for r in tl]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(frames, inlr, color="#2f81f7", lw=1.6, label="PnP inlier ratio (OpenCV health signal)")
    ax.plot(frames, [min(b / 200.0, 1.0) for b in blur], color="#d29922", lw=1.1,
            ls="--", label="blur (Laplacian var /200)")
    if any(c is not None for c in conf):
        ax.plot(frames, [c if c is not None else float("nan") for c in conf],
                color="#3fb950", lw=1.1, label="per-frame confidence")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("frame")
    ax.set_ylabel("health / confidence  →")
    ate = (res.get("ate") or {}).get("rmse")
    ax.set_title(f"AEGIS-SLAM decision trace — {res['mode']} "
                 f"({'ok' if res['initialised'] else 'not init'}; "
                 f"{res['frames']} frames; ATE {ate:.3f} m; coverage "
                 f"{res['coverage_trusted']:.0%})" if ate else
                 f"AEGIS-SLAM decision trace — {res['mode']}")

    palette = {"kept": "#3fb950", "rolled_back": "#f85149", "pending": "#8b949e",
               "noop": "#8b949e"}
    ylev = {"set_feature_backend": 1.04, "apply_preprocessing": 0.93,
            "retune_thresholds": 0.82, "rebootstrap": 0.71,
            "mark_segment_untrusted": 0.60, "request_recapture": 0.49}
    for d in decisions:
        f = d["frame"]
        tool = d.get("action", {}).get("tool", "?")
        verdict = (d.get("outcome") or {}).get("verdict", "pending")
        y = ylev.get(tool, 1.08)
        c = palette.get(verdict, "#8b949e")
        ax.axvline(f, color=c, lw=0.8, ls=":", alpha=0.7)
        ax.annotate(f"{d['diagnosis']}\n→ {tool} [{verdict}]",
                    xy=(f, y), xytext=(f, y + 0.10),
                    ha="center", fontsize=7, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8))
    if not decisions:
        ax.text(0.5, 1.05, "no interventions (healthy scene)", transform=ax.transAxes,
                ha="center", fontsize=8, color="#8b949e")

    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  ({len(decisions)} decisions annotated)")


if __name__ == "__main__":
    main()
