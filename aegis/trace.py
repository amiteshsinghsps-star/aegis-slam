"""Structured decision trace (FR-5).

Every agent decision is recorded as a JSON-serialisable record with the exact
shape from the technical report:

  {
    "decision_id": "d-0042",
    "frame": 1180,
    "evidence": {"inliers": 11, "inlier_ratio": 0.18, "blur_var": 22.4, ...},
    "diagnosis": "motion_blur",
    "decided_by": "rules" | "bedrock" | "fallback",
    "action": {"tool": "set_feature_backend", "args": {"backend": "aliked"}},
    "outcome": {"window": [1180, 1210], "inlier_ratio_after": 0.52, "verdict": "improved"},
    "config_before": {...}, "config_after": {...}
  }

The trace is the primary artifact judges read for the Agentic Vision Award: it
shows OpenCV-5 outputs (evidence) changing a later decision, and it records the
verify-and-rollback outcome. It is written incrementally so a live UI can tail
it while a job runs.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from typing import List, Optional


def json_safe(obj):
    """Replace non-finite floats with `null`, recursively.

    A run with no ground-truth trajectory reports ATE as NaN. Python's `json`
    writes that as the bare literal `NaN`, which `json.load` tolerates but every
    strict parser — including a browser's `JSON.parse` — rejects. Passing every
    payload through this keeps `result.json`, `trace.json` and the HTTP responses
    valid JSON, with "no ground truth" expressed as `null`.
    """
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    tolist = getattr(obj, "tolist", None)          # numpy scalar / array
    if tolist is not None:
        return json_safe(tolist())
    return obj


def _compact_evidence(sig: dict) -> dict:
    keys = ("inliers", "pnp_inliers", "inlier_ratio", "blur_var", "brightness",
            "contrast", "feature_count", "tracked_landmarks", "mean_reproj_px",
            "frame_to_frame_matches", "texture")
    return {k: _round(sig.get(k)) for k in keys if k in sig}


def _round(x):
    if isinstance(x, float):
        return round(x, 4)
    return x


class DecisionTrace:
    def __init__(self) -> None:
        self.records: List[dict] = []
        self._counter = 0
        self._lock = threading.Lock()

    def next_id(self) -> str:
        self._counter += 1
        return f"d-{self._counter:04d}"

    def add(self, *, frame: int, evidence: dict, diagnosis: str,
            decided_by: str, action: dict, config_before: dict,
            config_after: dict, outcome: Optional[dict] = None,
            extra: Optional[dict] = None) -> dict:
        with self._lock:
            rec = {
                "decision_id": self.next_id(),
                "frame": int(frame),
                "evidence": _compact_evidence(evidence),
                "diagnosis": diagnosis,
                "decided_by": decided_by,
                "action": action,
                "config_before": config_before,
                "config_after": config_after,
                "outcome": outcome,
            }
            if extra:
                rec.update(extra)
            self.records.append(rec)
            return rec

    def set_outcome(self, decision_id: str, outcome: dict) -> None:
        with self._lock:
            for r in self.records:
                if r["decision_id"] == decision_id:
                    r["outcome"] = outcome
                    return

    def summary(self) -> dict:
        by_diag, by_decider, verdicts = {}, {}, {}
        for r in self.records:
            by_diag[r["diagnosis"]] = by_diag.get(r["diagnosis"], 0) + 1
            by_decider[r["decided_by"]] = by_decider.get(r["decided_by"], 0) + 1
            v = (r.get("outcome") or {}).get("verdict", "pending")
            verdicts[v] = verdicts.get(v, 0) + 1
        return {
            "num_decisions": len(self.records),
            "by_diagnosis": by_diag,
            "by_decider": by_decider,
            "by_verdict": verdicts,
            "num_rolled_back": verdicts.get("rolled_back", 0),
            "num_kept": verdicts.get("improved", 0),
            "num_backend_switches": sum(
                1 for r in self.records
                if r["action"].get("tool") == "set_feature_backend"
            ),
        }

    def dump(self, path: str) -> None:
        payload = json_safe({
            "schema_version": 1,
            "summary": self.summary(),
            "decisions": self.records,
        })
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def to_json(self) -> str:
        return json.dumps(json_safe({
            "schema_version": 1,
            "summary": self.summary(),
            "decisions": self.records,
        }), indent=2)
