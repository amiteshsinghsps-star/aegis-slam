"""Human approval gate for re-capture requests (FR-6).

When the supervisor decides a segment is unrecoverable it can request a
re-capture. That request is surfaced to a human (in the web UI) with the
evidence frame and diagnosis. If it is approved, the run can flag the capture
for re-recording; if it is skipped, or if no human answers before the timeout,
the segment is marked untrusted so the report never presents it as reliable.

This class is backend-agnostic: in the FastAPI service it blocks on a real
event until the ``/decisions`` endpoint fires; in offline evaluation it uses a
zero timeout so the run is reproducible and non-interactive.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ApprovalRequest:
    decision_id: str
    frame: int
    reason: str
    evidence: dict
    timeout_s: float
    status: str = "pending"          # pending | approved | skipped | timeout
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    decided_by: Optional[str] = None      # audit: who/what resolved it
    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    def resolve(self, status: str, by: Optional[str] = None) -> None:
        if self.status == "pending":
            self.status = status
            self.resolved_at = time.time()
            self.decided_by = by or ("system:auto" if status != "timeout" else None)
            self._event.set()

    def wait(self) -> str:
        self._event.wait(timeout=self.timeout_s)
        if self.status == "pending":
            self.status = "timeout"
            self.resolved_at = time.time()
            self.decided_by = "system:timeout"
        return self.status

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "frame": self.frame,
            "reason": self.reason,
            "evidence": self.evidence,
            "status": self.status,
            "timeout_s": self.timeout_s,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "decided_by": self.decided_by,
        }


class ApprovalGate:
    def __init__(self, default_timeout_s: float = 25.0, auto: Optional[str] = None):
        # auto in {None, "approve", "skip"}; when set, requests resolve instantly
        # (offline evaluation uses auto="skip").
        self.default_timeout_s = default_timeout_s
        self.auto = auto
        self._requests: Dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()

    def request(self, frame: int, reason: str, evidence: dict,
                timeout_s: Optional[float] = None) -> ApprovalRequest:
        req = ApprovalRequest(
            decision_id="recapture-" + uuid.uuid4().hex[:8],
            frame=int(frame), reason=str(reason), evidence=dict(evidence),
            timeout_s=float(timeout_s or self.default_timeout_s),
        )
        with self._lock:
            self._requests[req.decision_id] = req
        if self.auto == "approve":
            req.resolve("approved", by=f"system:auto({self.auto})")
        elif self.auto == "skip":
            req.resolve("skipped", by=f"system:auto({self.auto})")
        return req

    def resolve(self, decision_id: str, approve: bool,
                by: Optional[str] = "web-session") -> Optional[ApprovalRequest]:
        with self._lock:
            req = self._requests.get(decision_id)
        if req is not None:
            req.resolve("approved" if approve else "skipped", by=by)
        return req

    def get(self, decision_id: str) -> Optional[ApprovalRequest]:
        return self._requests.get(decision_id)

    def pending(self) -> List[dict]:
        with self._lock:
            return [r.to_dict() for r in self._requests.values() if r.status == "pending"]

    def all(self) -> List[dict]:
        with self._lock:
            return [r.to_dict() for r in self._requests.values()]
