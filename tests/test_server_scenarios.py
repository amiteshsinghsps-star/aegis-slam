"""Live-run scenario injection: the web/API path must be able to apply the
same fault kernels as the benchmark, otherwise every mode sees clean video."""
import pytest

from aegis.faults import FAULTS
from aegis.server import _SCENARIOS, _fault_plan, create_app


def test_scenarios_cover_every_fault_plus_clean():
    assert _SCENARIOS[0] == "clean"
    assert set(FAULTS).issubset(_SCENARIOS)


def test_fault_plan_uses_benchmark_window():
    plan = _fault_plan("motion_blur", 100)
    assert plan is not None and len(plan.segments) == 1
    s = plan.segments[0]
    assert (s.kind, s.start, s.end, s.severity) == ("motion_blur", 35, 65, 1.0)


def test_clean_and_empty_have_no_plan():
    assert _fault_plan("clean", 100) is None
    assert _fault_plan(None, 100) is None


def test_unknown_fault_rejected_before_any_run():
    from fastapi.testclient import TestClient
    r = TestClient(create_app()).post("/jobs", data={"mode": "supervised", "fault": "nope"})
    assert r.status_code == 400
    assert "unknown scenario" in r.json()["detail"]
