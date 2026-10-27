"""Regression: the agent may swap the descriptor family mid-run.

ORB descriptors are 32-byte binary rows; SIFT/ALIKED are float rows. A map
built under one family cannot be matched, PnP-tracked or extended under
another (LightGlue even asserts on the descriptor width). The pipeline must
therefore stash the old-family map, re-bootstrap under the new family, and
reinstate the stashed map when the backend switches back (the shape of a
verification rollback).
"""
import json

import aegis.pipeline as P
from aegis.pipeline import load_source, run_pipeline
from aegis.synth import make_dataset
from aegis.tools import invoke


class _SwitchStub:
    """Stands in for Supervisor; swaps ORB->SIFT then back to ORB."""

    def __init__(self, cfg, params, **_kw):
        self.cfg = cfg
        self.stage = 0

    def step(self, window, trace):
        frame = window[-1].frame
        if self.stage == 0 and frame >= 6:
            invoke("set_feature_backend", self.cfg, {"backend": "sift"}, frame)
            self.stage = 1
        elif self.stage == 1 and frame >= 16:
            invoke("set_feature_backend", self.cfg, {"backend": "orb"}, frame)
            self.stage = 2


def test_backend_family_switch_round_trip(tmp_path, monkeypatch):
    d = tmp_path / "frames"
    make_dataset(str(d), n_frames=26, seed=4, write_mp4=False)
    monkeypatch.setattr(P, "Supervisor", _SwitchStub)

    res = run_pipeline(load_source(str(d)), str(tmp_path / "run"),
                       mode="ablation", save_images=False)

    assert res["final_config"]["backend"] == "orb"
    assert res["initialised"]                       # map reinstated after switch-back
    assert res["poses_estimated"] > 0
    frames = [t["frame"] for t in json.loads(
        (tmp_path / "run" / "health_timeline.json").read_text())]
    assert len(frames) >= 20                        # no crash through the round trip
