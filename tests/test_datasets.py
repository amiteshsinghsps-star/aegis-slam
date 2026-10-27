"""TUM RGB-D adapter: timestamp association, intrinsics, portable layout."""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from aegis import datasets
from aegis.pipeline import load_source


def _make_tum(tmp_path, stamps=None, n=5, calib=True, name="rgbd_dataset_freiburg1_test",
              gt_stamps=None):
    seq = tmp_path / name
    rgb = seq / "rgb"
    rgb.mkdir(parents=True)
    rng = np.random.default_rng(0)
    stamps = stamps if stamps is not None else [1000.0 + i / 30.0 for i in range(n)]
    for t in stamps:
        img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(rgb / f"{t:.6f}.png"), img)
    with open(seq / "rgb.txt", "w", encoding="utf-8") as f:
        f.write("# timestamp filename\n")
        for t in stamps:
            f.write(f"{t:.6f} rgb/{t:.6f}.png\n")
    if gt_stamps is None:
        gt_stamps = [1000.0 + k / 100.0 for k in range(400)]
    with open(seq / "groundtruth.txt", "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for k, tt in enumerate(gt_stamps):
            f.write(f"{tt:.6f} {k * 0.01:.6f} 0.0 1.0 0 0 0 1\n")
    if calib:
        (seq / "calib.txt").write_text(
            "517.306504 516.469215 318.643040 255.313989 "
            "0.262383 -0.953104 -0.005358 0.002628 1.163314\n", encoding="utf-8")
    return seq


def test_nearest_indices_respects_tolerance():
    idx = datasets.nearest_indices(np.array([1.0, 1.03, 1.5]),
                                   np.array([1.0, 1.04, 1.6]), tol=0.02)
    assert idx.tolist() == [0, 1, -1]


def test_parse_groundtruth_yields_camera_centres(tmp_path):
    seq = _make_tum(tmp_path)
    ts, centers = datasets.parse_groundtruth(str(seq / "groundtruth.txt"))
    assert ts.shape == (400,) and centers.shape == (400, 3)
    assert centers[10].tolist() == [0.1, 0.0, 1.0]


def test_prepare_tum_writes_portable_dataset(tmp_path):
    seq = _make_tum(tmp_path, n=6)
    out = tmp_path / "prepared"
    info = datasets.prepare_tum(str(seq), str(out))
    assert info["frames"] == 6 and info["dropped_no_gt"] == 0
    names = sorted(os.listdir(out / "rgb"))
    assert names[0] == "frame_00000.png" and len(names) == 6
    gt = [line.split() for line in (out / "groundtruth.txt").read_text().splitlines()]
    assert len(gt) == 6 and all(len(row) == 4 for row in gt)
    ic = json.loads((out / "intrinsics.json").read_text())
    assert ic["adapter"] == "tum" and ic["undistorted"] is True
    assert ic["K"][0][0] == 517.306504 and 29.0 < ic["fps"] < 31.0

    src = load_source(str(out))
    assert len(src.frames) == src.gt_centers.shape[0] == 6
    assert src.meta.get("fps") and abs(src.meta["fps"] - 30.0) < 1.0


def test_prepare_tum_drops_frames_without_groundtruth(tmp_path):
    seq = _make_tum(tmp_path, stamps=[1000.0, 1000.25, 1000.5],
                    gt_stamps=[1000.0, 1000.01, 1000.02, 1000.5, 1000.51])
    info = datasets.prepare_tum(str(seq), str(tmp_path / "out"), tol=0.02)
    assert info["frames"] == 2 and info["dropped_no_gt"] == 1


def test_prepare_tum_max_frames_and_stride(tmp_path):
    seq = _make_tum(tmp_path, n=9)
    info = datasets.prepare_tum(str(seq), str(tmp_path / "a"), max_frames=4)
    assert info["frames"] == 4
    info = datasets.prepare_tum(str(seq), str(tmp_path / "b"), stride=3)
    assert info["frames"] == 3


def test_intrinsics_fall_back_to_official_table(tmp_path):
    seq = _make_tum(tmp_path, calib=False)
    K, dist, source = datasets.tum_intrinsics(str(seq))
    assert K[0][0] == 517.306504 and K[1][2] == 255.313989
    assert dist.shape == (5,) and source.startswith("official fr1")


def test_undistortion_can_be_disabled(tmp_path):
    seq = _make_tum(tmp_path, n=3)
    out = tmp_path / "raw"
    info = datasets.prepare_tum(str(seq), str(out), undistort=False)
    assert info["undistorted"] is False
    ic = json.loads((out / "intrinsics.json").read_text())
    assert ic["undistorted"] is False


def test_cli_prepares_and_reports(tmp_path, capsys):
    seq = _make_tum(tmp_path, n=5)
    out = tmp_path / "cli"
    rc = datasets.main(["--src", str(seq), "--out", str(out), "--max-frames", "3"])
    assert rc == 0
    assert "3 frames" in capsys.readouterr().out
    assert len(os.listdir(out / "rgb")) == 3
