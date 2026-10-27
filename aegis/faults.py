"""Fault injection (FR-9 / section 11).

Controlled degradations applied with OpenCV-5 to a clean sequence so we can
measure whether the supervisor recovers tracking where a fixed pipeline does
not. Each fault is an image transform over a frame range.

  * motion_blur   — linear (box along a direction) kernel, three severities
  * low_light     — gamma darkening + additive sensor noise
  * texture_loss  — strong blur inside a moving region (kills local texture)
  * frame_drop    — corrupted/black burst frames (simulates dropped frames)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np


def _motion_blur_kernel(length: int, angle_deg: float = 45.0) -> np.ndarray:
    length = max(3, int(length) | 1)  # odd
    k = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    a = np.radians(angle_deg)
    dx, dy = np.cos(a), np.sin(a)
    for i in range(length):
        x = int(round(center + (i - center) * dx))
        y = int(round(center + (i - center) * dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s > 0 else np.eye(length, dtype=np.float32) / length


def motion_blur(img: np.ndarray, severity: float = 1.0) -> np.ndarray:
    length = int(np.clip(5 + 20 * severity, 5, 61))
    k = _motion_blur_kernel(length, angle_deg=35.0)
    return cv2.filter2D(img, -1, k)


def low_light(img: np.ndarray, severity: float = 1.0) -> np.ndarray:
    gamma = 1.0 + 2.0 * severity
    inv = np.clip((img / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    noise = np.random.default_rng(int(severity * 1000) & 0xFFFF).normal(0, 4 + 8 * severity, img.shape[:2])
    noisy = np.clip(inv.astype(np.float32) + noise[:, :, None], 0, 255).astype(np.uint8)
    return noisy


def texture_loss(img: np.ndarray, severity: float = 1.0, frame: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    cx = int((0.5 + 0.25 * np.sin(frame / 10.0)) * w)
    cy = int((0.5 + 0.2 * np.cos(frame / 12.0)) * h)
    rx = int(w * (0.15 + 0.25 * severity))
    ry = int(h * (0.15 + 0.25 * severity))
    out = img.copy()
    x0, x1 = max(0, cx - rx), min(w, cx + rx)
    y0, y1 = max(0, cy - ry), min(h, cy + ry)
    region = out[y0:y1, x0:x1]
    if region.size:
        k = int(np.clip(9 + 25 * severity, 5, 61)) | 1
        out[y0:y1, x0:x1] = cv2.GaussianBlur(region, (k, k), 0)
    return out


def frame_drop(img: np.ndarray, severity: float = 1.0) -> np.ndarray:
    # A dropped frame is a hard blackout: no features survive, tracking collapses.
    # Deliberately level-independent — there is no "mildly missing" frame, so the
    # severity knob grades the *duration* of the burst (see standard_scenarios).
    return np.full_like(img, 14)


FAULTS = {
    "motion_blur": motion_blur,
    "low_light": low_light,
    "texture_loss": texture_loss,
    "frame_drop": frame_drop,
}


@dataclass
class FaultSegment:
    kind: str
    start: int
    end: int
    severity: float = 1.0

    def active(self, frame: int) -> bool:
        return self.start <= frame <= self.end


@dataclass
class FaultPlan:
    segments: List[FaultSegment] = field(default_factory=list)

    def apply(self, img: np.ndarray, frame: int) -> np.ndarray:
        for s in self.segments:
            if s.active(frame):
                if s.kind == "texture_loss":
                    img = FAULTS[s.kind](img, s.severity, frame=frame)
                else:
                    img = FAULTS[s.kind](img, s.severity)
        return img

    def windows(self) -> List[Tuple[int, int, str]]:
        return [(s.start, s.end, f"{s.kind}@{s.severity}") for s in self.segments]


def standard_scenarios(n_frames: int, severity: float = 1.0) -> List[FaultPlan]:
    """Four representative mid-sequence faults sized to the clip length.

    `severity` scales each scenario's nominal level (clamped to 0.05..1.0).
    Blur kernel length, light gamma + noise and the texture-removal radius all
    grow with it; a dropped frame has no intensity to scale (it is a blackout),
    so for `frame_drop` the *duration* scales instead — 8 frames at severity 1.0.
    """
    s = float(np.clip(severity, 0.05, 1.0))
    lo, hi = int(n_frames * 0.35), int(n_frames * 0.65)
    drop_len = max(2, int(round(8 * s)))
    return [
        FaultPlan([FaultSegment("motion_blur", lo, hi, severity=s)]),
        FaultPlan([FaultSegment("low_light", lo, hi, severity=s)]),
        FaultPlan([FaultSegment("texture_loss", lo, hi, severity=0.8 * s)]),
        FaultPlan([FaultSegment("frame_drop", lo, lo + drop_len, severity=s)]),
    ]
