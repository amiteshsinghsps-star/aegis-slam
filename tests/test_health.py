import cv2
import numpy as np
import pytest

from aegis.health import (blur_score, brightness_score, contrast_score,
                               texture_energy, image_quality, make_signals,
                               frame_confidence, HealthSignals)


def _textured_frame(seed=0, size=(120, 160)):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=size + (3,), dtype=np.uint8)
    return img


def _blurred(img):
    k = np.zeros((15, 15), np.float32); k[7, :] = 1.0 / 15
    return cv2.filter2D(img, -1, k)


def test_blur_score_drops_under_motion_blur():
    img = _textured_frame()
    sharp = blur_score(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    blur = blur_score(cv2.cvtColor(_blurred(img), cv2.COLOR_BGR2GRAY))
    assert blur < 0.2 * sharp


def test_brightness_and_contrast_signals():
    gray = np.full((50, 50), 30, np.uint8)
    assert brightness_score(gray) == pytest.approx(30.0)
    assert contrast_score(gray) == pytest.approx(0.0)
    noisy = np.random.default_rng(1).integers(0, 255, (50, 50), dtype=np.uint8)
    assert contrast_score(noisy) > 50


def test_texture_energy_high_for_noise_flat_for_wall():
    noise = _textured_frame()
    wall = np.full((120, 160, 3), 128, np.uint8)
    assert texture_energy(cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)) > 10
    assert texture_energy(cv2.cvtColor(wall, cv2.COLOR_BGR2GRAY)) == pytest.approx(0.0, abs=1e-6)


def test_image_quality_reports_feature_count():
    orb = cv2.ORB_create(nfeatures=500)
    q = image_quality(_textured_frame(), orb)
    assert q["feature_count"] > 50
    assert set(q) == {"blur_var", "brightness", "contrast", "texture", "feature_count"}


def test_make_signals_geometry_and_inlier_ratio():
    img = _textured_frame()
    s = make_signals(7, img, geometry={"pnp_inliers": 40, "pnp_candidates": 100,
                                       "tracking_lost": False, "initialised": True})
    assert s.frame == 7
    assert s.inlier_ratio == pytest.approx(0.4)
    s0 = make_signals(0, img, geometry={"pnp_candidates": 0})
    assert s0.inlier_ratio == 0.0


def test_frame_confidence_bounds_and_loss():
    healthy = HealthSignals(frame=1, initialised=True, pnp_inliers=50,
                            pnp_candidates=50, tracked_landmarks=200)
    assert frame_confidence(healthy) == pytest.approx(1.0)
    lost = HealthSignals(frame=2, initialised=True, tracking_lost=True,
                         pnp_inliers=50, pnp_candidates=50)
    assert frame_confidence(lost) == 0.0
    weak = HealthSignals(frame=3, initialised=True, pnp_inliers=5,
                         pnp_candidates=50, tracked_landmarks=30)
    assert 0.0 < frame_confidence(weak) < 0.5
