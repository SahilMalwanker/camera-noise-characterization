import numpy as np
import pytest

from camera_noise_characterization.protocols import auto_expose, exposure_sweep, gain_sweep
from camera_noise_characterization.sensor import SensorModel, SimulatedCamera


def camera(seed: int = 0) -> SimulatedCamera:
    return SimulatedCamera(SensorModel(width=160, height=120, seed=seed), seed=seed)


def test_auto_expose_hits_target_mean():
    cam = camera(1)
    t = auto_expose(cam, 128.0, tolerance=1.0)
    assert cam.exposure_us == t
    assert cam.grab().mean() == pytest.approx(128.0, abs=1.5)


def test_auto_expose_recovers_from_saturation():
    cam = camera(2)
    cam.exposure_us = 5e6
    auto_expose(cam, 60.0, tolerance=1.0)
    assert cam.grab().mean() == pytest.approx(60.0, abs=1.5)


def test_relative_noise_falls_like_shot_noise():
    cam = camera(3)
    points = exposure_sweep(cam, 1000.0, 8000.0, steps=8)
    rel = np.array([p.stats.relative_noise for p in points])
    assert rel[0] > 2 * rel[-1]
    mean = np.array([p.stats.mean for p in points])
    sigma = np.array([p.stats.sigma for p in points])
    slope = np.polyfit(np.log(mean), np.log(sigma), 1)[0]
    assert 0.35 < slope < 0.55


def test_noise_rises_with_gain_at_constant_signal():
    cam = camera(4)
    points = gain_sweep(cam, [1, 2, 4, 8], target_mean=128.0)
    sigmas = [p.stats.sigma for p in points]
    assert sigmas == sorted(sigmas)
    assert all(p.stats.mean == pytest.approx(127.0, abs=2.5) for p in points)
