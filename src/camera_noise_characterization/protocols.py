"""Measurement protocols that drive any camera exposing ``exposure_us``, ``gain`` and ``grab()``.

The same code runs against :class:`camera_noise_characterization.sensor.SimulatedCamera` in
tests and against real hardware through a three-member adapter - see the README for an example.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from camera_noise_characterization.noise import PairStats, pair_stats


class Camera(Protocol):
    exposure_us: float
    gain: float

    def grab(self) -> np.ndarray: ...


@dataclass(frozen=True)
class SweepPoint:
    exposure_us: float
    gain: float
    stats: PairStats


def measure_pair(camera: Camera, black_level: float = 0.0) -> PairStats:
    """Grab two frames back to back and return their mean and temporal noise."""
    return pair_stats(camera.grab(), camera.grab(), black_level)


def auto_expose(
    camera: Camera,
    target_mean: float,
    *,
    tolerance: float = 1.0,
    black_level: float = 0.0,
    min_us: float = 4.0,
    max_us: float = 60e6,
    max_iter: int = 40,
) -> float:
    """Find the exposure time that yields ``target_mean`` DN and leave the camera set to it.

    Proportional steps converge in two or three frames while the sensor is linear; a
    geometric bisection takes over whenever clipping makes the proportional guess unsafe.
    """
    lo, hi = min_us, max_us
    t = float(np.clip(camera.exposure_us, lo, hi))
    for _ in range(max_iter):
        camera.exposure_us = t
        mean = float(camera.grab().mean())
        if abs(mean - target_mean) <= tolerance:
            return t
        if mean < target_mean:
            lo = t
        else:
            hi = t
        signal = mean - black_level
        guess = t * (target_mean - black_level) / signal if signal > 0.5 else 4.0 * t
        if not lo < guess < hi:
            guess = float(np.sqrt(lo * hi))
        t = guess
    raise RuntimeError(f"no exposure reached {target_mean} +- {tolerance} DN")


def exposure_sweep(
    camera: Camera,
    t_start_us: float,
    t_stop_us: float,
    steps: int = 20,
    gain: float = 1.0,
    black_level: float = 0.0,
) -> list[SweepPoint]:
    """Mean and noise at ``steps`` evenly spaced exposure times (fixed gain)."""
    camera.gain = gain
    points = []
    for t in np.linspace(t_start_us, t_stop_us, steps):
        camera.exposure_us = float(t)
        points.append(SweepPoint(float(t), gain, measure_pair(camera, black_level)))
    return points


def gain_sweep(
    camera: Camera,
    gains: Iterable[float],
    target_mean: float = 128.0,
    black_level: float = 0.0,
    tolerance: float = 1.0,
) -> list[SweepPoint]:
    """Re-expose to the same mean signal at every gain, then measure the noise."""
    points = []
    for g in gains:
        camera.gain = float(g)
        t = auto_expose(camera, target_mean, tolerance=tolerance, black_level=black_level)
        points.append(SweepPoint(t, float(g), measure_pair(camera, black_level)))
    return points


def photon_transfer_sweep(
    camera: Camera,
    n_points: int = 24,
    gain: float = 1.0,
    overshoot: float = 1.15,
    black_level: float = 0.0,
    full_scale: float = 255.0,
) -> list[SweepPoint]:
    """EMVA-1288-style series from darkness to just beyond saturation."""
    camera.gain = gain
    t_sat = auto_expose(camera, 0.9 * full_scale, tolerance=2.0, black_level=black_level) / 0.9
    times = np.linspace(t_sat / n_points, overshoot * t_sat, n_points)
    return exposure_sweep(camera, times[0], times[-1], n_points, gain, black_level)
