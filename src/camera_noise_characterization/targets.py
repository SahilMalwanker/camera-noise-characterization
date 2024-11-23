"""Synthetic test targets with exact ground truth."""

from __future__ import annotations

import numpy as np


def siemens_star(
    size: int = 256,
    spokes: int = 16,
    supersample: int = 4,
    low: float = 0.15,
    high: float = 0.85,
    blind_radius: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anti-aliased Siemens star filling the frame.

    Returns ``(image, edges, care)``: the image in ``[low, high]``, the ground-truth edge
    pixels (every pixel straddling a spoke boundary) and a mask excluding the centre, where
    spokes converge below the sampling limit.
    """
    n = size * supersample
    c = (np.arange(n) + 0.5) / supersample - size / 2.0
    xx, yy = np.meshgrid(c, c)
    sector = np.floor((np.arctan2(yy, xx) + np.pi) / (np.pi / spokes)).astype(int) % 2
    fine = sector.reshape(size, supersample, size, supersample)
    coverage = fine.mean(axis=(1, 3))
    image = low + (high - low) * coverage
    edges = (coverage > 0.0) & (coverage < 1.0)
    r = np.hypot(
        *np.meshgrid(np.arange(size) + 0.5 - size / 2.0, np.arange(size) + 0.5 - size / 2.0)
    )
    care = r > blind_radius * size / 2.0
    return image, edges, care
