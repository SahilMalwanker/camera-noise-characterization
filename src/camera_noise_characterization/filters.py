"""Denoising filters and a benchmark that holds measurements against closed-form theory."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb

import numpy as np
from scipy import ndimage

from camera_noise_characterization.noise import pair_stats


def binomial_kernel(size: int) -> np.ndarray:
    """Normalised 2-D binomial kernel - the integer-friendly approximation of a Gaussian."""
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    row = np.array([comb(size - 1, k) for k in range(size)], dtype=float)
    row /= row.sum()
    return np.outer(row, row)


def box_kernel(size: int) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    return np.full((size, size), 1.0 / size**2)


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Linear filtering in float with mirrored borders."""
    return ndimage.convolve(np.asarray(image, dtype=float), kernel, mode="reflect")


def median(image: np.ndarray, size: int = 3) -> np.ndarray:
    return ndimage.median_filter(np.asarray(image, dtype=float), size=size, mode="reflect")


def white_noise_gain(kernel: np.ndarray) -> float:
    """Factor by which a linear filter scales the std of white noise: ``sqrt(sum w^2)``."""
    return float(np.sqrt(np.sum(np.asarray(kernel, dtype=float) ** 2)))


@lru_cache(maxsize=8)
def median_noise_gain(n_samples: int, trials: int = 400_000, seed: int = 7) -> float:
    """Std of the median of ``n`` unit Gaussians (Monte Carlo; about sqrt(pi / 2n))."""
    rng = np.random.default_rng(seed)
    return float(np.median(rng.standard_normal((trials, n_samples)), axis=1).std())


@dataclass(frozen=True)
class DenoiseResult:
    name: str
    sigma_before: float
    sigma_after: float
    predicted_gain: float

    @property
    def measured_gain(self) -> float:
        return self.sigma_after / self.sigma_before


def denoise_benchmark(
    a: np.ndarray,
    b: np.ndarray,
    binomial_sizes: tuple[int, ...] = (3, 5, 7),
    median_size: int = 3,
) -> list[DenoiseResult]:
    """Filter both frames of a flat-field pair and re-measure the temporal noise.

    Linear filters are compared with ``white_noise_gain``; the median with its Gaussian
    efficiency. Deviations reveal spatially correlated noise or quantisation effects.
    """
    before = pair_stats(a, b).sigma
    results = [DenoiseResult("unfiltered", before, before, 1.0)]
    for size in binomial_sizes:
        k = binomial_kernel(size)
        after = pair_stats(convolve(a, k), convolve(b, k)).sigma
        results.append(DenoiseResult(f"binomial {size}x{size}", before, after, white_noise_gain(k)))
    after = pair_stats(median(a, median_size), median(b, median_size)).sigma
    results.append(
        DenoiseResult(
            f"median {median_size}x{median_size}",
            before,
            after,
            median_noise_gain(median_size**2),
        )
    )
    return results


def add_impulse_noise(
    image: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
    low: float = 0.0,
    high: float = 255.0,
) -> np.ndarray:
    """Salt-and-pepper noise: a ``fraction`` of pixels is forced to ``low`` or ``high``."""
    out = np.array(image, dtype=float, copy=True)
    hit = rng.random(out.shape) < fraction
    out[hit] = np.where(rng.random(int(hit.sum())) < 0.5, low, high)
    return out
