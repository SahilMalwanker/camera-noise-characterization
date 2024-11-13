"""Temporal-noise statistics and photon-transfer analysis (EMVA 1288 style).

The photon-transfer method rests on one fact: photo-electrons obey Poisson statistics, so
their variance equals their mean. Seen through a camera with system gain ``K`` (DN/e-)::

    var_DN = K * mean_DN + var_0            (var_0 = read noise^2 + quantisation 1/12)

A straight line through (mean, variance) pairs therefore yields the conversion gain from
two flat-field frames per point - no photometer, no datasheet needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

QUANTISATION_VAR = 1.0 / 12.0
"""Variance of the uniform rounding error of an ideal ADC, in DN^2."""


@dataclass(frozen=True)
class PairStats:
    """Statistics of two frames captured with identical settings."""

    mean: float
    """Mean signal of both frames in DN (black level removed if one was given)."""
    sigma_diff: float
    """Standard deviation of the difference image ``A - B`` in DN - the classic lab "noise"."""

    @property
    def sigma(self) -> float:
        """Temporal noise of one frame. Differencing adds two noise draws, hence the sqrt(2)."""
        return self.sigma_diff / math.sqrt(2.0)

    @property
    def variance(self) -> float:
        return self.sigma**2

    @property
    def relative_noise(self) -> float:
        """Difference noise over mean signal, as plotted in the original lab protocol."""
        return self.sigma_diff / self.mean

    @property
    def snr(self) -> float:
        return self.mean / self.sigma


def pair_stats(
    a: np.ndarray,
    b: np.ndarray,
    black_level: float = 0.0,
    roi: tuple[slice, slice] | None = None,
) -> PairStats:
    """Mean and temporal noise of a frame pair.

    Subtracting the frames cancels everything that is identical in both - scene texture,
    illumination fall-off and fixed-pattern noise - leaving only temporal noise.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"frame shapes differ: {a.shape} vs {b.shape}")
    if roi is not None:
        a, b = a[roi], b[roi]
    mean = 0.5 * (a.mean() + b.mean()) - black_level
    return PairStats(float(mean), float(np.std(a - b, ddof=1)))


def _ols(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares returning coefficients, their standard errors and R^2."""
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = max(len(y) - design.shape[1], 1)
    cov = (resid @ resid / dof) * np.linalg.pinv(design.T @ design)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 1.0
    return coef, np.sqrt(np.clip(np.diag(cov), 0.0, None)), r2


@dataclass(frozen=True)
class PTCFit:
    """Photon-transfer line ``var = K * mean + var_0``."""

    conversion_gain: float
    """System gain K in DN per electron."""
    conversion_gain_se: float
    noise_floor_var: float
    """Intercept var_0 in DN^2: read noise and quantisation as seen at the output."""
    r_squared: float
    n_points: int

    @property
    def electrons_per_dn(self) -> float:
        return 1.0 / self.conversion_gain

    @property
    def read_noise_dn(self) -> float:
        return math.sqrt(max(self.noise_floor_var - QUANTISATION_VAR, 0.0))

    @property
    def read_noise_e(self) -> float:
        return self.read_noise_dn / self.conversion_gain

    def saturation_e(self, max_dn: float = 255.0) -> float:
        """Electrons at full scale - the usable well depth if the ADC clips first."""
        return max_dn / self.conversion_gain

    def dynamic_range_db(self, max_dn: float = 255.0) -> float:
        floor_dn = math.sqrt(max(self.noise_floor_var, QUANTISATION_VAR))
        return 20.0 * math.log10(max_dn / floor_dn)


def fit_photon_transfer(
    means: np.ndarray, variances: np.ndarray, max_mean: float | None = None
) -> PTCFit:
    """Fit the photon-transfer line to per-frame variances (use ``PairStats.variance``).

    Points above ``max_mean`` are ignored: near saturation clipping squeezes the variance.
    """
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    keep = np.ones_like(means, dtype=bool) if max_mean is None else means <= max_mean
    if keep.sum() < 2:
        raise ValueError("need at least two unsaturated points")
    design = np.column_stack([means[keep], np.ones(keep.sum())])
    (k, var0), (k_se, _), r2 = _ols(design, variances[keep])
    return PTCFit(float(k), float(k_se), float(var0), r2, int(keep.sum()))


@dataclass(frozen=True)
class GainSweepFit:
    """Noise model of a gain sweep: ``var = K1 g mean + (K1 sigma_r)^2 g^2 + 1/12``."""

    conversion_gain: float
    """K1, the system gain at analogue gain 1, DN per electron."""
    conversion_gain_se: float
    read_noise_e: float
    """Noise added before the amplifier (electrons rms), so it is amplified with the signal."""
    r_squared: float
    n_points: int

    @property
    def read_noise_dn(self) -> float:
        """Read noise in DN at gain 1."""
        return self.read_noise_e * self.conversion_gain

    def predict_sigma(self, gains: np.ndarray, means: np.ndarray) -> np.ndarray:
        """Per-frame temporal noise predicted for the given gains and mean signals."""
        g = np.asarray(gains, dtype=float)
        mu = np.asarray(means, dtype=float)
        var = (
            self.conversion_gain * g * mu
            + (self.conversion_gain * self.read_noise_e * g) ** 2
            + QUANTISATION_VAR
        )
        return np.sqrt(var)


def fit_gain_sweep(gains: np.ndarray, means: np.ndarray, variances: np.ndarray) -> GainSweepFit:
    """Separate shot noise (grows like g) from pre-amplifier read noise (grows like g^2)."""
    g = np.asarray(gains, dtype=float)
    mu = np.asarray(means, dtype=float)
    v = np.asarray(variances, dtype=float) - QUANTISATION_VAR
    if len(g) < 3:
        raise ValueError("need at least three gain settings")
    (k1, b), (k1_se, _), r2 = _ols(np.column_stack([g * mu, g**2]), v)
    return GainSweepFit(float(k1), float(k1_se), math.sqrt(max(b, 0.0)) / k1, r2, len(g))


def shot_noise_limited_snr(electrons: np.ndarray, read_noise_e: float = 0.0) -> np.ndarray:
    """Ideal SNR ``mu / sqrt(mu + sigma_r^2)`` for a photo-electron count ``mu``."""
    mu = np.asarray(electrons, dtype=float)
    return mu / np.sqrt(mu + read_noise_e**2)
