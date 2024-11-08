"""A physics-based digital twin of a linear camera (EMVA 1288 sensor model).

Signal chain simulated for every pixel::

    photons --QE x PRNU--> photo-electrons --Poisson--> (+ dark current, DSNU)
            --clip at full well--> + read noise --x gain x K--> + black level --ADC--> DN

Because every parameter is known, the twin is the ground truth that the analysis code in
:mod:`camera_noise_characterization.noise` is validated against before it ever touches
real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SensorModel:
    """Linear sensor model. Defaults are fitted to the Baumer TXG08 lab data (8-bit output)."""

    width: int = 1032
    height: int = 776
    quantum_efficiency: float = 0.45
    conversion_gain: float = 0.022
    """System gain K at analogue gain 1, in DN per electron."""
    read_noise_e: float = 29.0
    """Temporal noise before the analogue amplifier, electrons rms."""
    dark_current_e_s: float = 5.0
    full_well_e: float = 13_000.0
    black_level_dn: float = 1.0
    bit_depth: int = 8
    prnu: float = 0.005
    """Photo-response non-uniformity, relative rms (fixed pattern)."""
    dsnu_e: float = 3.0
    """Dark-signal non-uniformity, electrons rms (fixed pattern)."""
    seed: int | None = 0
    _prnu_map: np.ndarray = field(init=False, repr=False, compare=False)
    _dsnu_map: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._prnu_map = 1.0 + self.prnu * rng.standard_normal(self.shape)
        self._dsnu_map = self.dsnu_e * rng.standard_normal(self.shape)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def max_dn(self) -> int:
        return 2**self.bit_depth - 1

    def mean_dn(self, photons: float, exposure_us: float, gain: float = 1.0) -> float:
        """Noise-free expected output for a uniform photon count (before clipping)."""
        electrons = self.quantum_efficiency * photons + self.dark_current_e_s * exposure_us * 1e-6
        return gain * self.conversion_gain * electrons + self.black_level_dn

    def expose(
        self,
        photons: float | np.ndarray,
        exposure_us: float,
        gain: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate one frame for a mean photon count per pixel (scalar or ``HxW`` array)."""
        rng = np.random.default_rng() if rng is None else rng
        photons = np.broadcast_to(np.asarray(photons, dtype=float), self.shape)
        mean_e = self.quantum_efficiency * photons * self._prnu_map
        mean_e = mean_e + self.dark_current_e_s * exposure_us * 1e-6
        electrons = rng.poisson(np.clip(mean_e, 0.0, None)).astype(float) + self._dsnu_map
        electrons = np.minimum(electrons, self.full_well_e)
        electrons += self.read_noise_e * rng.standard_normal(self.shape)
        dn = gain * self.conversion_gain * electrons + self.black_level_dn
        dn = np.clip(np.rint(dn), 0, self.max_dn)
        return dn.astype(np.uint8 if self.bit_depth <= 8 else np.uint16)


class SimulatedCamera:
    """Drop-in stand-in for a GigE camera: set ``exposure_us`` and ``gain``, then ``grab()``.

    ``photon_rate`` is the photon flux per pixel and microsecond for a white target;
    ``scene`` is an optional ``HxW`` reflectance map in ``[0, 1]`` (default: flat field).
    """

    def __init__(
        self,
        sensor: SensorModel | None = None,
        photon_rate: float = 3.0,
        scene: np.ndarray | None = None,
        seed: int | None = 0,
    ) -> None:
        self.sensor = sensor if sensor is not None else SensorModel(seed=seed)
        if scene is not None and scene.shape != self.sensor.shape:
            raise ValueError(f"scene shape {scene.shape} != sensor shape {self.sensor.shape}")
        self.photon_rate = photon_rate
        self.scene = scene
        self.exposure_us = 1000.0
        self.gain = 1.0
        self._rng = np.random.default_rng(seed)

    def grab(self) -> np.ndarray:
        photons = self.photon_rate * self.exposure_us
        if self.scene is not None:
            photons = photons * self.scene
        return self.sensor.expose(photons, self.exposure_us, self.gain, self._rng)
