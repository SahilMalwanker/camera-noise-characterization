"""camera-noise-characterization - characterise a machine-vision camera like a metrologist."""

from camera_noise_characterization.noise import (
    GainSweepFit,
    PairStats,
    PTCFit,
    fit_gain_sweep,
    fit_photon_transfer,
    pair_stats,
)
from camera_noise_characterization.optics import BAUMER_TXG08, Lens, OpticalSetup, Sensor
from camera_noise_characterization.sensor import SensorModel, SimulatedCamera

__all__ = [
    "BAUMER_TXG08",
    "GainSweepFit",
    "Lens",
    "OpticalSetup",
    "PTCFit",
    "PairStats",
    "Sensor",
    "SensorModel",
    "SimulatedCamera",
    "fit_gain_sweep",
    "fit_photon_transfer",
    "pair_stats",
]

__version__ = "1.0.0"
