import csv
from pathlib import Path

import numpy as np
import pytest

from camera_noise_characterization.noise import fit_gain_sweep, fit_photon_transfer, pair_stats
from camera_noise_characterization.protocols import gain_sweep, photon_transfer_sweep
from camera_noise_characterization.sensor import SensorModel, SimulatedCamera

DATA = Path(__file__).resolve().parents[1] / "data"


def load_lab(name: str) -> dict[str, np.ndarray]:
    with (DATA / name).open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["include"] == "1"]
    return {
        key: np.array([float(r[key]) for r in rows]) for key in ("gain", "mean_dn", "sigma_diff_dn")
    }


def test_pair_stats_separates_difference_and_frame_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(100.0, 2.0, (400, 400))
    b = rng.normal(100.0, 2.0, (400, 400))
    s = pair_stats(a, b)
    assert s.mean == pytest.approx(100.0, abs=0.05)
    assert s.sigma == pytest.approx(2.0, rel=0.01)
    assert s.sigma_diff == pytest.approx(2.0 * np.sqrt(2.0), rel=0.01)
    assert s.relative_noise == pytest.approx(s.sigma_diff / s.mean)


def test_pair_stats_rejects_mismatched_frames():
    with pytest.raises(ValueError):
        pair_stats(np.zeros((2, 2)), np.zeros((3, 3)))


def test_photon_transfer_fit_recovers_twin_conversion_gain():
    sensor = SensorModel(width=256, height=192, seed=11)
    cam = SimulatedCamera(sensor, seed=12)
    points = photon_transfer_sweep(cam, n_points=16, black_level=sensor.black_level_dn)
    fit = fit_photon_transfer(
        [p.stats.mean for p in points], [p.stats.variance for p in points], max_mean=230
    )
    assert fit.conversion_gain == pytest.approx(sensor.conversion_gain, rel=0.03)
    assert fit.read_noise_e == pytest.approx(sensor.read_noise_e, rel=0.15)
    assert fit.r_squared > 0.99


def test_gain_sweep_fit_recovers_twin_parameters():
    sensor = SensorModel(width=256, height=192, seed=21)
    cam = SimulatedCamera(sensor, seed=22)
    points = gain_sweep(cam, range(1, 9), target_mean=128.0, black_level=sensor.black_level_dn)
    fit = fit_gain_sweep(
        [p.gain for p in points],
        [p.stats.mean for p in points],
        [p.stats.variance for p in points],
    )
    assert fit.conversion_gain == pytest.approx(sensor.conversion_gain, rel=0.05)
    assert fit.read_noise_e == pytest.approx(sensor.read_noise_e, rel=0.1)


def test_two_independent_lab_experiments_agree_on_conversion_gain():
    exp = load_lab("lab_exposure_sweep.csv")
    ptc = fit_photon_transfer(exp["mean_dn"], exp["sigma_diff_dn"] ** 2 / 2)
    gain = load_lab("lab_gain_sweep.csv")
    sweep = fit_gain_sweep(gain["gain"], gain["mean_dn"], gain["sigma_diff_dn"] ** 2 / 2)
    assert ptc.conversion_gain == pytest.approx(0.0217, abs=5e-4)
    assert sweep.conversion_gain == pytest.approx(0.0213, abs=5e-4)
    assert ptc.conversion_gain == pytest.approx(sweep.conversion_gain, rel=0.05)
    # datasheet: read noise < 0.5 LSB (8 bit) typical
    assert 0.3 < ptc.read_noise_dn < 0.7
    assert 0.3 < sweep.read_noise_dn < 0.7


def test_fits_need_enough_points():
    with pytest.raises(ValueError):
        fit_photon_transfer([10.0], [1.0])
    with pytest.raises(ValueError):
        fit_gain_sweep([1.0, 2.0], [128.0, 128.0], [3.0, 7.0])
