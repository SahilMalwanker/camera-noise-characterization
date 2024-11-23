import numpy as np
import pytest

from camera_noise_characterization.noise import pair_stats
from camera_noise_characterization.sensor import SensorModel, SimulatedCamera


def small_sensor(**kw) -> SensorModel:
    return SensorModel(width=160, height=120, **kw)


def test_mean_is_linear_in_exposure():
    cam = SimulatedCamera(small_sensor(), photon_rate=3.0, seed=1)
    means = []
    for t in (1000.0, 2000.0, 4000.0):
        cam.exposure_us = t
        means.append(cam.grab().mean() - cam.sensor.black_level_dn)
    assert means[1] / means[0] == pytest.approx(2.0, rel=0.03)
    assert means[2] / means[0] == pytest.approx(4.0, rel=0.03)


def test_mean_matches_noise_free_model():
    sensor = small_sensor()
    cam = SimulatedCamera(sensor, photon_rate=3.0, seed=2)
    cam.exposure_us = 3000.0
    expected = sensor.mean_dn(3.0 * 3000.0, 3000.0)
    assert cam.grab().mean() == pytest.approx(expected, rel=0.01)


def test_temporal_variance_follows_photon_transfer():
    sensor = small_sensor(prnu=0.0, dsnu_e=0.0)
    cam = SimulatedCamera(sensor, photon_rate=3.0, seed=3)
    cam.exposure_us = 6000.0
    s = pair_stats(cam.grab(), cam.grab(), sensor.black_level_dn)
    k = sensor.conversion_gain
    expected = k * s.mean + (k * sensor.read_noise_e) ** 2 + 1 / 12
    assert s.variance == pytest.approx(expected, rel=0.06)


def test_saturation_clips_at_full_scale():
    cam = SimulatedCamera(small_sensor(), photon_rate=3.0, seed=4)
    cam.exposure_us = 1e6
    assert (cam.grab() == 255).all()


def test_fixed_pattern_noise_cancels_in_frame_difference():
    def noise_for(prnu: float) -> tuple[float, float]:
        cam = SimulatedCamera(small_sensor(prnu=prnu), photon_rate=3.0, seed=5)
        cam.exposure_us = 6000.0
        a, b = cam.grab(), cam.grab()
        return float(a.std()), pair_stats(a, b).sigma

    spatial_clean, temporal_clean = noise_for(0.0)
    spatial_fpn, temporal_fpn = noise_for(0.05)
    assert spatial_fpn > 2 * spatial_clean
    assert temporal_fpn == pytest.approx(temporal_clean, rel=0.05)


def test_seeded_camera_is_reproducible():
    frames = []
    for _ in range(2):
        cam = SimulatedCamera(small_sensor(seed=9), seed=9)
        frames.append(cam.grab())
    np.testing.assert_array_equal(frames[0], frames[1])


def test_scene_must_match_sensor_shape():
    with pytest.raises(ValueError):
        SimulatedCamera(small_sensor(), scene=np.ones((10, 10)))


def test_sixteen_bit_output():
    sensor = small_sensor(bit_depth=12, conversion_gain=0.3)
    cam = SimulatedCamera(sensor, seed=6)
    frame = cam.grab()
    assert frame.dtype == np.uint16
    assert frame.max() <= 4095
