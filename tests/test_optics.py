import math

import pytest

from camera_noise_characterization.optics import (
    BAUMER_TXG08,
    Lens,
    OpticalSetup,
    airy_diameter_um,
    depth_of_field_mm,
    diffraction_limited_f_number,
    field_of_view_mm,
    hyperfocal_mm,
    magnification,
)

LAB = OpticalSetup(BAUMER_TXG08, Lens(focal_mm=12.0, f_number=1.4), distance_mm=342.0)


def test_sensor_geometry_matches_datasheet():
    assert BAUMER_TXG08.width_mm == pytest.approx(4.80, abs=0.01)
    assert BAUMER_TXG08.height_mm == pytest.approx(3.61, abs=0.01)


def test_depth_of_field_matches_hand_calculation():
    near, far, total = LAB.depth_of_field_mm()
    assert near == pytest.approx(336.97, abs=0.01)
    assert far == pytest.approx(347.18, abs=0.01)
    assert total == pytest.approx(10.21, abs=0.01)


def test_thin_lens_and_pinhole_field_of_view():
    w_thin, h_thin = LAB.field_of_view_mm()
    w_pin, _ = LAB.field_of_view_mm("pinhole")
    assert w_thin == pytest.approx(BAUMER_TXG08.width_mm * 330 / 12)
    assert w_pin == pytest.approx(BAUMER_TXG08.width_mm * 342 / 12)
    assert h_thin / w_thin == pytest.approx(776 / 1032)


def test_object_pixel_is_pixel_pitch_over_magnification():
    # 4.65 um * (342 - 12) / 12 = 0.128 mm - dividing instead of multiplying gives 0.00016 mm
    assert LAB.object_pixel_mm() == pytest.approx(0.1279, abs=1e-4)
    assert LAB.object_pixel_mm() == pytest.approx(LAB.field_of_view_mm()[0] / 1032)


def test_angles_of_view():
    h, v, d = LAB.angle_of_view_deg()
    assert h == pytest.approx(22.62, abs=0.01)
    assert v == pytest.approx(17.10, abs=0.01)
    assert d > h > v


def test_hyperfocal_distance_ends_depth_of_field_at_infinity():
    h = hyperfocal_mm(12.0, 1.4, 0.00465)
    near, far = depth_of_field_mm(12.0, 1.4, h, 0.00465)
    assert math.isinf(far)
    assert near == pytest.approx(h / 2.0)


def test_stopping_down_increases_depth_of_field():
    totals = [
        OpticalSetup(BAUMER_TXG08, Lens(12.0, n), 342.0).depth_of_field_mm()[2]
        for n in (1.4, 2.8, 5.6, 11)
    ]
    assert totals == sorted(totals)


def test_diffraction_limits():
    assert airy_diameter_um(1.4) == pytest.approx(1.879, abs=1e-3)
    assert diffraction_limited_f_number(4.65) == pytest.approx(6.93, abs=0.01)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        magnification(12.0, 10.0)
    with pytest.raises(ValueError):
        field_of_view_mm(4.8, 12.0, 342.0, model="fisheye")
