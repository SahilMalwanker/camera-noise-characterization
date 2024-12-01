import numpy as np
import pytest

from camera_noise_characterization.filters import (
    add_impulse_noise,
    binomial_kernel,
    convolve,
    denoise_benchmark,
    median,
    median_noise_gain,
    white_noise_gain,
)


def test_binomial_kernel_is_outer_product_of_pascal_row():
    np.testing.assert_allclose(binomial_kernel(3), np.outer([1, 2, 1], [1, 2, 1]) / 16)
    for size in (3, 5, 7):
        k = binomial_kernel(size)
        assert k.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(k, k.T)


def test_even_kernel_sizes_are_rejected():
    with pytest.raises(ValueError):
        binomial_kernel(4)


@pytest.mark.parametrize(("size", "gain"), [(3, 6 / 16), (5, 70 / 256), (7, 924 / 4096)])
def test_white_noise_gain_closed_form(size, gain):
    assert white_noise_gain(binomial_kernel(size)) == pytest.approx(gain)


def test_median_gain_of_nine_gaussian_samples():
    assert median_noise_gain(9) == pytest.approx(0.41, abs=0.01)


def test_benchmark_matches_theory_for_white_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(128.0, 3.0, (512, 512))
    b = rng.normal(128.0, 3.0, (512, 512))
    results = denoise_benchmark(a, b)
    for r in results[1:]:
        assert r.measured_gain == pytest.approx(r.predicted_gain, rel=0.04), r.name


def test_median_wins_against_impulse_noise():
    rng = np.random.default_rng(2)
    clean = np.full((200, 200), 128.0)
    noisy = add_impulse_noise(clean, 0.05, rng)
    assert np.abs(median(noisy) - clean).mean() < 0.5
    assert np.abs(convolve(noisy, binomial_kernel(3)) - clean).mean() > 5.0
