import numpy as np
import pytest

from camera_noise_characterization.edges import (
    canny,
    edge_scores,
    gradients,
    laplacian_kernel,
    log_edges,
    threshold_magnitude,
)
from camera_noise_characterization.targets import siemens_star


def step_image(size: int = 64, x0: int = 32) -> np.ndarray:
    img = np.full((size, size), 40.0)
    img[:, x0:] = 200.0
    return img


def test_sobel_sign_and_orientation():
    gx, gy = gradients(step_image(), "sobel")
    assert gx[:, 31:33].min() > 0
    assert np.abs(gy).max() == 0
    assert np.abs(gx[:, :30]).max() == 0


def test_laplacian_kernel_matches_reference_values():
    expected = np.array([[1 / 6, 2 / 3, 1 / 6], [2 / 3, -10 / 3, 2 / 3], [1 / 6, 2 / 3, 1 / 6]])
    np.testing.assert_allclose(laplacian_kernel(0.2), expected)
    assert laplacian_kernel(0.0).sum() == pytest.approx(0.0)


def test_canny_returns_one_pixel_wide_edge_at_the_step():
    edges = canny(step_image(), sigma=1.0)
    rows = edges[4:-4]
    assert (rows.sum(axis=1) == 1).all()
    assert set(np.nonzero(rows)[1]) <= {31, 32}


def test_log_zero_crossings_find_the_step():
    edges = log_edges(step_image(), sigma=1.5)
    cols = np.nonzero(edges[10:-10])[1]
    assert cols.size > 0
    assert np.all(np.abs(cols - 31.5) <= 1.5)


def test_edge_scores_perfect_and_empty_maps():
    truth = np.zeros((20, 20), dtype=bool)
    truth[:, 10] = True
    assert edge_scores(truth, truth) == (1.0, 1.0, 1.0)
    assert edge_scores(np.zeros_like(truth), truth) == (0.0, 0.0, 0.0)


def test_star_ground_truth_and_canny_beats_plain_sobel_under_noise():
    image, truth, care = siemens_star(size=192, spokes=12)
    assert truth[care].mean() > 0.03
    rng = np.random.default_rng(3)
    noisy = 255 * image + rng.normal(0.0, 12.0, image.shape)
    gx, gy = gradients(noisy)
    f1_sobel = edge_scores(threshold_magnitude(np.hypot(gx, gy)), truth, care=care)[2]
    f1_canny = edge_scores(canny(noisy, sigma=1.4), truth, care=care)[2]
    assert f1_canny > f1_sobel
    assert f1_canny > 0.7


def test_wider_canny_smoothing_survives_low_contrast_to_noise():
    image, truth, care = siemens_star(size=256, spokes=16, low=0.38, high=0.62)
    noisy = 255 * image + np.random.default_rng(4).normal(0.0, 30.0, image.shape)
    gx, gy = gradients(noisy)
    assert edge_scores(canny(noisy, sigma=2.5), truth, care=care)[2] > 0.9
    assert edge_scores(threshold_magnitude(np.hypot(gx, gy)), truth, care=care)[2] < 0.75
