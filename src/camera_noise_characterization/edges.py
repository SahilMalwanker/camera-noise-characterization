"""Edge operators: Sobel, Prewitt, Laplace, LoG and a from-scratch Canny detector."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_Y = SOBEL_X.T.copy()
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
PREWITT_Y = PREWITT_X.T.copy()
OPERATORS = {"sobel": (SOBEL_X, SOBEL_Y), "prewitt": (PREWITT_X, PREWITT_Y)}


def laplacian_kernel(alpha: float = 0.2) -> np.ndarray:
    """3x3 Laplacian with diagonal weight ``alpha`` (same family as MATLAB's ``fspecial``)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    a, b = alpha / 4.0, (1.0 - alpha) / 4.0
    return 4.0 / (alpha + 1.0) * np.array([[a, b, a], [b, -1.0, b], [a, b, a]])


def gradients(image: np.ndarray, operator: str = "sobel") -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical derivative; positive where brightness grows to the right/down."""
    kx, ky = OPERATORS[operator]
    img = np.asarray(image, dtype=float)
    return (
        ndimage.correlate(img, kx, mode="nearest"),
        ndimage.correlate(img, ky, mode="nearest"),
    )


def laplacian(image: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return ndimage.correlate(
        np.asarray(image, dtype=float), laplacian_kernel(alpha), mode="nearest"
    )


def threshold_magnitude(magnitude: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Binary edge map; the default cut-off is twice the rms gradient (MATLAB's heuristic)."""
    if threshold is None:
        threshold = 2.0 * float(np.sqrt(np.mean(magnitude**2)))
    return magnitude > threshold


def non_maximum_suppression(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Keep only pixels that are maximal along their gradient direction (4 sectors)."""
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    sector = ((angle + 22.5) // 45.0).astype(int) % 4
    padded = np.pad(mag, 1, mode="constant")
    h, w = mag.shape

    def shifted(dy: int, dx: int) -> np.ndarray:
        return padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

    keep = np.zeros(mag.shape, dtype=bool)
    for s, (dy, dx) in enumerate([(0, 1), (1, 1), (1, 0), (1, -1)]):
        keep |= (sector == s) & (mag > shifted(dy, dx)) & (mag >= shifted(-dy, -dx))
    return np.where(keep, mag, 0.0)


def hysteresis(mag: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep weak edge pixels (``> low``) only if they are 8-connected to a strong one."""
    weak = mag > low
    labels, n = ndimage.label(weak, structure=np.ones((3, 3), dtype=bool))
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(labels[weak & (mag >= high)])] = True
    keep[0] = False
    return keep[labels]


def auto_canny_thresholds(mag: np.ndarray, non_edge_fraction: float = 0.7) -> tuple[float, float]:
    """MATLAB's rule: ``high`` is where a 64-bin magnitude histogram passes 70 % of the pixels."""
    if not 0.0 < non_edge_fraction < 1.0:
        raise ValueError("non_edge_fraction must lie strictly between 0 and 1")
    peak = float(mag.max())
    if peak == 0.0:
        return 0.0, 0.0
    counts, _ = np.histogram(mag, bins=64, range=(0.0, peak))
    idx = int(np.searchsorted(np.cumsum(counts), non_edge_fraction * mag.size, side="right"))
    high = (idx + 1) / 64.0 * peak
    return 0.4 * high, high


def canny(
    image: np.ndarray,
    sigma: float = 1.4,
    low: float | None = None,
    high: float | None = None,
    non_edge_fraction: float = 0.7,
) -> np.ndarray:
    """Canny edge detector: Gaussian smoothing, Sobel, non-maximum suppression, hysteresis.

    Thresholds are absolute gradient magnitudes; omitted ones follow
    :func:`auto_canny_thresholds` with ``non_edge_fraction`` and ``low = 0.4 * high``.
    Raise ``non_edge_fraction`` on textured scenes to keep only the strongest edges.
    """
    smooth = ndimage.gaussian_filter(np.asarray(image, dtype=float), sigma)
    gx, gy = gradients(smooth, "sobel")
    mag = np.hypot(gx, gy)
    if high is None:
        auto_low, high = auto_canny_thresholds(mag, non_edge_fraction)
        low = auto_low if low is None else low
    elif low is None:
        low = 0.4 * high
    return hysteresis(non_maximum_suppression(mag, gx, gy), low, high)


def log_edges(image: np.ndarray, sigma: float = 2.0, threshold: float | None = None) -> np.ndarray:
    """Marr-Hildreth: zero crossings of the Laplacian of Gaussian with a slope threshold."""
    response = ndimage.gaussian_laplace(np.asarray(image, dtype=float), sigma)
    if threshold is None:
        threshold = 0.75 * float(np.mean(np.abs(response)))
    edges = np.zeros(response.shape, dtype=bool)
    for axis in (0, 1):
        a = response
        b = np.roll(response, -1, axis=axis)
        crossing = (a * b < 0) & (np.abs(a - b) > threshold)
        valid = np.ones_like(crossing)
        if axis == 0:
            valid[-1, :] = False
        else:
            valid[:, -1] = False
        crossing &= valid
        take_a = crossing & (np.abs(a) <= np.abs(b))
        edges |= take_a
        edges |= np.roll(crossing & ~take_a, 1, axis=axis)
    return edges


def edge_scores(
    predicted: np.ndarray,
    truth: np.ndarray,
    tolerance: float = 1.5,
    care: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Precision, recall and F1 of an edge map, matching within ``tolerance`` pixels."""
    pred = np.asarray(predicted, dtype=bool)
    true = np.asarray(truth, dtype=bool)
    if care is not None:
        pred = pred & care
        true = true & care
    if not pred.any() or not true.any():
        return 0.0, 0.0, 0.0
    dist_to_true = ndimage.distance_transform_edt(~true)
    dist_to_pred = ndimage.distance_transform_edt(~pred)
    precision = float(np.mean(dist_to_true[pred] <= tolerance))
    recall = float(np.mean(dist_to_pred[true] <= tolerance))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1
