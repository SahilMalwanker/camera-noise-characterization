"""Image input/output helpers (grayscale, bit depth preserved)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: str | Path) -> np.ndarray:
    """Load an image as a 2-D array; colour is converted to luminance, 16-bit is preserved."""
    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            return np.asarray(im, dtype=np.uint16 if im.mode.startswith("I;16") else np.int32)
        if im.mode not in ("L", "1"):
            im = im.convert("L")
        return np.asarray(im, dtype=np.uint8)


def save_gray(path: str | Path, image: np.ndarray) -> None:
    """Save a 2-D array as an 8-bit PNG/TIFF (floats are clipped to [0, 255])."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
