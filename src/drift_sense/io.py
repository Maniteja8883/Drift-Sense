"""Image input/output with explicit grayscale and normalization rules."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


PathLike = Union[str, Path]


def to_gray_float32(image: np.ndarray) -> np.ndarray:
    """Convert 2-D/RGB/RGBA input to grayscale float32 in approximately [0, 1]."""
    if image is None:
        raise ValueError("image must not be None")
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr.astype(np.float32, copy=False)
    elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        if arr.shape[2] == 1:
            gray = arr[..., 0].astype(np.float32, copy=False)
        else:
            gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] +
                    0.114 * arr[..., 2]).astype(np.float32)
    else:
        raise ValueError(f"expected 2-D or RGB/RGBA image, got shape {arr.shape}")

    if arr.dtype == np.uint8:
        gray = gray / 255.0
    elif arr.dtype == np.uint16:
        gray = gray / 65535.0
    return np.asarray(gray, dtype=np.float32)


def load_gray(path: PathLike) -> np.ndarray:
    """Load an image file as a 2-D float32 grayscale array."""
    with Image.open(path) as image:
        return to_gray_float32(np.asarray(image))


def save_gray(image: np.ndarray, path: PathLike) -> None:
    """Save a float image as an 8-bit PNG, clipping to [0, 1]."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("save_gray expects a 2-D array")
    pixels = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def fit_center(image: np.ndarray, shape: tuple) -> np.ndarray:
    """Center-crop or zero-pad an image to ``shape``; useful only for demo IO."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("fit_center expects a 2-D array")
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    copy_h, copy_w = min(h, arr.shape[0]), min(w, arr.shape[1])
    src_y = max(0, (arr.shape[0] - h) // 2)
    src_x = max(0, (arr.shape[1] - w) // 2)
    dst_y = max(0, (h - arr.shape[0]) // 2)
    dst_x = max(0, (w - arr.shape[1]) // 2)
    out[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = arr[
        src_y:src_y + copy_h, src_x:src_x + copy_w]
    return out
