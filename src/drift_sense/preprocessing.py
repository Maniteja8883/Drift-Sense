"""Small deterministic image preprocessing primitives."""

from __future__ import annotations

import numpy as np


def area_downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Block-average downsample with explicit divisibility and shape checks."""
    if int(factor) < 1:
        raise ValueError("factor must be >= 1")
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("area_downsample expects a 2-D image")
    if factor == 1:
        return arr.copy()
    h = arr.shape[0] - arr.shape[0] % factor
    w = arr.shape[1] - arr.shape[1] % factor
    if h == 0 or w == 0:
        raise ValueError("factor is larger than the image")
    return arr[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(
        axis=(1, 3), dtype=np.float32).astype(np.float32)


def rotate_scale(image: np.ndarray, angle_deg: float, scale: float) -> np.ndarray:
    """Rotate and scale around the image centre using bilinear sampling."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("rotate_scale expects a 2-D image")
    if scale <= 0:
        raise ValueError("scale must be positive")
    h, w = arr.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    theta = np.deg2rad(float(angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tx, ty = (xx - cx) / scale, (yy - cy) / scale
    sx = c * tx + s * ty + cx
    sy = -s * tx + c * ty + cy
    x0, y0 = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    fx, fy = sx - x0, sy - y0
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    out = ((1 - fx) * (1 - fy) * arr[y0, x0] + fx * (1 - fy) * arr[y0, x1] +
           (1 - fx) * fy * arr[y1, x0] + fx * fy * arr[y1, x1])
    return out.astype(np.float32)
