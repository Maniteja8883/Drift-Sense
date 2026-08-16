"""Subpixel peak refinement with explicit boundary handling."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _symmetric_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    return 0.5 * (left - right) / denominator


def subpixel_refine(score_map: np.ndarray, peak_x: int, peak_y: int) -> Tuple[float, float]:
    """Return a peak position refined by separable three-point parabolas."""
    scores = np.asarray(score_map, dtype=np.float64)
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError("score_map must be non-empty and 2-D")
    x, y = int(peak_x), int(peak_y)
    if not (0 <= x < scores.shape[1] and 0 <= y < scores.shape[0]):
        raise ValueError("peak is outside score_map")
    dx = 0.0 if x == 0 or x == scores.shape[1] - 1 else _symmetric_offset(
        scores[y, x - 1], scores[y, x], scores[y, x + 1])
    dy = 0.0 if y == 0 or y == scores.shape[0] - 1 else _symmetric_offset(
        scores[y - 1, x], scores[y, x], scores[y + 1, x])
    return float(x + np.clip(dx, -0.5, 0.5)), float(y + np.clip(dy, -0.5, 0.5))

