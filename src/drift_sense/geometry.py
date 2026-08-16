"""Coordinate and transform math used by localization and tests."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import Geometry


def center_from_top_left(x: float, y: float, template_size: int) -> Tuple[float, float]:
    """Map a valid template top-left to its pixel-center coordinate."""
    offset = (int(template_size) - 1) / 2.0
    return float(x + offset), float(y + offset)


def top_left_from_center(x: float, y: float, template_size: int) -> Tuple[float, float]:
    """Inverse of :func:`center_from_top_left`."""
    offset = (int(template_size) - 1) / 2.0
    return float(x - offset), float(y - offset)


def template_center_from_geometry(x: float, y: float, geometry: Geometry) -> Tuple[float, float]:
    return center_from_top_left(x, y, geometry.template_size)


def search_to_physical_nm(x: float, y: float, geometry: Geometry) -> Tuple[float, float]:
    """Convert search-image pixels to nanometres from the image origin."""
    return float(x * geometry.search_nm_per_pixel), float(y * geometry.search_nm_per_pixel)


def rotation_scale_matrix(angle_deg: float, scale: float,
                          center_xy: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return forward and inverse 2-D affine matrices for a centered warp."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    theta = np.deg2rad(float(angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    cx, cy = center_xy
    linear = np.array([[c * scale, -s * scale],
                       [s * scale, c * scale]], dtype=np.float64)
    offset = np.array([cx, cy], dtype=np.float64) - linear @ np.array([cx, cy])
    inverse_linear = np.linalg.inv(linear)
    inverse_offset = -inverse_linear @ offset
    forward = np.concatenate([linear, offset[:, None]], axis=1)
    inverse = np.concatenate([inverse_linear, inverse_offset[:, None]], axis=1)
    return forward, inverse


def apply_affine_point(x: float, y: float, affine: np.ndarray) -> Tuple[float, float]:
    point = affine[:, :2] @ np.array([x, y], dtype=np.float64) + affine[:, 2]
    return float(point[0]), float(point[1])

