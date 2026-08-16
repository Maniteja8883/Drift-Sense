"""Compatibility exports for the official numerical primitives."""

from drift_sense.io import load_gray, to_gray_float32
from drift_sense.matching import ZnccPlan, direct_zncc, fft_zncc, find_peaks
from drift_sense.preprocessing import area_downsample, rotate_scale
from drift_sense.refinement import subpixel_refine

__all__ = ["load_gray", "to_gray_float32", "ZnccPlan", "direct_zncc",
           "fft_zncc", "find_peaks", "area_downsample", "rotate_scale",
           "subpixel_refine", "score_to_center"]


def score_to_center(x, y, template_size=100):
    offset = (template_size - 1) / 2.0
    return x + offset, y + offset
