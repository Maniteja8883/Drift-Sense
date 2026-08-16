import numpy as np

from drift_sense.refinement import subpixel_refine


def test_fractional_peak_refinement():
    yy, xx = np.mgrid[0:9, 0:9]
    score = 1.0 - ((xx - 4.25) ** 2 + 2.0 * (yy - 3.65) ** 2) / 100.0
    x, y = subpixel_refine(score, 4, 4)
    assert abs(x - 4.25) < 0.03
    assert abs(y - 3.65) < 0.03


def test_boundary_peak_is_stable():
    score = np.zeros((4, 4))
    score[0, 0] = 1.0
    assert subpixel_refine(score, 0, 0) == (0.0, 0.0)

