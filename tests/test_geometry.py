import numpy as np

from drift_sense.geometry import (apply_affine_point, center_from_top_left,
                                  rotation_scale_matrix, top_left_from_center)
from drift_sense.preprocessing import area_downsample


def test_center_mapping_is_derived_from_template_extent():
    assert center_from_top_left(10, 20, 100) == (59.5, 69.5)
    assert top_left_from_center(59.5, 69.5, 100) == (10.0, 20.0)


def test_area_downsample_block_average():
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.testing.assert_allclose(area_downsample(image, 2), [[2.5, 4.5], [10.5, 12.5]])


def test_rotation_scale_matrix_round_trip():
    forward, inverse = rotation_scale_matrix(3.0, 1.05, (50.0, 50.0))
    point = (63.0, 42.0)
    transformed = apply_affine_point(*point, forward)
    recovered = apply_affine_point(*transformed, inverse)
    np.testing.assert_allclose(recovered, point, atol=1e-10)

