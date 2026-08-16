import numpy as np
import pytest

from drift_sense.pipeline import LocalizationEngine


def test_pipeline_rejects_wrong_dimensions():
    with pytest.raises(ValueError):
        LocalizationEngine().predict_arrays(np.zeros((10, 10)), np.zeros((10, 10)))


def test_pipeline_smoke_on_simple_center_feature():
    ref = np.zeros((1000, 1000), dtype=np.float32)
    ref[100:900:20, 100:900] = 1.0
    search = np.zeros_like(ref)
    template = ref.reshape(100, 10, 100, 10).mean(axis=(1, 3))
    search[450:550, 450:550] = template
    result = LocalizationEngine().predict_arrays(ref, search)
    assert np.hypot(result.x - 499.5, result.y - 499.5) < 2.0
    assert result.status in {"SUCCESS", "AMBIGUOUS", "LOW_CONFIDENCE"}

