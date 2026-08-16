import numpy as np

from drift_sense.matching import ZnccPlan, direct_zncc, find_peaks


def test_fft_zncc_matches_trusted_reference():
    rng = np.random.default_rng(4)
    search = rng.normal(size=(9, 10))
    template = rng.normal(size=(3, 4))
    np.testing.assert_allclose(ZnccPlan(search).match(template),
                               direct_zncc(search, template), atol=1e-6)


def test_constant_inputs_are_safe():
    assert np.all(ZnccPlan(np.ones((6, 6))).match(np.ones((2, 2))) == 0)
    assert np.all(ZnccPlan(np.ones((6, 6))).match(np.array([[0, 1], [1, 0]], dtype=float)) == 0)


def test_peaks_and_nms():
    score = np.zeros((12, 12), dtype=float)
    score[3, 3] = 1.0
    score[4, 4] = 0.95
    score[9, 9] = 0.9
    peaks = find_peaks(score, relative_threshold=0.8, nms_radius=2)
    assert [(p.x, p.y) for p in peaks] == [(3, 3), (9, 9)]

