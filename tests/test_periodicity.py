from drift_sense.matching import Peak
from drift_sense.periodicity import analyze_peaks


def test_near_equal_separated_peaks_are_ambiguous():
    report = analyze_peaks([Peak(0.90, 50, 50), Peak(0.89, 100, 50)], (50, 50))
    assert report.ambiguous
    assert report.separation_px == 50.0

