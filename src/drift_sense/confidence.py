"""Heuristic confidence and explicit status classification."""

from __future__ import annotations

from .periodicity import AmbiguityReport


def confidence_from_evidence(report: AmbiguityReport, center_distance_px: float,
                             center_radius_px: float) -> float:
    """Combine score, peak margin, and prior agreement into a bounded heuristic.

    This value is a ranking/diagnostic signal, not a calibrated probability.
    """
    score_term = max(0.0, min(1.0, (report.best_score + 1.0) / 2.0))
    margin_term = max(0.0, min(1.0, report.relative_margin * 25.0))
    prior_term = max(0.0, min(1.0, 1.0 - center_distance_px / max(center_radius_px, 1.0)))
    return float(max(0.0, min(1.0, 0.65 * score_term + 0.25 * margin_term +
                             0.10 * prior_term)))


def classify_status(report: AmbiguityReport, confidence: float,
                    center_distance_px: float, ood_radius_px: float,
                    low_score: float) -> str:
    """Return SUCCESS, AMBIGUOUS, LOW_CONFIDENCE, or OUT_OF_DISTRIBUTION."""
    if center_distance_px > ood_radius_px:
        return "OUT_OF_DISTRIBUTION"
    if report.ambiguous:
        return "AMBIGUOUS"
    if report.best_score < low_score or confidence < 0.35:
        return "LOW_CONFIDENCE"
    return "SUCCESS"
