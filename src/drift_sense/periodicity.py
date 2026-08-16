"""Peak-set diagnostics for periodic-layout ambiguity."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Tuple

from .matching import Peak


@dataclass(frozen=True)
class AmbiguityReport:
    """Interpretable evidence used by the confidence/status policy."""

    peak_count: int
    best_score: float
    second_score: float
    relative_margin: float
    separation_px: float
    center_distance_px: float
    ambiguous: bool


def analyze_peaks(peaks: Iterable[Peak], center: Tuple[float, float],
                  ambiguity_margin: float = 0.02,
                  min_separation_px: float = 10.0) -> AmbiguityReport:
    ordered = sorted(list(peaks), key=lambda peak: peak.score, reverse=True)
    if not ordered:
        return AmbiguityReport(0, 0.0, 0.0, 0.0, 0.0, float("inf"), False)
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    second_score = second.score if second else 0.0
    margin = best.score - second_score
    separation = hypot(best.x - second.x, best.y - second.y) if second else float("inf")
    center_distance = hypot(best.x - center[0], best.y - center[1])
    ambiguous = second is not None and margin <= float(ambiguity_margin) and \
        separation >= float(min_separation_px)
    return AmbiguityReport(len(ordered), best.score, second_score, margin,
                           separation, center_distance, ambiguous)
