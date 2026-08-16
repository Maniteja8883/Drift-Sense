"""Public result type for official localization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class LocalizationResult:
    """Prediction plus evidence; ``confidence`` is explicitly heuristic."""

    x: float
    y: float
    score: float
    confidence: float
    status: str
    latency_ms: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def xy(self) -> Tuple[float, float]:
        return self.x, self.y

