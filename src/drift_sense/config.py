"""Central configuration for the official, deterministic localization path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Geometry:
    """Image geometry in search-image pixels and nanometres per pixel."""

    reference_size: int = 1000
    search_size: int = 1000
    scale_factor: int = 10
    search_nm_per_pixel: float = 10.0

    @property
    def template_size(self) -> int:
        return self.reference_size // self.scale_factor

    @property
    def template_center_offset(self) -> float:
        return (self.template_size - 1) / 2.0


@dataclass(frozen=True)
class PipelineConfig:
    """Algorithm parameters used by the official inference and benchmark."""

    geometry: Geometry = field(default_factory=Geometry)
    center_prior: Tuple[float, float] = (500.0, 500.0)
    rotation_degrees: Tuple[float, ...] = (-3.0, -1.5, 0.0, 1.5, 3.0)
    scale_candidates: Tuple[float, ...] = (0.95, 1.0, 1.05)
    coarse_factor: int = 4
    coarse_roi_radius: int = 160
    nms_radius_px: float = 20.0
    score_relative_threshold: float = 0.70
    tie_break_fraction: float = 0.05
    variance_epsilon: float = 1e-8
    low_confidence_score: float = 0.25
    ambiguity_margin: float = 0.02
    ambiguity_min_separation_px: float = 10.0
    ood_center_radius_px: float = 250.0

    @property
    def candidate_count(self) -> int:
        return len(self.rotation_degrees) * len(self.scale_candidates)


DEFAULT_CONFIG = PipelineConfig()

BENCHMARK_SEED = 0xD1575317
BENCHMARK_VERSION = "synthetic-v2"
EVAL_THRESHOLDS_PX: Tuple[float, ...] = (1.0, 3.0, 5.0)
SCORE_THRESHOLDS: Tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
