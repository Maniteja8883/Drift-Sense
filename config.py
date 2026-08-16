"""
Drift-Sense Configuration Module
=================================
Centralizes all physical, geometric, and algorithmic parameters for the
semiconductor stage-drift recovery engine.

Physical Ground Truth:
  - Reference image: 1000x1000 px @ 1 nm/px (100x zoom, 1 um x 1 um FOV)
  - Search image:    1000x1000 px @ 10 nm/px (10x zoom, 10 um x 10 um FOV)
  - Scale factor between reference and search: 10x
  - The 1000x1000 reference maps to a 100x100 footprint in the search frame.

Hackathon Rubric Targets:
  - Accuracy @ 1 px  > 90%
  - Accuracy @ 5 px  > 99%
  - Single-pair latency < 80 ms
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Image geometry
# ---------------------------------------------------------------------------
REF_FINE_SIZE: int = 1000          # pixels (high-res reference)
SEARCH_FINE_SIZE: int = 1000       # pixels (low-res search)
SCALE_FACTOR: int = 10             # ref is SCALE_FACTORx finer than search
TEMPLATE_SIZE: int = REF_FINE_SIZE // SCALE_FACTOR  # 100 px

# Coordinate of the nominal center in the search image
CENTER_PRIOR: Tuple[float, float] = (500.0, 500.0)

# Sub-pixel center offset when the template spans [u, u+99] in each dim
# Center index = (TEMPLATE_SIZE - 1) / 2 = 49.5
TEMPLATE_CENTER_OFFSET: float = (TEMPLATE_SIZE - 1) / 2.0  # 49.5

# ---------------------------------------------------------------------------
# Search grids  (rotation x scale = 15 candidates)
# ---------------------------------------------------------------------------
ROTATION_DEGREES: List[float] = [-3.0, -1.5, 0.0, 1.5, 3.0]
SCALE_FACTORS: List[float] = [0.95, 1.00, 1.05]

# Total candidates for exhaustive pass
NUM_CANDIDATES: int = len(ROTATION_DEGREES) * len(SCALE_FACTORS)  # 15

# ---------------------------------------------------------------------------
# Correlation & NMS parameters
# ---------------------------------------------------------------------------
# Minimum score relative to global max to be considered a peak
SCORE_REL_THRESHOLD: float = 0.70

# Exclusion radius around a detected peak (pixels in search space)
NMS_RADIUS: float = 20.0

# Tie-breaker: candidates within this fraction of max score use center proximity
TIE_BREAKER_FRACTION: float = 0.05  # 5%

# Variance floor to prevent divide-by-zero in ZNCC
VARIANCE_EPSILON: float = 1e-6

# ---------------------------------------------------------------------------
# Hierarchical matching
# ---------------------------------------------------------------------------
# Coarse stage downsampling factor applied to the full search image.
COARSE_FACTOR: int = 4                       # 1000 -> 250 px
# Radius (px) of the fine-stage ROI crop around the coarse center.
COARSE_ROI_RADIUS: int = 160
# Minimum candidates evaluated before early exit is allowed.
MIN_CANDIDATES: int = 5
# Confidence that triggers early exit (coarse / fine stages).
COARSE_EARLY_EXIT_SCORE: float = 0.85
EARLY_EXIT_SCORE: float = 0.96

# ---------------------------------------------------------------------------
# Latency budget
# ---------------------------------------------------------------------------
MAX_LATENCY_MS: float = 80.0

# ---------------------------------------------------------------------------
# Dataset generator defaults
# ---------------------------------------------------------------------------
SEM_LAMBDA: float = 100.0          # photon/electron count for Poisson shot noise
EDGE_CHARGE_STRENGTH: float = 0.15  # fractional bloom at material boundaries
CHARGING_RAMP_AMPLITUDE: float = 0.05  # max low-freq intensity variation
SCANLINE_JITTER_PX: float = 0.3   # max row displacement in pixels
ROTATION_RANGE_DEG: Tuple[float, float] = (-3.0, 3.0)
SCALE_RANGE: Tuple[float, float] = (0.95, 1.05)

# ---------------------------------------------------------------------------
# Evaluation thresholds
# ---------------------------------------------------------------------------
EVAL_THRESHOLDS_PX: List[float] = [1.0, 3.0, 5.0]
SCORE_THRESHOLDS: List[float] = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
NUM_CHALLENGE_PAIRS: int = 30
NUM_DRAM_PAIRS: int = 15
NUM_FINFET_PAIRS: int = 15
