"""
Drift-Sense Inference Pipeline
==============================
Given a high-resolution reference image and a low-resolution search image,
recover the physical (X, Y) location of the reference feature within the
search frame with sub-pixel precision.

Pipeline
--------
1. Template extraction  : area-average downsample the 1000x1000 reference by
                          10x -> pristine 100x100 template (physical 1 um).
2. Coarse stage         : search downsampled 4x (250x250) + template 25x25,
                          ZNCC over all 15 rotation/scale candidates, NMS,
                          center-proximity tie-break -> coarse candidate.
3. Fine stage           : local ROI (<=256x256) in the full search image,
                          full-resolution 100x100 template ZNCC over the 3x3
                          (angle, scale) neighborhood, parabolic sub-pixel
                          refinement of the winning peak.
4. Coordinate mapping   : peak top-left + (100-1)/2 = 49.5 -> feature center.

CLI:
    python inference.py --reference <path> --search <path>
    prints strictly:  <X> <Y>
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple, List, Dict, Optional

import numpy as np

from config import (
    REF_FINE_SIZE, SEARCH_FINE_SIZE, SCALE_FACTOR, TEMPLATE_SIZE,
    CENTER_PRIOR, TEMPLATE_CENTER_OFFSET,
    ROTATION_DEGREES, SCALE_FACTORS,
    SCORE_REL_THRESHOLD, NMS_RADIUS, TIE_BREAKER_FRACTION,
    COARSE_FACTOR, COARSE_ROI_RADIUS, EARLY_EXIT_SCORE, MIN_CANDIDATES,
    COARSE_EARLY_EXIT_SCORE,
)
from common import (
    load_gray, area_downsample, fast_rotate_scale, ZnccPlan,
    subpixel_refine_2d, find_peaks,
)

# ---------------------------------------------------------------------------
# Candidate ordering  (likelihood-descending for early exit)
# ---------------------------------------------------------------------------
_CAND_ORDER: List[Tuple[float, float]] = [
    (0.0, 1.00),
    (-1.5, 1.00), (1.5, 1.00),
    (0.0, 0.95), (0.0, 1.05),
    (-1.5, 0.95), (-1.5, 1.05),
    (1.5, 0.95), (1.5, 1.05),
    (-3.0, 1.00), (3.0, 1.00),
    (-3.0, 0.95), (-3.0, 1.05),
    (3.0, 0.95), (3.0, 1.05),
]


def _grid_neighborhood(angle: float, scale: float,
                       half: int = 1) -> List[Tuple[float, float]]:
    """3x3 neighborhood of (angle, scale) clipped to the search grid."""
    near_angles = [a for a in ROTATION_DEGREES if abs(a - angle) <= half * 1.5 + 1e-9]
    near_scales = [s for s in SCALE_FACTORS if abs(s - scale) <= half * 0.05 + 1e-9]
    return [(a, s) for a in near_angles for s in near_scales]


def _closest_to_center(peaks: List[dict], center: Tuple[float, float],
                       frac: float = TIE_BREAKER_FRACTION) -> dict:
    """Pick the peak nearest `center` among those within `frac` of max score."""
    if not peaks:
        raise RuntimeError("no correlation peaks found")
    best = max(peaks, key=lambda p: p['score'])
    band = best['score'] * (1.0 - frac)
    cx, cy = center
    eligible = [p for p in peaks if p['score'] >= band]
    return min(eligible,
               key=lambda p: (p['x'] - cx) ** 2 + (p['y'] - cy) ** 2)


# ---------------------------------------------------------------------------
# Coarse stage
# ---------------------------------------------------------------------------
def _coarse_scan(search: np.ndarray, template: np.ndarray,
                 coarse_factor: int
                 ) -> Dict[str, float]:
    """
    Evaluate all 15 (angle, scale) candidates on a downsampled image pair.
    Returns best {'angle', 'scale', 'x', 'y'} in full-res coordinates and the
    score map of the best candidate (for NMS / tie-breaking).
    """
    coarse_search = area_downsample(search, coarse_factor)
    coarse_tpl = area_downsample(template, coarse_factor)
    plan = ZnccPlan(coarse_search)

    best = {'score': -np.inf, 'angle': 0.0, 'scale': 1.0}
    best_map = None
    n_eval = 0
    for angle, scale in _CAND_ORDER:
        warped = fast_rotate_scale(coarse_tpl, angle, scale)
        score = plan.match(warped)
        n_eval += 1
        s = float(score.max())
        if s > best['score']:
            best['score'] = s
            best['angle'] = angle
            best['scale'] = scale
            best_map = score
        # Early exit: high-confidence match after minimum coverage.
        if best['score'] >= COARSE_EARLY_EXIT_SCORE and n_eval >= MIN_CANDIDATES:
            break

    # ---- NMS + center-proximity tie-break on the best coarse map ---------
    peaks = find_peaks(best_map, rel_threshold=SCORE_REL_THRESHOLD,
                       nms_radius=NMS_RADIUS / coarse_factor)
    if not peaks:
        py, px = np.unravel_index(np.argmax(best_map), best_map.shape)
        peaks = [{'score': float(best_map[py, px]), 'x': int(px), 'y': int(py)}]

    # Scale coarse center prior into coarse space for the tie-breaker.
    center_c = (CENTER_PRIOR[0] / coarse_factor, CENTER_PRIOR[1] / coarse_factor)
    peak = _closest_to_center(peaks, center_c, TIE_BREAKER_FRACTION)

    # Convert coarse peak top-left to full-resolution feature center estimate.
    cf = coarse_factor
    tpl_c = coarse_tpl.shape[0]
    center_full_x = (peak['x'] + (tpl_c - 1) / 2.0) * cf
    center_full_y = (peak['y'] + (tpl_c - 1) / 2.0) * cf
    return {
        'score': float(peak['score']),
        'angle': best['angle'],
        'scale': best['scale'],
        'x': float(center_full_x),
        'y': float(center_full_y),
    }


# ---------------------------------------------------------------------------
# Fine stage
# ---------------------------------------------------------------------------
def _fine_match(search: np.ndarray, template: np.ndarray,
                center_guess: Tuple[float, float],
                angle: float, scale: float
                ) -> Dict[str, float]:
    """Full-resolution ZNCC in a local ROI + sub-pixel refinement."""
    H, W = search.shape
    cx, cy = center_guess
    r = COARSE_ROI_RADIUS
    y0 = int(np.clip(cy - r, 0, H - TEMPLATE_SIZE))
    x0 = int(np.clip(cx - r, 0, W - TEMPLATE_SIZE))
    y1 = int(np.clip(cy + r, y0 + TEMPLATE_SIZE, H))
    x1 = int(np.clip(cx + r, x0 + TEMPLATE_SIZE, W))
    roi = search[y0:y1, x0:x1]

    plan = ZnccPlan(roi)
    warped = fast_rotate_scale(template, angle, scale)
    score = plan.match(warped)

    # NMS inside ROI then tie-break towards the ROI center (== true center).
    peaks = find_peaks(score, rel_threshold=SCORE_REL_THRESHOLD,
                       nms_radius=NMS_RADIUS)
    roi_center = ((x1 - x0) / 2.0, (y1 - y0) / 2.0)
    if not peaks:
        py, px = np.unravel_index(np.argmax(score), score.shape)
        peaks = [{'score': float(score[py, px]), 'x': int(px), 'y': int(py)}]
    peak = _closest_to_center(peaks, roi_center, TIE_BREAKER_FRACTION)

    px, py = subpixel_refine_2d(score, peak['x'], peak['y'])
    # Full-image top-left -> feature center.
    X = x0 + px + TEMPLATE_CENTER_OFFSET
    Y = y0 + py + TEMPLATE_CENTER_OFFSET
    return {'score': float(peak['score']), 'x': float(X), 'y': float(Y)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_xy(ref_path: str, search_path: str
               ) -> Tuple[float, float, float]:
    """
    Predict the (X, Y) center of the reference feature in the search image.

    Returns:
        (X, Y, confidence_score)
    """
    ref = load_gray(ref_path)
    search = load_gray(search_path)

    if ref.shape[0] != REF_FINE_SIZE or ref.shape[1] != REF_FINE_SIZE:
        ref = _fit_size(ref, REF_FINE_SIZE, REF_FINE_SIZE)
    if search.shape[0] != SEARCH_FINE_SIZE or search.shape[1] != SEARCH_FINE_SIZE:
        search = _fit_size(search, SEARCH_FINE_SIZE, SEARCH_FINE_SIZE)

    template = area_downsample(ref, SCALE_FACTOR)  # 100x100

    # ---- Coarse pass ------------------------------------------------------
    coarse = _coarse_scan(search, template, coarse_factor=4)

    # ---- Fine pass (3x3 grid neighborhood) --------------------------------
    candidates = _grid_neighborhood(coarse['angle'], coarse['scale'], half=1)
    fine_best = None
    for angle, scale in candidates:
        res = _fine_match(search, template, (coarse['x'], coarse['y']),
                          angle, scale)
        if fine_best is None or res['score'] > fine_best['score']:
            fine_best = res
        if fine_best['score'] >= EARLY_EXIT_SCORE:
            break

    return float(fine_best['x']), float(fine_best['y']), float(fine_best['score'])


def _fit_size(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Center-crop or pad an image to exactly (h, w)."""
    H, W = img.shape
    if H == h and W == w:
        return img
    out = np.zeros((h, w), dtype=np.float32)
    y0, x0 = max(0, (H - h) // 2), max(0, (W - w) // 2)
    sy0, sx0 = max(0, (h - H) // 2), max(0, (w - W) // 2)
    out[sy0:sy0 + min(h, H), sx0:sx0 + min(w, W)] = img[y0:y0 + min(h, H),
                                                        x0:x0 + min(w, W)]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Drift-Sense stage-drift recovery")
    ap.add_argument("--reference", required=True, help="high-res reference image")
    ap.add_argument("--search", required=True, help="low-res search image")
    ap.add_argument("--time", action="store_true", help="print latency to stderr")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    X, Y, score = predict_xy(args.reference, args.search)
    dt = (time.perf_counter() - t0) * 1000.0

    # Strict rubric output: "<X> <Y>"
    print(f"{X:.4f} {Y:.4f}")
    if args.time:
        print(f"[latency {dt:.2f} ms | score {score:.4f}]", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
