"""Official deterministic coarse-to-fine FFT-ZNCC localization pipeline."""

from __future__ import annotations

import time
from math import hypot
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .confidence import classify_status, confidence_from_evidence
from .config import DEFAULT_CONFIG, PipelineConfig
from .geometry import center_from_top_left
from .localization import LocalizationResult
from .matching import Peak, ZnccPlan, find_peaks
from .periodicity import analyze_peaks
from .preprocessing import area_downsample, rotate_scale
from .refinement import subpixel_refine


def _candidate_grid(config: PipelineConfig) -> List[Tuple[float, float]]:
    return [(angle, scale) for angle in config.rotation_degrees
            for scale in config.scale_candidates]


def _neighborhood(angle: float, scale: float, config: PipelineConfig) -> List[Tuple[float, float]]:
    return [(a, s) for a, s in _candidate_grid(config)
            if abs(a - angle) <= 1.5 + 1e-9 and abs(s - scale) <= 0.05 + 1e-9]


def _best_peak(peaks: Iterable[Peak], center: Tuple[float, float],
               tie_fraction: float) -> Peak:
    peaks = list(peaks)
    if not peaks:
        raise RuntimeError("no valid correlation peaks")
    best_score = max(p.score for p in peaks)
    threshold = best_score * (1.0 - tie_fraction)
    eligible = [p for p in peaks if p.score >= threshold]
    return min(eligible, key=lambda p: ((p.x - center[0]) ** 2 +
                                        (p.y - center[1]) ** 2, -p.score, p.y, p.x))


class LocalizationEngine:
    """Single official implementation used by inference, demo, and benchmark."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config

    def predict_arrays(self, reference: np.ndarray, search: np.ndarray) -> LocalizationResult:
        start = time.perf_counter()
        cfg = self.config
        geom = cfg.geometry
        ref = np.asarray(reference, dtype=np.float32)
        sea = np.asarray(search, dtype=np.float32)
        if ref.ndim != 2 or sea.ndim != 2:
            raise ValueError("reference and search must be 2-D grayscale arrays")
        if ref.shape != (geom.reference_size, geom.reference_size):
            raise ValueError(f"reference must have shape {(geom.reference_size, geom.reference_size)}, got {ref.shape}")
        if sea.shape != (geom.search_size, geom.search_size):
            raise ValueError(f"search must have shape {(geom.search_size, geom.search_size)}, got {sea.shape}")
        if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(sea)):
            raise ValueError("reference and search must contain finite values")

        template = area_downsample(ref, geom.scale_factor)
        coarse_search = area_downsample(sea, cfg.coarse_factor)
        coarse_template = area_downsample(template, cfg.coarse_factor)
        coarse_plan = ZnccPlan(coarse_search, cfg.variance_epsilon)
        coarse_records: List[Dict[str, Any]] = []
        coarse_peaks: List[Peak] = []
        best_angle, best_scale = 0.0, 1.0
        best_score = -float("inf")
        for angle, scale in _candidate_grid(cfg):
            score_map = coarse_plan.match(rotate_scale(coarse_template, angle, scale))
            peaks = find_peaks(score_map, cfg.score_relative_threshold,
                               cfg.nms_radius_px / cfg.coarse_factor, max_peaks=8)
            if not peaks:
                y, x = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
                peaks = [Peak(float(score_map[y, x]), int(x), int(y))]
            coarse_peaks.extend(peaks)
            top = peaks[0]
            coarse_records.append({"angle_deg": angle, "scale": scale,
                                   "score": top.score, "x": top.x, "y": top.y})
            if top.score > best_score:
                best_score, best_angle, best_scale = top.score, angle, scale

        coarse_best = _best_peak(coarse_peaks,
                                  (cfg.center_prior[0] / cfg.coarse_factor,
                                   cfg.center_prior[1] / cfg.coarse_factor),
                                  cfg.tie_break_fraction)
        coarse_center = center_from_top_left(
            coarse_best.x * cfg.coarse_factor,
            coarse_best.y * cfg.coarse_factor,
            template.shape[0])
        center_fallback = False
        if hypot(coarse_center[0] - cfg.center_prior[0],
                 coarse_center[1] - cfg.center_prior[1]) > cfg.coarse_roi_radius * 2:
            # The challenge prior is allowed to constrain the coarse ROI. A
            # weak, remote peak is treated as unreliable instead of allowing
            # the fine stage to spend its entire budget on a remote background
            # tile. This is explicitly tied to the challenge center guarantee.
            coarse_center = cfg.center_prior
            center_fallback = True

        radius = cfg.coarse_roi_radius
        x0 = int(np.clip(round(coarse_center[0] - radius), 0,
                         sea.shape[1] - template.shape[1]))
        y0 = int(np.clip(round(coarse_center[1] - radius), 0,
                         sea.shape[0] - template.shape[0]))
        x1 = min(sea.shape[1], x0 + 2 * radius)
        y1 = min(sea.shape[0], y0 + 2 * radius)
        roi = sea[y0:y1, x0:x1]
        fine_plan = ZnccPlan(roi, cfg.variance_epsilon)
        fine_peaks: List[Peak] = []
        fine_records: List[Dict[str, Any]] = []
        maps: Dict[Tuple[float, float], np.ndarray] = {}
        for angle, scale in _neighborhood(best_angle, best_scale, cfg):
            score_map = fine_plan.match(rotate_scale(template, angle, scale))
            maps[(angle, scale)] = score_map
            peaks = find_peaks(score_map, cfg.score_relative_threshold,
                               cfg.nms_radius_px, max_peaks=32)
            if not peaks:
                y, x = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
                peaks = [Peak(float(score_map[y, x]), int(x), int(y))]
            for peak in peaks:
                fine_peaks.append(Peak(peak.score, peak.x + x0, peak.y + y0))
            top = peaks[0]
            fine_records.append({"angle_deg": angle, "scale": scale,
                                 "score": top.score, "x": top.x + x0,
                                 "y": top.y + y0})

        fine_best = _best_peak(fine_peaks, cfg.center_prior, cfg.tie_break_fraction)
        prior_selection = False
        if fine_best.score < 0.35 and fine_peaks:
            # At weak scores, the challenge prior is more informative than a
            # small correlation advantage from a background fluctuation.
            fine_best = min(fine_peaks,
                            key=lambda p: ((p.x + 49.5 - cfg.center_prior[0]) ** 2 +
                                           (p.y + 49.5 - cfg.center_prior[1]) ** 2,
                                           -p.score))
            prior_selection = True
        if prior_selection:
            pred_x, pred_y = center_from_top_left(fine_best.x, fine_best.y,
                                                  template.shape[0])
            winning_record = min(fine_records,
                                 key=lambda r: (abs(r["x"] - fine_best.x) +
                                                abs(r["y"] - fine_best.y), -r["score"]))
        else:
            winning_record = min(fine_records,
                                 key=lambda r: (-abs(r["score"] - fine_best.score),
                                                 r["angle_deg"], r["scale"]))
            winning_map = maps[(winning_record["angle_deg"], winning_record["scale"])]
            local_x, local_y = subpixel_refine(winning_map,
                                               fine_best.x - x0, fine_best.y - y0)
            pred_x, pred_y = center_from_top_left(x0 + local_x, y0 + local_y,
                                                  template.shape[0])

        report = analyze_peaks(fine_peaks, cfg.center_prior,
                               cfg.ambiguity_margin,
                               cfg.ambiguity_min_separation_px)
        confidence = confidence_from_evidence(
            report, report.center_distance_px, cfg.ood_center_radius_px)
        status = classify_status(report, confidence, report.center_distance_px,
                                 cfg.ood_center_radius_px, cfg.low_confidence_score)
        latency = (time.perf_counter() - start) * 1000.0
        diagnostics = {
            "official_path": "fft_zncc_geometric_search_subpixel",
            "confidence_kind": "heuristic_not_calibrated_probability",
            "template_shape": list(template.shape),
            "coarse_peak": {"x": coarse_best.x, "y": coarse_best.y,
                            "score": coarse_best.score},
            "selected_transform": {"angle_deg": winning_record["angle_deg"],
                                    "scale": winning_record["scale"]},
            "center_prior_fallback": center_fallback,
            "weak_score_prior_selection": prior_selection,
            "coarse_candidates": coarse_records,
            "fine_candidates": fine_records,
            "ambiguity": report.__dict__,
            "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        }
        return LocalizationResult(float(pred_x), float(pred_y),
                                  float(fine_best.score), float(confidence),
                                  status, float(latency), diagnostics)
