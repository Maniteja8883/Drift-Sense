"""Trusted numerical matching primitives for the official baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


def _integral(array: np.ndarray) -> np.ndarray:
    acc = np.cumsum(np.cumsum(np.asarray(array, dtype=np.float64), axis=0), axis=1)
    out = np.zeros((array.shape[0] + 1, array.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = acc
    return out


def _window_sums(integral: np.ndarray, height: int, width: int) -> np.ndarray:
    return (integral[height:, width:] - integral[:-height, width:] -
            integral[height:, :-width] + integral[:-height, :-width])


def direct_zncc(search: np.ndarray, template: np.ndarray,
                variance_epsilon: float = 1e-12) -> np.ndarray:
    """Reference valid-mode ZNCC implementation for small-array verification."""
    s = np.asarray(search, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    if s.ndim != 2 or t.ndim != 2:
        raise ValueError("search and template must be 2-D")
    if t.shape[0] > s.shape[0] or t.shape[1] > s.shape[1]:
        raise ValueError("template must not be larger than search")
    tz = t - t.mean()
    t_energy = float(np.sum(tz * tz))
    out = np.zeros((s.shape[0] - t.shape[0] + 1,
                   s.shape[1] - t.shape[1] + 1), dtype=np.float32)
    if t_energy <= variance_epsilon:
        return out
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            window = s[y:y + t.shape[0], x:x + t.shape[1]]
            wz = window - window.mean()
            denom = np.sqrt(t_energy * float(np.sum(wz * wz)))
            if denom > variance_epsilon:
                out[y, x] = float(np.sum(tz * wz) / denom)
    return np.clip(out, -1.0, 1.0)


class ZnccPlan:
    """Reusable FFT-ZNCC plan for a fixed search image and template shape."""

    def __init__(self, search: np.ndarray, variance_epsilon: float = 1e-12):
        self.search = np.asarray(search, dtype=np.float64)
        if self.search.ndim != 2 or min(self.search.shape) < 1:
            raise ValueError("search must be a non-empty 2-D array")
        self.eps = float(variance_epsilon)
        self._shape = None
        self._fsearch = None
        self._sum = None
        self._sum2 = None

    def _prepare(self, template_shape) -> None:
        if self._shape == tuple(template_shape):
            return
        th, tw = template_shape
        if th > self.search.shape[0] or tw > self.search.shape[1]:
            raise ValueError("template must not be larger than search")
        pad = (self.search.shape[0] + th - 1, self.search.shape[1] + tw - 1)
        self._shape = (th, tw)
        self._fsearch = np.fft.rfft2(self.search, s=pad)
        self._sum = _integral(self.search)
        self._sum2 = _integral(self.search * self.search)

    def match(self, template: np.ndarray) -> np.ndarray:
        t = np.asarray(template, dtype=np.float64)
        if t.ndim != 2 or min(t.shape) < 1:
            raise ValueError("template must be a non-empty 2-D array")
        self._prepare(t.shape)
        th, tw = t.shape
        h, w = self.search.shape
        tz = t - t.mean()
        template_energy = float(np.sum(tz * tz))
        result_shape = (h - th + 1, w - tw + 1)
        if not np.isfinite(template_energy) or template_energy <= self.eps:
            return np.zeros(result_shape, dtype=np.float32)

        pad = (h + th - 1, w + tw - 1)
        ft = np.fft.rfft2(tz, s=pad)
        corr = np.fft.irfft2(self._fsearch * np.conj(ft), s=pad)
        numerator = corr[:result_shape[0], :result_shape[1]]
        sums = _window_sums(self._sum, th, tw)
        sums2 = _window_sums(self._sum2, th, tw)
        n = float(th * tw)
        window_energy = sums2 - sums * sums / n
        valid = np.isfinite(window_energy) & (window_energy > self.eps)
        denom = np.sqrt(np.maximum(window_energy, 0.0) * template_energy)
        scores = np.zeros_like(numerator, dtype=np.float64)
        scores[valid] = numerator[valid] / denom[valid]
        return np.clip(scores, -1.0, 1.0).astype(np.float32)


def fft_zncc(search: np.ndarray, template: np.ndarray,
             variance_epsilon: float = 1e-12) -> np.ndarray:
    return ZnccPlan(search, variance_epsilon).match(template)


@dataclass(frozen=True)
class Peak:
    score: float
    x: int
    y: int


def find_peaks(score_map: np.ndarray, relative_threshold: float = 0.70,
               nms_radius: float = 20.0, max_peaks: int = 32) -> List[Peak]:
    """Find local maxima and apply greedy Chebyshev-radius NMS."""
    scores = np.asarray(score_map, dtype=np.float64)
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError("score_map must be a non-empty 2-D array")
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    best = float(np.max(finite))
    if not np.isfinite(best):
        return []
    threshold = best * float(relative_threshold)
    candidates = []
    for y, x in zip(*np.nonzero(finite >= threshold)):
        y0, y1 = max(0, y - 1), min(scores.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(scores.shape[1], x + 2)
        if finite[y, x] >= np.max(finite[y0:y1, x0:x1]):
            candidates.append(Peak(float(finite[y, x]), int(x), int(y)))
    candidates.sort(key=lambda p: (-p.score, p.y, p.x))
    selected: List[Peak] = []
    for candidate in candidates:
        if any(abs(candidate.x - p.x) <= nms_radius and
               abs(candidate.y - p.y) <= nms_radius for p in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_peaks:
            break
    return selected

