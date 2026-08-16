"""
Drift-Sense Numerical Engine
=============================
Low-level, dependency-light numerical primitives shared across the pipeline.

Every routine is implemented in pure NumPy (float32/float64 aware) so the
package runs hermetically on any Python 3.9+ environment with only NumPy and
Pillow installed.

Core algorithm summary
----------------------
FFT-based Zero-Mean Normalized Cross-Correlation (ZNCC):

    ZNCC(u, v) = sum_{i,j} T'(i,j) * S'(u+i, v+j)
                 --------------------------------------------------
                 sqrt( sum T'^2  *  sum_{(u..u+h-1, v..v+w-1)} S'^2 )

where T' = T - mean(T), S' = S - local_mean(S).

  * Numerator  : FFT cross-correlation  (single rfft2 pair, reused for all
                 template candidates).
  * Denominator: local sums of S and S^2 obtained in O(1) per pixel from
                 integral images (cumulative sums), so the variance map never
                 becomes a bottleneck.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Image IO / color conversion
# ---------------------------------------------------------------------------


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to float32 grayscale using Rec.601 luminance weights.

    Auto-detects shape:
      * 2D input            -> returned unchanged (as float32)
      * (H, W, 3)  RGB      -> 0.299 R + 0.587 G + 0.114 B
      * (H, W, 4)  RGBA     -> luminance with alpha ignored
      * (H, W, 1)           -> squeezed to 2D

    Accepts uint8, uint16, float inputs. Output is normalized to [0, 1].
    """
    if img is None:
        raise ValueError("rgb_to_gray: received None image")
    arr = np.asarray(img)
    if arr.ndim == 2:
        gray = arr.astype(np.float32, copy=False)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            gray = arr[..., 0].astype(np.float32, copy=False)
        elif arr.shape[2] == 3:
            gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] +
                    0.114 * arr[..., 2]).astype(np.float32)
        elif arr.shape[2] == 4:
            gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] +
                    0.114 * arr[..., 2]).astype(np.float32)
        else:
            raise ValueError(f"rgb_to_gray: unsupported channel count "
                             f"{arr.shape[2]}")
    else:
        raise ValueError(f"rgb_to_gray: expected 2D or 3D array, got "
                         f"ndim={arr.ndim}")

    if arr.dtype == np.uint8:
        gray = gray / 255.0
    elif arr.dtype == np.uint16:
        gray = gray / 65535.0
    return gray.astype(np.float32)


def load_gray(path: str) -> np.ndarray:
    """Load an image file as a float32 grayscale array in [0, 1]."""
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im)
    return rgb_to_gray(arr)


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------


def area_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """
    Robust block-averaging downsample via reshape/mean in float32.

    If a dimension is not divisible by `factor`, the trailing remainder is
    trimmed first (this is a pristine low-pass -- no aliasing is introduced).

    Example: area_downsample(1000x1000, 10) -> 100x100 (exact).
    """
    if factor <= 1:
        return np.asarray(img, dtype=np.float32)
    arr = np.asarray(img, dtype=np.float32)
    H, W = arr.shape
    h_crop, w_crop = H - (H % factor), W - (W % factor)
    arr = arr[:h_crop, :w_crop]
    sh = (h_crop // factor, factor, w_crop // factor, factor)
    return arr.reshape(sh).mean(axis=(1, 3)).astype(np.float32)


def fast_rotate_scale(template: np.ndarray,
                      angle_deg: float,
                      scale: float) -> np.ndarray:
    """
    Rotate (about center) and scale a small template using vectorized
    bilinear interpolation -- pure NumPy, no OpenCV required.

    The output has the same dimensions as the input. The inverse mapping is

        src = c + R(-theta) @ (dst - c) / s

    so `angle_deg > 0` rotates the feature clockwise in the output (the SEM
    stage is rotated w.r.t. the search frame). Edge samples are clamped.
    """
    arr = np.asarray(template, dtype=np.float32)
    H, W = arr.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    if scale <= 0.0:
        raise ValueError("fast_rotate_scale: scale must be > 0")

    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    # Inverse rotation matrix R(-theta)
    a00, a01 = c, s
    a10, a11 = -s, c
    inv = 1.0 / scale

    # Output coordinate grid (column-major to match row/col layout).
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Map dst -> src (centered coordinates).
    tx = (xx - cx) * inv
    ty = (yy - cy) * inv
    sx = a00 * tx + a01 * ty + cx
    sy = a10 * tx + a11 * ty + cy

    # Bilinear sampling.
    sx0 = np.floor(sx).astype(np.int64)
    sy0 = np.floor(sy).astype(np.int64)
    fx = sx - sx0
    fy = sy - sy0
    sx0 = np.clip(sx0, 0, W - 1)
    sy0 = np.clip(sy0, 0, H - 1)
    sx1 = np.clip(sx0 + 1, 0, W - 1)
    sy1 = np.clip(sy0 + 1, 0, H - 1)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    out = (w00 * arr[sy0, sx0] + w10 * arr[sy0, sx1] +
           w01 * arr[sy1, sx0] + w11 * arr[sy1, sx1])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Integral images (O(1) window statistics)
# ---------------------------------------------------------------------------


def _integral(a: np.ndarray) -> np.ndarray:
    """Zero-padded integral image: out[i+1, j+1] = sum a[0:i+1, 0:j+1]."""
    acc = np.cumsum(np.cumsum(a.astype(np.float64), axis=0), axis=1)
    p = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    p[1:, 1:] = acc
    return p


def _window_sums(integral: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    For every top-left corner (u, v) in [0, H-h] x [0, W-w], return the sum
    of the h x w window. Shape: (H-h+1, W-w+1).
    """
    p = integral
    bottom = p[h:, w:]          # u+h, v+w
    right = p[:-h, w:] if h > 0 else p[:, w:]      # u, v+w
    left = p[h:, :-w] if w > 0 else p[h:, :]       # u+h, v
    top = p[:-h, :-w] if (h > 0 and w > 0) else p[:-h, :-w]
    # Shapes: bottom (H-h+1, W-w+1), right (H-h+1, W-w+1),
    #         left (H-h+1, W-w+1), top (H-h+1, W-w+1)
    return bottom - right - left + top


# ---------------------------------------------------------------------------
# FFT-ZNCC core
# ---------------------------------------------------------------------------


def fft_zncc(search_img: np.ndarray,
             template: np.ndarray,
             variance_eps: float = 1e-6) -> np.ndarray:
    """
    Full zero-mean normalized cross-correlation (valid mode).

    Returns a float32 score map of shape (H_s - H_t + 1, W_s - W_t + 1) with
    values in roughly [-1, 1]. Position (u, v) of the map corresponds to the
    template's top-left corner placed at (u, v) in the search image.

    The search image's FFT and integral images are computed inside this call;
    for multi-candidate matching prefer `ZnccPlan` below, which precomputes
    them once and reuses them.
    """
    return ZnccPlan(search_img, variance_eps).match(template)


class ZnccPlan:
    """
    Reusable ZNCC engine for one search image.

    Precomputes once:
      * rfft2 of the search image (at padded size for valid-mode correlation)
      * integral images of S and S^2

    then every call to `match(template)` only pays for one small template FFT,
    one inverse FFT and vectorized array arithmetic.
    """

    def __init__(self, search_img: np.ndarray, variance_eps: float = 1e-6):
        self.search = np.asarray(search_img, dtype=np.float64)
        self.eps = float(variance_eps)
        H, W = self.search.shape
        self.H, self.W = H, W

        # ---- FFT of search image at padded size (valid correlation) ------
        # We evaluate the template at every placement where it fully overlaps
        # the search image -> score map (H-h+1, W-w+1). The padded FFT size
        # must be >= H+h-1 to avoid circular wrap.
        self._pad_h = None   # set lazily once template size is known
        self._pad_w = None
        self._fsearch = None
        self._iS = None      # integral of S
        self._iS2 = None     # integral of S^2

    def _prepare(self, t_h: int, t_w: int) -> None:
        if (self._pad_h == t_h and self._pad_w == t_w and
                self._fsearch is not None):
            return
        H, W = self.H, self.W
        pad_h = H + t_h - 1
        pad_w = W + t_w - 1
        self._pad_h, self._pad_w = t_h, t_w
        self._fsearch = np.fft.rfft2(self.search, s=(pad_h, pad_w))
        # Integral images in float64 for exact local statistics.
        self._iS = _integral(self.search)
        self._iS2 = _integral(self.search * self.search)

    def match(self, template: np.ndarray) -> np.ndarray:
        t = np.asarray(template, dtype=np.float64)
        if t.ndim != 2:
            raise ValueError("fft_zncc: template must be 2D")
        t_h, t_w = t.shape
        H, W = self.H, self.W
        if t_h > H or t_w > W:
            raise ValueError("fft_zncc: template larger than search image")

        self._prepare(t_h, t_w)

        # ---- Numerator: zero-mean template cross-correlation ------------
        t_mean = t.mean()
        t_zero = t - t_mean
        f_tpl = np.fft.rfft2(t_zero, s=(H + t_h - 1, W + t_w - 1))
        corr_full = np.fft.irfft2(self._fsearch * np.conj(f_tpl),
                                  s=(H + t_h - 1, W + t_w - 1))
        num = corr_full[0:H - t_h + 1, 0:W - t_w + 1]

        # ---- Denominator: local window statistics from integral images ---
        sum_s = _window_sums(self._iS, t_h, t_w)
        sum_s2 = _window_sums(self._iS2, t_h, t_w)
        n = t_h * t_w
        # var(S_window) = (sum_s2 - sum_s^2 / n) / n   (>= 0)
        var_window = (sum_s2 - (sum_s * sum_s) / n) / n
        # sum(T')^2  -- note mean removed, so sum T' = 0
        sum_t2 = np.sum((t_zero) ** 2)

        var_window = np.maximum(var_window, self.eps)
        denom = np.sqrt(sum_t2 * n * var_window)
        denom = np.maximum(denom, self.eps)

        with np.errstate(invalid='ignore', divide='ignore'):
            score = np.where(denom > 0, num / denom, 0.0)
        return np.clip(score, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Peak refinement
# ---------------------------------------------------------------------------


def _parabola_offset(fm1: float, f0: float, fp1: float) -> float:
    """Vertex offset from center of a parabola through 3 samples."""
    denom = fm1 - 2.0 * f0 + fp1
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (fm1 - fp1) / denom


def _one_sided_offset(f0: float, f1: float, f2: float) -> float:
    """Vertex offset from x=0 of parabola through samples at x=0,1,2."""
    denom = f0 - 2.0 * f1 + f2
    if abs(denom) < 1e-12:
        return 0.0
    # f(x) = a x^2 + b x + c; vertex = -b/(2a)
    return -(4.0 * f1 - f2 - 3.0 * f0) / (2.0 * denom)


def subpixel_refine_2d(score_map: np.ndarray,
                       peak_x: int,
                       peak_y: int) -> Tuple[float, float]:
    """
    Continuous 2D parabolic peak refinement on a correlation score map.

    Args:
        score_map: 2D array of correlation scores.
        peak_x:    integer column of the detected peak.
        peak_y:    integer row of the detected peak.

    Returns:
        (x, y) refined float coordinates of the peak.

    Interior peaks use the symmetric central-difference parabola fit; edge
    peaks fall back to a one-sided finite-difference parabola so the routine
    never fails at image boundaries.
    """
    H, W = score_map.shape
    px, py = int(peak_x), int(peak_y)

    if 1 <= px <= W - 2:
        dx = _parabola_offset(score_map[py, px - 1],
                              score_map[py, px],
                              score_map[py, px + 1])
    elif px == 0 and W >= 3:
        dx = _one_sided_offset(score_map[py, 0],
                               score_map[py, 1],
                               score_map[py, 2])
    elif px == W - 1 and W >= 3:
        dx = -_one_sided_offset(score_map[py, W - 1],
                                score_map[py, W - 2],
                                score_map[py, W - 3])
    else:
        dx = 0.0

    if 1 <= py <= H - 2:
        dy = _parabola_offset(score_map[py - 1, px],
                              score_map[py, px],
                              score_map[py + 1, px])
    elif py == 0 and H >= 3:
        dy = _one_sided_offset(score_map[0, px],
                               score_map[1, px],
                               score_map[2, px])
    elif py == H - 1 and H >= 3:
        dy = -_one_sided_offset(score_map[H - 1, px],
                                score_map[H - 2, px],
                                score_map[H - 3, px])
    else:
        dy = 0.0

    dx = float(np.clip(dx, -0.5, 0.5))
    dy = float(np.clip(dy, -0.5, 0.5))
    return px + dx, py + dy


# ---------------------------------------------------------------------------
# Peak detection / NMS
# ---------------------------------------------------------------------------


def find_peaks(score_map: np.ndarray,
               rel_threshold: float = 0.70,
               nms_radius: float = 20.0,
               max_peaks: int = 32) -> list:
    """
    Greedy non-maximum suppression over correlation peaks.

    Peaks are the local maxima of the score map whose value is at least
    `rel_threshold * global_max`. After ranking by score, each accepted peak
    suppresses any remaining candidate within `nms_radius` (Chebyshev radius
    for speed).

    Returns a list of dicts: {'score': float, 'x': int, 'y': int}.
    """
    sm = np.asarray(score_map, dtype=np.float64)
    global_max = float(sm.max())
    if global_max <= 0.0 or not np.isfinite(global_max):
        return []

    thr = rel_threshold * global_max
    H, W = sm.shape

    # Local maxima via 3x3 max filter (shift trick).
    interior = np.zeros_like(sm, dtype=bool)
    interior[1:-1, 1:-1] = (
        (sm[1:-1, 1:-1] >= sm[:-2, :-2]) & (sm[1:-1, 1:-1] >= sm[:-2, 1:-1]) &
        (sm[1:-1, 1:-1] >= sm[:-2, 2:]) & (sm[1:-1, 1:-1] >= sm[1:-1, :-2]) &
        (sm[1:-1, 1:-1] >= sm[1:-1, 2:]) & (sm[1:-1, 1:-1] >= sm[2:, :-2]) &
        (sm[1:-1, 1:-1] >= sm[2:, 1:-1]) & (sm[1:-1, 1:-1] >= sm[2:, 2:])
    )
    cand_mask = interior & (sm >= thr)
    ys, xs = np.nonzero(cand_mask)
    cands = sorted(zip(sm[ys, xs].tolist(), xs.tolist(), ys.tolist()),
                   key=lambda c: -c[0])

    picked = []
    for score, x, y in cands:
        if any(abs(x - px) <= nms_radius and abs(y - py) <= nms_radius
               for _, px, py in picked):
            continue
        picked.append((score, x, y))
        if len(picked) >= max_peaks:
            break
    return [{'score': float(s), 'x': int(x), 'y': int(y)}
            for s, x, y in picked]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def score_to_center(x: float, y: float, template_size: int = 100) -> Tuple[float, float]:
    """
    Convert a template top-left coordinate (x, y) in the search image to the
    feature center coordinate.

        X_center = x + (template_size - 1) / 2
        Y_center = y + (template_size - 1) / 2

    With template_size=100 this is exactly x + 49.5 (mathematically exact,
    no arbitrary cropping).
    """
    off = (template_size - 1) / 2.0
    return x + off, y + off
