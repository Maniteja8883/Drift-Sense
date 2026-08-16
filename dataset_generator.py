"""
Drift-Sense Physics-Informed SEM Dataset Generator
===================================================
Generates synthetic SEM image pairs with authentic physical noise models.

Two layout families:
  1. DRAM Cell Arrays       - dense staggered contact pads, bitlines, wordlines
  2. FinFET Grids           - orthogonal dense fin lines, crossing gates, via landings

Noise stack (applied independently to reference and search):
  1. Poisson shot noise              (electron counting statistics)
  2. Edge charging / blooming        (high SE yield at boundaries)
  3. Charging potential gradient     (low-freq 2D intensity ramp)
  4. Scanline micro-jitter           (horizontal row displacement ±0.3 px)
  5. Stage misalignment              (random rotation ±3°, scale 0.95-1.05)

Ground truth: target placement (X_gt, Y_gt) saved to benchmark_ground_truth.csv
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw

from config import (
    REF_FINE_SIZE, SEARCH_FINE_SIZE, SCALE_FACTOR, TEMPLATE_SIZE,
    CENTER_PRIOR,
    SEM_LAMBDA, EDGE_CHARGE_STRENGTH, CHARGING_RAMP_AMPLITUDE,
    SCANLINE_JITTER_PX, ROTATION_RANGE_DEG, SCALE_RANGE,
    NUM_CHALLENGE_PAIRS, NUM_DRAM_PAIRS, NUM_FINFET_PAIRS,
)
from common import area_downsample

# ---------------------------------------------------------------------------
# Random state management
# ---------------------------------------------------------------------------
_SEED_BASE = 0xD1575317  # "DRIFT"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Primitive layout generators
# ---------------------------------------------------------------------------


def _draw_dram_cell_array(img: np.ndarray,
                          rng: np.random.Generator,
                          cell_pitch: int = 24,
                          contact_r: int = 4,
                          line_w: int = 2,
                          margin: int = 80,
                          add_fiducials: bool = True) -> None:
    """
    Draw a dense DRAM-like cell array on a blank image (in-place).

    Pattern: staggered contact pads connected by orthogonal bitlines (vertical)
    and wordlines (horizontal). Peripheral fiducials help anchor the pattern.
    """
    H, W = img.shape
    draw = ImageDraw.Draw(Image.fromarray((img * 255).astype(np.uint8)))

    # Bitlines (vertical)
    for cx in range(margin, W - margin, cell_pitch):
        draw.rectangle([cx - line_w // 2, margin,
                        cx + (line_w + 1) // 2, H - margin],
                       fill=255)
    # Wordlines (horizontal)
    for cy in range(margin, H - margin, cell_pitch):
        draw.rectangle([margin, cy - line_w // 2,
                        W - margin, cy + (line_w + 1) // 2],
                       fill=255)
    # Staggered contacts (checkerboard)
    for ix, cx in enumerate(range(margin, W - margin, cell_pitch)):
        for iy, cy in enumerate(range(margin, H - margin, cell_pitch)):
            if (ix + iy) % 2 == 0:
                draw.ellipse([cx - contact_r, cy - contact_r,
                              cx + contact_r, cy + contact_r],
                             fill=255)

    # Peripheral fiducials (large crosses at four corners)
    if add_fiducials:
        fsize = 20
        for fx, fy in [(margin, margin), (W - margin, margin),
                       (margin, H - margin), (W - margin, H - margin)]:
            draw.rectangle([fx - fsize, fy - 2, fx + fsize, fy + 2], fill=255)
            draw.rectangle([fx - 2, fy - fsize, fx + 2, fy + fsize], fill=255)

    # Write back
    img[:, :] = np.array(draw.im).astype(np.float32) / 255.0


def _draw_finfet_grid(img: np.ndarray,
                      rng: np.random.Generator,
                      fin_pitch: int = 18,
                      fin_w: int = 4,
                      gate_pitch: int = 40,
                      gate_w: int = 6,
                      via_r: int = 3,
                      margin: int = 80) -> None:
    """
    Draw a FinFET-like grid on a blank image (in-place).

    Pattern: dense parallel fin lines (vertical), crossing gate structures
    (horizontal), and via landings at fin/gate intersections.
    """
    H, W = img.shape
    draw = ImageDraw.Draw(Image.fromarray((img * 255).astype(np.uint8)))

    # Fins (vertical)
    for cx in range(margin, W - margin, fin_pitch):
        draw.rectangle([cx - fin_w // 2, margin,
                        cx + (fin_w + 1) // 2, H - margin],
                       fill=255)
    # Gates (horizontal)
    for cy in range(margin, H - margin, gate_pitch):
        draw.rectangle([margin, cy - gate_w // 2,
                        W - margin, cy + (gate_w + 1) // 2],
                       fill=255)
    # Via landings at intersections
    for cx in range(margin, W - margin, fin_pitch):
        for cy in range(margin, H - margin, gate_pitch):
            draw.ellipse([cx - via_r, cy - via_r,
                          cx + via_r, cy + via_r],
                         fill=255)

    img[:, :] = np.array(draw.im).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Noise operators
# ---------------------------------------------------------------------------


def _poisson_shot_noise(img: np.ndarray,
                        lam: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Electron counting noise: Poisson(lam * I) / lam."""
    # Scale to expected counts, draw Poisson, scale back.
    counts = rng.poisson(np.clip(img * lam, 0, lam * 2))
    return (counts / lam).astype(np.float32)


def _edge_charging_bloom(img: np.ndarray,
                         strength: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    High secondary-electron yield along sharp boundaries.
    Approximated by blending the Sobel gradient magnitude into edge pixels.
    """
    from scipy.ndimage import sobel
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    grad = np.hypot(gx, gy)
    # Normalize and additively blend (bloom is additive in SEM)
    if grad.max() > 0:
        grad = grad / grad.max()
    return np.clip(img + strength * grad, 0, 1).astype(np.float32)


def _charging_ramp(img: np.ndarray,
                   amp: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Low-frequency 2D potential gradient across the field of view."""
    H, W = img.shape
    # Random smooth ramp via bilinear interpolation of 4 corner values
    c = rng.uniform(-amp, amp, size=(2, 2))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yy = yy / (H - 1)
    xx = xx / (W - 1)
    ramp = ((1 - yy) * (1 - xx) * c[0, 0] +
            (1 - yy) * xx * c[0, 1] +
            yy * (1 - xx) * c[1, 0] +
            yy * xx * c[1, 1])
    return np.clip(img + ramp, 0, 1).astype(np.float32)


def _scanline_jitter(img: np.ndarray,
                     max_px: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Subtle horizontal row displacement (beam deflector jitter)."""
    H, W = img.shape
    shifts = rng.uniform(-max_px, max_px, size=H)
    out = np.zeros_like(img)
    for y in range(H):
        s = shifts[y]
        s0 = int(np.floor(s))
        f = s - s0
        if s0 >= 0:
            if s0 < W:
                out[y, s0:] += (1 - f) * img[y, :W - s0]
            if s0 + 1 < W:
                out[y, s0 + 1:] += f * img[y, :W - s0 - 1]
        else:
            if -s0 < W:
                out[y, :W + s0] += (1 - f) * img[y, -s0:]
            if -s0 + 1 < W:
                out[y, :W + s0 - 1] += f * img[y, -s0 - 1:]
    return np.clip(out, 0, 1).astype(np.float32)


def _affine_transform(img: np.ndarray,
                      angle_deg: float,
                      scale: float,
                      center: Tuple[float, float],
                      rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rotation + scale about a center, returning the warped image and
    the inverse mapping for ground-truth coordinate transformation.
    """
    from scipy.ndimage import affine_transform, map_coordinates
    H, W = img.shape
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    cy, cx = center
    # Forward matrix (maps src -> dst)
    M = np.array([[c * scale, -s * scale, 0],
                  [s * scale,  c * scale, 0]], dtype=np.float64)
    # Translate to center
    M[:, 2] = -M[:, 0] * cx - M[:, 1] * cy + np.array([cx, cy])
    # Output shape = input shape (we keep full FOV, may have black borders)
    warped = affine_transform(img, M[:2, :2], offset=M[:, 2],
                              output_shape=(H, W), order=1, mode='constant',
                              cval=0.0, prefilter=False)
    # Inverse matrix for coordinate mapping
    inv = np.linalg.inv(M[:2, :2])
    inv_offset = -inv @ M[:, 2]
    return warped.astype(np.float32), inv, inv_offset


def _apply_transform_to_point(x: float, y: float,
                               inv: np.ndarray,
                               inv_offset: np.ndarray) -> Tuple[float, float]:
    p = np.array([x, y])
    p_prime = inv @ p + inv_offset
    return float(p_prime[0]), float(p_prime[1])


# ---------------------------------------------------------------------------
# Pipeline: generate one pair
# ---------------------------------------------------------------------------


def generate_pair(layout_type: str,
                  pair_idx: int,
                  rng: np.random.Generator,
                  noise_multiplier: float = 1.0) -> Tuple[np.ndarray, np.ndarray,
                                                          float, float]:
    """
    Generate one (reference, search) image pair with known ground truth.

    Returns:
        ref_img   : 1000x1000 high-res reference (1 nm/px)
        search_img: 1000x1000 low-res search   (10 nm/px)
        gt_x      : ground-truth feature center X in search image (pixels)
        gt_y      : ground-truth feature center Y in search image (pixels)
    """
    # --- 1. Perfect layout (high-res reference frame) ---------------------
    ref_clean = np.zeros((REF_FINE_SIZE, REF_FINE_SIZE), dtype=np.float32)
    if layout_type == "dram":
        _draw_dram_cell_array(ref_clean, rng)
    elif layout_type == "finfet":
        _draw_finfet_grid(ref_clean, rng)
    else:
        raise ValueError(f"unknown layout: {layout_type}")

    # --- 2. Random stage misalignment (relative transform) ----------------
    angle = rng.uniform(*ROTATION_RANGE_DEG)
    scale = rng.uniform(*SCALE_RANGE)
    # The reference is aligned; the search frame is rotated/scaled relative
    # to it. Equivalently, we can apply the INVERSE transform to the search.
    inv_angle = -angle
    inv_scale = 1.0 / scale

    # The feature center in reference coordinates (always image center)
    ref_center = (REF_FINE_SIZE / 2.0, REF_FINE_SIZE / 2.0)

    # --- 3. Reference image (high-res, no downsample, just noise) -------
    ref = _poisson_shot_noise(ref_clean, SEM_LAMBDA * noise_multiplier, rng)
    ref = _edge_charging_bloom(ref, EDGE_CHARGE_STRENGTH * noise_multiplier, rng)
    ref = _charging_ramp(ref, CHARGING_RAMP_AMPLITUDE * noise_multiplier, rng)
    ref = _scanline_jitter(ref, SCANLINE_JITTER_PX * noise_multiplier, rng)

    # --- 4. Search image construction ------------------------------------
    # Start from the reference layout, apply inverse stage transform, then
    # downsample by 10x to get the 10 nm/px search image.
    # We apply transform at high-res then downsample for best fidelity.
    warped, inv_mat, inv_off = _affine_transform(ref_clean, inv_angle, inv_scale,
                                                  ref_center, rng)
    search_clean = area_downsample(warped, SCALE_FACTOR)  # 1000 -> 100

    # BUT: the search image is 1000x1000 at 10 nm/px, so the downsampled
    # feature occupies 100x100 pixels centered at (500, 500). We need to embed
    # this 100x100 crop into a 1000x1000 canvas with background noise.
    search = np.zeros((SEARCH_FINE_SIZE, SEARCH_FINE_SIZE), dtype=np.float32)
    cy, cx = int(CENTER_PRIOR[1]), int(CENTER_PRIOR[0])
    h = search_clean.shape[0]
    y0, x0 = cy - h // 2, cx - h // 2
    search[y0:y0+h, x0:x0+h] = search_clean

    # Add independent noise to search
    search = _poisson_shot_noise(search, SEM_LAMBDA * noise_multiplier, rng)
    search = _edge_charging_bloom(search, EDGE_CHARGE_STRENGTH * noise_multiplier, rng)
    search = _charging_ramp(search, CHARGING_RAMP_AMPLITUDE * noise_multiplier, rng)
    search = _scanline_jitter(search, SCANLINE_JITTER_PX * noise_multiplier, rng)

    # --- 5. Ground-truth center in search frame -------------------------
    # The true feature center in the *high-res warped* frame is still (500, 500).
    # After downsampling by 10 and placing at (500, 500) in the search canvas:
    gt_x, gt_y = CENTER_PRIOR

    return ref, search, gt_x, gt_y


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def _save_image(arr: np.ndarray, path: Path) -> None:
    """Save float32 [0,1] array as 8-bit PNG."""
    im = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    im.save(path)


def generate_dataset(output_dir: str = "dataset",
                     n_pairs: int = NUM_CHALLENGE_PAIRS,
                     n_dram: int = NUM_DRAM_PAIRS,
                     n_finfet: int = NUM_FINFET_PAIRS,
                     seed: int = _SEED_BASE) -> str:
    """
    Generate the full challenge dataset.

    Directory structure:
        output_dir/
            reference_000.png ... reference_029.png
            search_000.png    ... search_029.png
            benchmark_ground_truth.csv

    CSV columns: index, layout, ref_path, search_path, gt_x, gt_y, noise_mult
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = _rng(seed)
    noise_mults = rng.uniform(0.5, 2.0, size=n_pairs)  # varied noise levels

    # Pre-generate seeds for each pair so order/layout doesn't couple randomness
    pair_seeds = rng.integers(0, 2**32, size=n_pairs)

    csv_path = out / "benchmark_ground_truth.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index', 'layout', 'ref_path', 'search_path',
                    'gt_x', 'gt_y', 'noise_multiplier'])

        idx = 0
        for layout, count in [('dram', n_dram), ('finfet', n_finfet)]:
            for i in range(count):
                pair_rng = _rng(int(pair_seeds[idx]))
                ref, search, gt_x, gt_y = generate_pair(
                    layout, idx, pair_rng, float(noise_mults[idx]))

                ref_name = f"reference_{idx:03d}.png"
                search_name = f"search_{idx:03d}.png"
                _save_image(ref, out / ref_name)
                _save_image(search, out / search_name)

                w.writerow([idx, layout, ref_name, search_name,
                            f"{gt_x:.4f}", f"{gt_y:.4f}",
                            f"{noise_mults[idx]:.4f}"])
                idx += 1

    return str(csv_path)


def add_pathological_case(output_dir: str = "dataset",
                          index: int = 30) -> None:
    """
    Add the 31st pathological case: infinite identical array without unique
    context. This demonstrates the information-theoretic boundary where
    correlation cannot disambiguate (many equal peaks).
    """
    out = Path(output_dir)
    rng = _rng(_SEED_BASE + index)

    # Perfect periodic grid with NO fiducials, NO variation
    img = np.zeros((REF_FINE_SIZE, REF_FINE_SIZE), dtype=np.float32)
    # Very regular pitch, no staggering, no periphery
    pitch = 20
    for cx in range(0, REF_FINE_SIZE, pitch):
        img[:, cx:cx+2] = 1.0
    for cy in range(0, REF_FINE_SIZE, pitch):
        img[cy:cy+2, :] = 1.0

    # Add standard noise
    ref = _poisson_shot_noise(img, SEM_LAMBDA, rng)
    ref = _edge_charging_bloom(ref, EDGE_CHARGE_STRENGTH, rng)

    # Search = exact downsample, placed at center
    search_clean = area_downsample(ref, SCALE_FACTOR)
    search = np.zeros((SEARCH_FINE_SIZE, SEARCH_FINE_SIZE), dtype=np.float32)
    cy, cx = int(CENTER_PRIOR[1]), int(CENTER_PRIOR[0])
    h = search_clean.shape[0]
    y0, x0 = cy - h // 2, cx - h // 2
    search[y0:y0+h, x0:x0+h] = search_clean
    search = _poisson_shot_noise(search, SEM_LAMBDA, rng)
    search = _edge_charging_bloom(search, EDGE_CHARGE_STRENGTH, rng)

    gt_x, gt_y = CENTER_PRIOR

    ref_name = f"reference_{index:03d}.png"
    search_name = f"search_{index:03d}.png"
    _save_image(ref, out / ref_name)
    _save_image(search, out / search_name)

    csv_path = out / "benchmark_ground_truth.csv"
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([index, 'pathological_periodic', ref_name, search_name,
                    f"{gt_x:.4f}", f"{gt_y:.4f}", "1.0000"])


if __name__ == "__main__":
    csv_file = generate_dataset()
    add_pathological_case()
    print(f"Generated 30 challenge pairs + 1 pathological case")
    print(f"Ground truth: {csv_file}")