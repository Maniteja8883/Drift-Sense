"""Deterministic physics-informed synthetic SEM-like benchmark generator.

The generator is an engineering approximation, not a validated electron/SEM
forward model. It is intentionally transparent so a benchmark can be rerun.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import BENCHMARK_SEED, BENCHMARK_VERSION
from .io import save_gray
from .preprocessing import area_downsample, rotate_scale


def _draw_layout(size: int, family: str, marker: bool = True) -> np.ndarray:
    """Create a simple DRAM/FinFET-like binary layout motif."""
    image = np.zeros((size, size), dtype=np.float32)
    if family == "dram":
        pitch = max(12, size // 42)
        for x in range(pitch, size - pitch, pitch):
            image[:, max(0, x - 1):min(size, x + 2)] = 0.8
        for y in range(pitch, size - pitch, pitch):
            image[max(0, y - 1):min(size, y + 2), :] = 0.55
        radius = max(2, size // 140)
        for y in range(pitch // 2, size, pitch * 2):
            for x in range(pitch // 2, size, pitch * 2):
                yy, xx = np.ogrid[:size, :size]
                image[(xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2] = 1.0
    elif family == "finfet":
        fin_pitch = max(16, size // 34)
        gate_pitch = max(22, size // 24)
        for x in range(fin_pitch, size - fin_pitch, fin_pitch):
            image[:, max(0, x - 1):min(size, x + 2)] = 0.9
        for y in range(gate_pitch, size - gate_pitch, gate_pitch):
            image[max(0, y - 2):min(size, y + 3), :] = 0.65
        for x in range(fin_pitch, size, fin_pitch):
            for y in range(gate_pitch, size, gate_pitch):
                image[max(0, y - 2):min(size, y + 3),
                      max(0, x - 2):min(size, x + 3)] = 1.0
    else:
        raise ValueError(f"unknown layout family: {family}")
    # A small site-specific fiducial makes ordinary pairs distinguishable from
    # the deliberately periodic adversarial partition.
    if marker:
        m = max(16, size // 12)
        image[m:2 * m, m:2 * m] = 1.0
        image[2 * m:3 * m, m:2 * m] = 0.25
    return image


def poisson_shot_noise(image: np.ndarray, electron_scale: float,
                       rng: np.random.Generator) -> np.ndarray:
    counts = rng.poisson(np.clip(image, 0.0, 1.0) * electron_scale)
    return (counts / max(electron_scale, 1.0)).astype(np.float32)


def edge_charging_approximation(image: np.ndarray, strength: float) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(image, dtype=np.float32))
    edge = np.hypot(gx, gy)
    if edge.max() > 0:
        edge = edge / edge.max()
    return np.clip(image + strength * edge, 0.0, 1.0).astype(np.float32)


def charging_gradient(image: np.ndarray, amplitude: float,
                      rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape
    corners = rng.uniform(-amplitude, amplitude, size=(2, 2))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= max(h - 1, 1)
    xx /= max(w - 1, 1)
    ramp = ((1 - yy) * (1 - xx) * corners[0, 0] + (1 - yy) * xx * corners[0, 1] +
            yy * (1 - xx) * corners[1, 0] + yy * xx * corners[1, 1])
    return np.clip(image + ramp, 0.0, 1.0).astype(np.float32)


def scanline_jitter(image: np.ndarray, max_shift_px: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Apply a row-wise subpixel horizontal shift using linear interpolation."""
    h, w = image.shape
    shifts = rng.uniform(-max_shift_px, max_shift_px, size=h)
    x = np.arange(w, dtype=np.float32)
    out = np.empty_like(image)
    for y, shift in enumerate(shifts):
        out[y] = np.interp(x - shift, x, image[y], left=image[y, 0], right=image[y, -1])
    return out.astype(np.float32)


def _apply_noise(image: np.ndarray, multiplier: float,
                 rng: np.random.Generator) -> np.ndarray:
    out = poisson_shot_noise(image, 100.0 * multiplier, rng)
    out = edge_charging_approximation(out, 0.15 * multiplier)
    out = charging_gradient(out, 0.05 * multiplier, rng)
    return scanline_jitter(out, 0.3 * multiplier, rng)


def _repeat_decoys(motif: np.ndarray, size: int, pitch: int,
                   rng: np.random.Generator, exact: bool) -> np.ndarray:
    canvas = np.full((size, size), 0.02, dtype=np.float32)
    h, w = motif.shape
    for y in range(20, size - h, pitch):
        for x in range(20, size - w, pitch):
            patch = motif if exact else np.clip(
                motif * rng.uniform(0.85, 1.10) + rng.normal(0, 0.01, motif.shape), 0, 1)
            canvas[y:y + h, x:x + w] = patch
    return canvas


def generate_pair(layout: str, rng: np.random.Generator,
                  noise_multiplier: float = 1.0,
                  target_center: Tuple[float, float] = (500.0, 500.0),
                  periodic: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Generate one pair and its search-frame target centre."""
    ref_clean = _draw_layout(1000, layout, marker=not periodic)
    ref = _apply_noise(ref_clean, noise_multiplier, rng)

    angle = float(rng.uniform(-3.0, 3.0))
    scale = float(rng.uniform(0.95, 1.05))
    motif = area_downsample(ref_clean, 10)
    decoy_base = area_downsample(_draw_layout(1000, layout, marker=False), 10)
    warped_motif = rotate_scale(motif, -angle, 1.0 / scale)
    warped_decoy = rotate_scale(decoy_base, -angle, 1.0 / scale)
    if periodic:
        # Exact 100-pixel tiling makes several placements mathematically
        # equivalent; the correct response is an ambiguity signal, not a
        # fabricated certainty.
        search = np.tile(warped_decoy, (10, 10)).astype(np.float32)
    else:
        # Ordinary benchmark pairs contain one site-specific target in a
        # weakly textured field; periodic decoys are reserved for the
        # explicitly labelled adversarial partition below.
        search = np.full((1000, 1000), 0.02, dtype=np.float32)
    cx, cy = map(int, target_center)
    x0 = int(np.clip(round(cx - warped_motif.shape[1] / 2), 0, 1000 - warped_motif.shape[1]))
    y0 = int(np.clip(round(cy - warped_motif.shape[0] / 2), 0, 1000 - warped_motif.shape[0]))
    search[y0:y0 + warped_motif.shape[0], x0:x0 + warped_motif.shape[1]] = warped_motif
    if not periodic:
        search = _apply_noise(search, noise_multiplier, rng)
    return ref, search, float(x0 + (warped_motif.shape[1] - 1) / 2.0), float(y0 + (warped_motif.shape[0] - 1) / 2.0)


def generate_pathological_pair(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Generate an exact repeated motif case for ambiguity testing."""
    return generate_pair("dram", rng, 0.15, (500.0, 500.0), periodic=True)


def generate_dataset(output_dir: str = "dataset", n_pairs: int = 30,
                     seed: int = BENCHMARK_SEED,
                     include_pathological: bool = True,
                     architecture: str = "both") -> str:
    """Generate benchmark PNGs and a manifest; return the manifest path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pair_seeds = rng.integers(0, 2 ** 32, size=n_pairs + 1, dtype=np.uint64)
    noise = rng.uniform(0.25, 0.85, size=n_pairs)
    architecture_key = architecture.strip().lower()
    if architecture_key not in {"dram", "finfet", "both"}:
        raise ValueError("architecture must be DRAM, FinFET, or both")
    rows: List[Dict[str, object]] = []
    for index in range(n_pairs):
        layout = ("dram" if architecture_key == "dram" else
                  "finfet" if architecture_key == "finfet" else
                  "dram" if index < n_pairs // 2 else "finfet")
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        ref, search, gx, gy = generate_pair(layout, pair_rng, float(noise[index]))
        ref_name, search_name = f"reference_{index:03d}.png", f"search_{index:03d}.png"
        save_gray(ref, out / ref_name)
        save_gray(search, out / search_name)
        rows.append({"index": index, "partition": "in_distribution",
                     "architecture": layout.upper() if layout == "dram" else "FinFET",
                     "layout": layout, "reference": ref_name,
                     "search": search_name, "ref_path": ref_name,
                     "search_path": search_name, "gt_x": gx, "gt_y": gy,
                     "noise_multiplier": float(noise[index])})
    if include_pathological:
        index = n_pairs
        ref, search, gx, gy = generate_pathological_pair(
            np.random.default_rng(int(pair_seeds[-1])))
        ref_name, search_name = f"reference_{index:03d}.png", f"search_{index:03d}.png"
        save_gray(ref, out / ref_name)
        save_gray(search, out / search_name)
        rows.append({"index": index, "partition": "adversarial_ambiguity",
                     "architecture": "DRAM", "layout": "periodic",
                     "reference": ref_name, "search": search_name,
                     "ref_path": ref_name, "search_path": search_name,
                     "gt_x": gx, "gt_y": gy,
                     "noise_multiplier": 0.15})
    manifest = out / "benchmark_ground_truth.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "manifest.csv").write_text(manifest.read_text())
    (out / "dataset_metadata.json").write_text(json.dumps({
        "dataset_version": BENCHMARK_VERSION,
        "seed": seed,
        "n_pairs": len(rows),
        "architecture": architecture_key,
        "partitions": {"in_distribution": n_pairs,
                       "adversarial_ambiguity": int(include_pathological)},
        "model_type": "physics-informed synthetic SEM corruption approximation",
    }, indent=2) + "\n")
    return str(manifest)
