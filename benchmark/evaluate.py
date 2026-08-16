"""Benchmark metrics and baseline/ablation evaluation."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from drift_sense.config import (BENCHMARK_VERSION, DEFAULT_CONFIG,
                                EVAL_THRESHOLDS_PX, PipelineConfig)
from drift_sense.geometry import center_from_top_left
from drift_sense.io import load_gray
from drift_sense.matching import ZnccPlan, find_peaks
from drift_sense.pipeline import LocalizationEngine
from drift_sense.preprocessing import area_downsample, rotate_scale


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def _git_worktree_dirty(root: Path) -> bool:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL)
        return bool(status.strip())
    except Exception:
        return True


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(np.max(arr))}


def _raw_zncc(reference: np.ndarray, search: np.ndarray,
              use_center_tie: bool = False) -> Tuple[float, float, float, str, Dict]:
    template = area_downsample(reference, 10)
    score = ZnccPlan(search).match(template)
    peaks = find_peaks(score, 0.70, 20.0, max_peaks=16)
    if not peaks:
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        peaks = [type("Peak", (), {"x": int(x), "y": int(y), "score": float(score[y, x])})()]
    if use_center_tie:
        best_score = max(p.score for p in peaks)
        eligible = [p for p in peaks if p.score >= best_score * 0.95]
        peak = min(eligible, key=lambda p: (p.x + 49.5 - 500.0) ** 2 +
                                         (p.y + 49.5 - 500.0) ** 2)
    else:
        peak = max(peaks, key=lambda p: p.score)
    x, y = center_from_top_left(peak.x, peak.y, template.shape[0])
    return x, y, float(peak.score), "SUCCESS", {"method": "single_scale_fft_zncc"}, None


def _coarse_geometric(reference: np.ndarray, search: np.ndarray) -> Tuple[float, float, float, str, Dict]:
    cfg = DEFAULT_CONFIG
    template = area_downsample(reference, 10)
    coarse_search = area_downsample(search, cfg.coarse_factor)
    coarse_template = area_downsample(template, cfg.coarse_factor)
    plan = ZnccPlan(coarse_search)
    candidates = []
    for angle in cfg.rotation_degrees:
        for scale in cfg.scale_candidates:
            sm = plan.match(rotate_scale(coarse_template, angle, scale))
            peaks = find_peaks(sm, 0.70, cfg.nms_radius_px / cfg.coarse_factor, 8)
            if peaks:
                p = peaks[0]
                candidates.append((p.score, p.x, p.y, angle, scale))
    candidates.sort(reverse=True)
    score, x, y, angle, scale = candidates[0]
    center = center_from_top_left(x * cfg.coarse_factor, y * cfg.coarse_factor,
                                  template.shape[0])
    return center[0], center[1], float(score), "SUCCESS", {
        "method": "fft_zncc_plus_geometric_search", "angle_deg": angle, "scale": scale}, None


def _run_method(name: str, reference: np.ndarray, search: np.ndarray):
    if name == "naive_single_scale":
        return _raw_zncc(reference, search, False)
    if name == "fft_zncc_center_tie":
        return _raw_zncc(reference, search, True)
    if name == "fft_zncc_geometric":
        return _coarse_geometric(reference, search)
    if name == "drift_sense_final":
        result = LocalizationEngine().predict_arrays(reference, search)
        return result.x, result.y, result.score, result.status, result.diagnostics, result.confidence
    raise ValueError(name)


def _metrics(records: List[Dict], include_partitions: Iterable[str]) -> Dict:
    selected = [r for r in records if r["partition"] in set(include_partitions)]
    valid = [r for r in selected if r.get("error_px") is not None]
    errors = np.asarray([r["error_px"] for r in valid], dtype=np.float64)
    latency = [float(r["latency_ms"]) for r in valid]
    result = {"n": len(valid), "accuracies": {
        f"acc@{thr:g}px": float(np.mean(errors <= thr) * 100.0) if len(errors) else 0.0
        for thr in EVAL_THRESHOLDS_PX},
        "median_error_px": float(np.median(errors)) if len(errors) else None,
        "mean_error_px": float(np.mean(errors)) if len(errors) else None,
        "p90_error_px": float(np.percentile(errors, 90)) if len(errors) else None,
        "p95_error_px": float(np.percentile(errors, 95)) if len(errors) else None,
        "failure_rate": float(np.mean([r.get("error_px") is None for r in selected])) if selected else 0.0,
        "status_counts": {status: sum(r.get("status") == status for r in selected)
                          for status in sorted({r.get("status") for r in selected})}}
    confidences = [float(r["confidence"]) for r in valid if r.get("confidence") is not None]
    result["confidence"] = {
        "mean": float(np.mean(confidences)) if confidences else None,
        "p10": float(np.percentile(confidences, 10)) if confidences else None,
    }
    result["latency"] = _percentiles(latency)
    return result


def evaluate_manifest(manifest: str, root: Path) -> Dict:
    rows = list(csv.DictReader(Path(manifest).open()))
    arrays = [(row, load_gray(Path(manifest).parent / row["ref_path"]),
               load_gray(Path(manifest).parent / row["search_path"])) for row in rows]
    methods = ["naive_single_scale", "fft_zncc_center_tie", "fft_zncc_geometric", "drift_sense_final"]
    all_results: Dict[str, object] = {}
    for method in methods:
        records = []
        for row, reference, search in arrays:
            start = time.perf_counter()
            try:
                x, y, score, status, diagnostics, confidence = _run_method(method, reference, search)
                error = float(np.hypot(x - float(row["gt_x"]), y - float(row["gt_y"])))
            except Exception as exc:
                x = y = score = None
                status = "FAILURE"
                diagnostics = {"exception": str(exc)}
                confidence = None
                error = None
            records.append({"index": int(row["index"]), "partition": row["partition"],
                            "layout": row["layout"], "pred_x": x, "pred_y": y,
                            "score": score, "status": status, "error_px": error,
                            "confidence": confidence,
                            "latency_ms": (time.perf_counter() - start) * 1000.0,
                            "diagnostics": diagnostics})
        all_results[method] = {
            "in_distribution": _metrics(records, ["in_distribution"]),
            "adversarial": _metrics(records, ["adversarial_ambiguity"]),
            "all": _metrics(records, ["in_distribution", "adversarial_ambiguity"]),
            "records": records,
        }

    # Measured ablations of the final decision path.
    rounded = []
    no_geometry = []
    for row, reference, search in arrays:
        result = LocalizationEngine().predict_arrays(reference, search)
        rounded.append({"partition": row["partition"], "error_px": float(np.hypot(round(result.x) - float(row["gt_x"]), round(result.y) - float(row["gt_y"]))),
                        "latency_ms": result.latency_ms})
        cfg = PipelineConfig(rotation_degrees=(0.0,), scale_candidates=(1.0,))
        simple = LocalizationEngine(cfg).predict_arrays(reference, search)
        no_geometry.append({"partition": row["partition"], "error_px": float(np.hypot(simple.x - float(row["gt_x"]), simple.y - float(row["gt_y"]))),
                            "latency_ms": simple.latency_ms})
    all_results["ablations"] = {
        "final_rounded_coordinates": {"in_distribution": _metrics(rounded, ["in_distribution"])},
        "no_rotation_scale_search": {"in_distribution": _metrics(no_geometry, ["in_distribution"])},
    }
    metadata = {
        "dataset_version": BENCHMARK_VERSION,
        "seed": _read_seed(Path(manifest).parent),
        "python": sys.version,
        "numpy": np.__version__,
        "pillow": _package_version("PIL"),
        "scipy": _package_version("scipy"),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "accelerator": "CPU; official path does not use GPU",
        "git_commit": _git_commit(root),
        "git_worktree_dirty_at_benchmark": _git_worktree_dirty(root),
        "latency_method": "time.perf_counter around in-memory prediction; image IO excluded",
        "accuracy_method": "Euclidean search-image-pixel error against manifest ground truth",
    }
    return {"metadata": metadata, "official_method": "drift_sense_final",
            "methods": all_results}


def _read_seed(dataset_dir: Path):
    try:
        return json.loads((dataset_dir / "dataset_metadata.json").read_text())["seed"]
    except Exception:
        return "unknown"


def _package_version(module: str) -> str:
    try:
        imported = __import__(module)
        return getattr(imported, "__version__", "installed")
    except Exception:
        return "not-installed"


def write_artifacts(results: Dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "benchmark_30.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    final = results["methods"]["drift_sense_final"]
    comparison_rows = []
    for method, data in results["methods"].items():
        if method == "ablations":
            continue
        m = data["in_distribution"]
        comparison_rows.append({"method": method,
                                "acc@1px": m["accuracies"]["acc@1px"],
                                "acc@5px": m["accuracies"]["acc@5px"],
                                "median_error_px": m["median_error_px"],
                                "p90_error_px": m["p90_error_px"],
                                "p50_latency_ms": m["latency"]["p50_ms"]})
    with (results_dir / "benchmark_30.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison_rows)
    official = final["in_distribution"]
    summary = ["# Verified benchmark summary", "", "Dataset: `synthetic-v2`, 30 in-distribution pairs; one adversarial periodic pair is reported separately.",
               "", "## Official path", "", "`drift_sense_final` = FFT-ZNCC + rotation/scale search + center-prior tie-break + subpixel refinement + ambiguity diagnostics.", "",
               "| Metric | Value |", "|---|---:|",
               f"| Acc@1px | {official['accuracies']['acc@1px']:.2f}% |",
               f"| Acc@3px | {official['accuracies']['acc@3px']:.2f}% |",
               f"| Acc@5px | {official['accuracies']['acc@5px']:.2f}% |",
               f"| Median error | {official['median_error_px']:.3f} px |",
               f"| P90 error | {official['p90_error_px']:.3f} px |",
               f"| P50/P90/P99/max latency | {official['latency']['p50_ms']:.2f}/{official['latency']['p90_ms']:.2f}/{official['latency']['p99_ms']:.2f}/{official['latency']['max_ms']:.2f} ms |",
               f"| Confidence mean/p10 | {official['confidence']['mean']:.3f}/{official['confidence']['p10']:.3f} |",
               f"| Status counts | `{official['status_counts']}` |",
               "", "Numbers above are generated by `python scripts/run_benchmark.py`; they are not universal SEM performance claims."]
    (results_dir / "benchmark_summary.md").write_text("\n".join(summary) + "\n")
