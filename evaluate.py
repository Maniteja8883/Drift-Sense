"""
Drift-Sense Evaluation & Benchmark Suite
=========================================
Computes hackathon rubric metrics on the generated challenge dataset.

Metrics:
  - Accuracy @ 1.0 px  (≤ 10 nm)    target > 90%
  - Accuracy @ 3.0 px  (≤ 30 nm)
  - Accuracy @ 5.0 px  (≤ 50 nm)    target > 99%
  - Median Error, P90 Error, Mean Error
  - Latency (ms)
  - Precision-Recall curves over score thresholds τ ∈ [0.5, 0.99]
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

from config import (
    EVAL_THRESHOLDS_PX, SCORE_THRESHOLDS,
    NUM_CHALLENGE_PAIRS,
)
from inference import predict_xy


def load_ground_truth(csv_path: str) -> Dict[int, Tuple[float, float, str]]:
    """Returns {index: (gt_x, gt_y, layout)}."""
    gt = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['index'])
            gt[idx] = (float(row['gt_x']), float(row['gt_y']), row['layout'])
    return gt


def evaluate_pair(ref_path: str, search_path: str,
                  gt_x: float, gt_y: float) -> Tuple[float, float, float, float, float]:
    """
    Run inference on one pair and return (error_px, pred_x, pred_y, score, latency_ms).
    """
    t0 = time.perf_counter()
    pred_x, pred_y, score = predict_xy(ref_path, search_path)
    latency = (time.perf_counter() - t0) * 1000.0
    error = float(np.hypot(pred_x - gt_x, pred_y - gt_y))
    return error, pred_x, pred_y, score, latency


def evaluate_dataset(dataset_dir: str = "dataset",
                     gt_csv: str = "dataset/benchmark_ground_truth.csv"
                     ) -> Dict:
    """
    Evaluate on all pairs (including pathological case if present).
    Returns a dict of aggregate metrics.
    """
    gt = load_ground_truth(gt_csv)
    n = len(gt)

    errors = []
    latencies = []
    scores = []
    layouts = []
    details = []

    for idx in range(n):
        ref_path = f"{dataset_dir}/reference_{idx:03d}.png"
        search_path = f"{dataset_dir}/search_{idx:03d}.png"
        gt_x, gt_y, layout = gt[idx]

        try:
            err, px, py, score, lat = evaluate_pair(ref_path, search_path, gt_x, gt_y)
            errors.append(err)
            latencies.append(lat)
            scores.append(score)
            layouts.append(layout)
            details.append({
                'index': idx,
                'layout': layout,
                'gt': (gt_x, gt_y),
                'pred': (px, py),
                'error': err,
                'latency': lat,
                'score': score,
            })
        except Exception as e:
            print(f"Failed on pair {idx}: {e}")
            details.append({
                'index': idx,
                'layout': layout,
                'error': None,
                'exception': str(e),
            })

    errors_arr = np.array([d['error'] for d in details if d.get('error') is not None])
    latencies_arr = np.array([d['latency'] for d in details if d.get('latency') is not None])
    scores_arr = np.array([d['score'] for d in details if d.get('score') is not None])

    if len(errors_arr) == 0:
        return {'error': 'no successful predictions'}

    # ---- Core metrics -----------------------------------------------------
    accuracies = {}
    for thr in EVAL_THRESHOLDS_PX:
        acc = float(np.mean(errors_arr <= thr) * 100)
        accuracies[f'acc@{thr:.1f}px'] = acc

    median_err = float(np.median(errors_arr))
    p90_err = float(np.percentile(errors_arr, 90))
    mean_err = float(np.mean(errors_arr))
    mean_lat = float(np.mean(latencies_arr))
    p99_lat = float(np.percentile(latencies_arr, 99))
    max_lat = float(np.max(latencies_arr))

    # ---- Precision-Recall over score thresholds --------------------------
    pr_data = []
    for tau in SCORE_THRESHOLDS:
        mask = scores_arr >= tau
        if mask.sum() == 0:
            pr_data.append({'threshold': tau, 'precision': 0.0, 'recall': 0.0})
            continue
        # True positive = error <= 5px among those above threshold
        tp = np.sum((errors_arr <= 5.0) & mask)
        fp = np.sum((errors_arr > 5.0) & mask)
        fn = np.sum(errors_arr <= 5.0) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pr_data.append({'threshold': tau, 'precision': prec, 'recall': rec})

    # ---- Layout-specific breakdown ---------------------------------------
    layout_metrics = {}
    for layout in ['dram', 'finfet', 'pathological_periodic']:
        mask = [d['layout'] == layout for d in details if 'layout' in d]
        idxs = [i for i, m in enumerate(mask) if m]
        if not idxs:
            continue
        le = errors_arr[np.array([d['layout'] == layout
                                  for d in details if 'layout' in d])]
        layout_metrics[layout] = {
            'count': int(len(le)),
            'acc@1px': float(np.mean(le <= 1.0) * 100),
            'acc@5px': float(np.mean(le <= 5.0) * 100),
            'median_err': float(np.median(le)),
        }

    return {
        'n_evaluated': int(len(errors_arr)),
        'n_total': n,
        'accuracies': accuracies,
        'median_error_px': median_err,
        'p90_error_px': p90_err,
        'mean_error_px': mean_err,
        'mean_latency_ms': mean_lat,
        'p99_latency_ms': p99_lat,
        'max_latency_ms': max_lat,
        'latency_budget_ok': max_lat < 80.0,
        'pr_curve': pr_data,
        'by_layout': layout_metrics,
        'details': details,
    }


def print_report(metrics: Dict) -> None:
    """Pretty-print the evaluation results."""
    print("\n" + "=" * 60)
    print("DRIFT-SENSE EVALUATION REPORT")
    print("=" * 60)
    print(f"Pairs evaluated:  {metrics['n_evaluated']} / {metrics['n_total']}")
    print(f"Median error:     {metrics['median_error_px']:.3f} px")
    print(f"P90 error:        {metrics['p90_error_px']:.3f} px")
    print(f"Mean error:       {metrics['mean_error_px']:.3f} px")
    print(f"Mean latency:     {metrics['mean_latency_ms']:.2f} ms")
    print(f"P99 latency:      {metrics['p99_latency_ms']:.2f} ms")
    print(f"Max latency:      {metrics['max_latency_ms']:.2f} ms")
    print(f"Latency < 80 ms:  {'PASS' if metrics['latency_budget_ok'] else 'FAIL'}")
    print("-" * 60)
    for k, v in metrics['accuracies'].items():
        target = " (>90%)" if "1.0" in k else " (>99%)" if "5.0" in k else ""
        status = "✓" if (("1.0" in k and v >= 90) or ("5.0" in k and v >= 99)) else "✗"
        print(f"  {k}: {v:.2f}%{target} {status}")
    print("-" * 60)
    print("By layout:")
    for layout, m in metrics['by_layout'].items():
        print(f"  {layout:20s}  n={m['count']:2d}  "
              f"acc@1px={m['acc@1px']:.1f}%  acc@5px={m['acc@5px']:.1f}%  "
              f"median={m['median_err']:.3f}")
    print("=" * 60)


def save_json(metrics: Dict, path: str = "evaluation_results.json") -> None:
    # Convert numpy types to Python types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    with open(path, 'w') as f:
        json.dump(convert(metrics), f, indent=2)


if __name__ == "__main__":
    metrics = evaluate_dataset()
    print_report(metrics)
    save_json(metrics)
    print(f"\nSaved detailed results to evaluation_results.json")