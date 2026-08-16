#!/usr/bin/env python3
"""
Drift-Sense Benchmark-30
=========================
Official 30-pair benchmark runner for the Applied Materials Hackathon.

Generates the 30 curated challenge pairs (15 DRAM, 15 FinFET) with varied
noise intensities, plus 1 explicit pathological periodic failure case.

Runs the full inference pipeline on all pairs and produces a results table
matching the official rubric format.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

def main() -> int:
    # Ensure we're in the drift_sense directory
    root = Path(__file__).parent

    print("=" * 70)
    print("DRIFT-SENSE BENCHMARK-30  (Applied Materials Hackathon)")
    print("=" * 70)

    # Phase 4: Generate dataset
    print("\n[1/3] Generating physics-informed SEM dataset (30 pairs + 1 pathological)...")
    result = subprocess.run([
        sys.executable, "-m", "dataset_generator"
    ], cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Dataset generation failed:\n{result.stderr}")
        return 1
    print(result.stdout.strip())

    # Phase 5: Evaluate
    print("\n[2/3] Running evaluation on all pairs...")
    result = subprocess.run([
        sys.executable, "-m", "evaluate"
    ], cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Evaluation failed:\n{result.stderr}")
        return 1
    print(result.stdout.strip())

    # Phase 6: Verify rubric targets
    print("\n[3/3] Verifying hackathon rubric targets...")
    import json
    with open(root / "evaluation_results.json") as f:
        metrics = json.load(f)

    print("\n" + "=" * 70)
    print("RUBRIC VERIFICATION")
    print("=" * 70)

    acc1 = metrics['accuracies']['acc@1.0px']
    acc5 = metrics['accuracies']['acc@5.0px']
    max_lat = metrics['max_latency_ms']

    targets = [
        ("Accuracy @ 1.0 px", acc1, 90.0, "≥ 90%"),
        ("Accuracy @ 5.0 px", acc5, 99.0, "≥ 99%"),
        ("Max Latency", max_lat, 80.0, "< 80 ms"),
    ]

    all_pass = True
    for name, value, target, direction in targets:
        if "Latency" in name:
            passed = value < target
        else:
            passed = value >= target
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s} : {value:6.2f}  (target {direction})  {status}")
        if not passed:
            all_pass = False

    print("=" * 70)
    if all_pass:
        print("ALL RUBRIC TARGETS MET  🎉")
        return 0
    else:
        print("SOME TARGETS MISSED")
        return 1


if __name__ == "__main__":
    sys.exit(main())