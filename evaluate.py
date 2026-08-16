#!/usr/bin/env python3
"""Compatibility wrapper for benchmark evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from benchmark.evaluate import evaluate_manifest, write_artifacts


def main() -> int:
    result = evaluate_manifest("dataset/manifest.csv", ROOT)
    write_artifacts(result, ROOT / "results")
    print("Saved results/benchmark_30.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
