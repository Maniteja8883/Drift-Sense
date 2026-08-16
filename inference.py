#!/usr/bin/env python3
"""Strict challenge-compatible wrapper around the one official path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from drift_sense.io import load_gray
from drift_sense.pipeline import LocalizationEngine


def predict_xy(ref_path: str, search_path: str) -> Tuple[float, float, float]:
    result = LocalizationEngine().predict_arrays(load_gray(ref_path), load_gray(search_path))
    return result.x, result.y, result.score


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Drift-Sense official localization")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--search", required=True)
    args = parser.parse_args(argv)
    try:
        x, y, _ = predict_xy(args.reference, args.search)
        print(f"{x:.4f} {y:.4f}")
        return 0
    except Exception as exc:
        print(f"Drift-Sense inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
