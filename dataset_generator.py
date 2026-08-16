#!/usr/bin/env python3
"""Standalone synthetic reference/search pair generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from drift_sense.dataset import generate_dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic Drift-Sense pairs")
    parser.add_argument("--architecture", choices=("DRAM", "FinFET", "both", "BOTH"),
                        default="both", help="layout family for every pair (default: both)")
    parser.add_argument("--num-pairs", type=int, default=30,
                        help="number of ordinary pairs to generate")
    parser.add_argument("--output-dir", default="dataset",
                        help="directory for PNGs and benchmark_ground_truth.csv")
    parser.add_argument("--seed", type=int, default=0xD1575317,
                        help="deterministic NumPy seed")
    parser.add_argument("--include-pathological", action="store_true",
                        help="also add one labelled periodic ambiguity pair")
    args = parser.parse_args(argv)
    print(generate_dataset(args.output_dir, args.num_pairs, args.seed,
                           args.include_pathological, args.architecture))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
