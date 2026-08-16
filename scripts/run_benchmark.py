#!/usr/bin/env python3
"""Canonical command: generate and evaluate the official benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from drift_sense.dataset import generate_dataset
from benchmark.evaluate import evaluate_manifest, write_artifacts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run Drift-Sense benchmark")
    parser.add_argument("--dataset", default=str(ROOT / "dataset"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0xD1575317)
    args = parser.parse_args(argv)
    manifest = generate_dataset(args.dataset, args.pairs, args.seed, include_pathological=True)
    results = evaluate_manifest(manifest, ROOT)
    write_artifacts(results, Path(args.results))
    official = results["methods"]["drift_sense_final"]["in_distribution"]
    adversarial = results["methods"]["drift_sense_final"]["adversarial"]
    print("Drift-Sense official benchmark")
    print(f"in-distribution: n={official['n']} Acc@1px={official['accuracies']['acc@1px']:.2f}% Acc@5px={official['accuracies']['acc@5px']:.2f}%")
    print(f"adversarial ambiguity: n={adversarial['n']} status is recorded per sample")
    print(f"artifacts: {Path(args.results).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

