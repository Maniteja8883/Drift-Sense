"""Library/CLI entry point for the official benchmark evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluate import evaluate_manifest, write_artifacts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="dataset/manifest.csv")
    parser.add_argument("--results", default="results")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    results = evaluate_manifest(args.manifest, root)
    write_artifacts(results, Path(args.results))
    official = results["methods"]["drift_sense_final"]["in_distribution"]
    print(f"Official Acc@1px: {official['accuracies']['acc@1px']:.2f}%")
    print(f"Official Acc@5px: {official['accuracies']['acc@5px']:.2f}%")
    print(f"Official median error: {official['median_error_px']:.3f} px")
    print(f"Official P50/P99 latency: {official['latency']['p50_ms']:.2f}/{official['latency']['p99_ms']:.2f} ms")
    print(f"Artifacts written to {args.results}/")
    return 0

