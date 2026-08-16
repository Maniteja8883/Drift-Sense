"""CLI helper for deterministic synthetic dataset generation."""

from __future__ import annotations

import argparse

from drift_sense.dataset import generate_dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset")
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--architecture", choices=("DRAM", "FinFET", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0xD1575317)
    parser.add_argument("--no-pathological", action="store_true")
    args = parser.parse_args(argv)
    path = generate_dataset(args.output, args.pairs, args.seed, not args.no_pathological,
                            args.architecture)
    print(f"Generated dataset manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
