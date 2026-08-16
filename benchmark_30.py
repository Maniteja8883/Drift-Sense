#!/usr/bin/env python3
"""Compatibility wrapper for the canonical benchmark command."""

from scripts.run_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
