#!/usr/bin/env python3
"""Run the deterministic included example and save an annotated image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_sense.config import DEFAULT_CONFIG
from drift_sense.io import load_gray
from drift_sense.pipeline import LocalizationEngine


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=str(ROOT / "examples" / "reference.png"))
    parser.add_argument("--search", default=str(ROOT / "examples" / "search.png"))
    parser.add_argument("--output", default=str(ROOT / "examples" / "result.png"))
    args = parser.parse_args(argv)
    result = LocalizationEngine().predict_arrays(load_gray(args.reference), load_gray(args.search))
    canvas = Image.open(args.search).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    x, y = result.x, result.y
    half = DEFAULT_CONFIG.geometry.template_size / 2.0
    draw.rectangle((x - half, y - half, x + half, y + half), outline=(255, 60, 40), width=3)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(40, 220, 80))
    canvas.save(args.output)
    print(json.dumps({"x": result.x, "y": result.y, "confidence": result.confidence,
                      "status": result.status, "score": result.score,
                      "latency_ms": result.latency_ms, "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
