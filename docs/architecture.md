# Architecture

## Official path

`LocalizationEngine` in `src/drift_sense/pipeline.py` is the only implementation used by the demo and canonical benchmark.

1. Load 1000×1000 grayscale reference and search images.
2. Area-average the 1000-pixel reference by 10× to obtain a 100×100 search-space template.
3. Search a 4× downsampled frame over a 5-angle × 3-scale grid using FFT-ZNCC.
4. Apply NMS and the challenge-specific center-prior tie-break to select a coarse ROI.
5. Search a full-resolution ROI over the neighboring transform grid.
6. Refine the winning correlation peak with a bounded three-point parabola.
7. Report coordinates plus heuristic confidence, ambiguity evidence, and status.

The output coordinate is `top_left + (template_size - 1)/2`. For the default 100×100 template this is `top_left + 49.5`; the value is derived in `geometry.py` and covered by tests.

## Components

| Component | Problem solved | Evidence | Reproduction |
|---|---|---|---|
| FFT-ZNCC | Robust similarity under gain/bias changes | direct-ZNCC unit comparison | `pytest -q` |
| Rotation/scale grid | Small acquisition geometry changes | measured baseline comparison | `python scripts/run_benchmark.py` |
| Center-prior tie-break | Challenge rule for near-equal periodic candidates | baseline and assumption ablation | benchmark JSON |
| Subpixel parabola | Fractional peak placement | synthetic fractional-peak test | `pytest -q` |
| Ambiguity diagnostics | Avoid hiding periodic ties | adversarial partition/status records | benchmark JSON |

## Experimental path

The original PyTorch architecture is in `experimental/ml/`. It is intentionally not imported by the official package, has no committed trained checkpoint, and is not used for the reported results.

