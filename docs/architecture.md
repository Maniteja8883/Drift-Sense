# Architecture

## Decision in one sentence

The only official system is a deterministic classical FFT-ZNCC localization pipeline; the retained PyTorch implementation is experimental and cannot affect inference or benchmark results.

## Official data flow

`inference.py` loads two grayscale images and calls `LocalizationEngine.predict_arrays()` from `src/drift_sense/pipeline.py`:

1. Validate that both images are finite 1000×1000 arrays.
2. Area-average the reference by 10× to obtain a 100×100 template in search-image pixels.
3. Downsample the search and template by 4× for a cheap coarse pass.
4. Evaluate FFT-ZNCC over five rotations and three scales.
5. Find local maxima, apply NMS, and use the documented center prior for configured ties/fallbacks.
6. Crop a full-resolution ROI around the coarse candidate.
7. Search the neighboring transform grid in that ROI.
8. Refine the winning peak with bounded three-point parabolas.
9. Convert top-left coordinates to template-center coordinates and classify evidence as `SUCCESS`, `AMBIGUOUS`, `LOW_CONFIDENCE`, or `OUT_OF_DISTRIBUTION`.

The public wrapper prints only `x y`. The package API and demo expose the score, heuristic confidence, status, latency, and diagnostics.

## Component evidence

| Component | Problem solved | Evidence | Reproduction |
|---|---|---|---|
| Area downsampling | Maps reference resolution into search coordinates | geometry and shape tests | `python -m pytest tests/test_geometry.py -v` |
| FFT-ZNCC | Measures local similarity while reducing brightness/contrast sensitivity | direct reference comparison and score-range tests | `python -m pytest tests/test_matching.py -v` |
| Rotation/scale grid | Handles bounded acquisition geometry mismatch | baseline comparison artifact | `python scripts/run_benchmark.py` |
| Peak finding/NMS | Retains separated candidates while suppressing duplicate neighbors | overlapping/separated peak tests | `python -m pytest tests/test_matching.py -v` |
| Center-prior tie-break | Uses the challenge’s near-center rule for genuine score ties | assumption and baseline evidence | `docs/assumptions.md`, benchmark JSON |
| Subpixel refinement | Estimates a fractional peak rather than returning only an integer | synthetic fractional-peak tests | `python -m pytest tests/test_refinement.py -v` |
| Ambiguity diagnostics | Avoids presenting periodic ties as certain matches | adversarial status record | `python scripts/run_benchmark.py` |
| Confidence heuristic | Provides a bounded ranking signal for downstream review | benchmark confidence/status counts | `results/benchmark_30.json` |

## Why coarse-to-fine?

An exhaustive full-resolution geometric search would spend the most computation at every possible position. The coarse pass reduces the candidate region and transform neighborhood; the fine pass then recovers accuracy where it matters. This is a predictable engineering tradeoff: less computation than a full dense search, with explicit bounds that define the supported operating envelope.

## Official versus experimental

The official path is implemented in `src/drift_sense/` and is used by `inference.py`, `scripts/run_demo.py`, and `scripts/run_benchmark.py`. No weights, training data, or PyTorch installation are required.

The original learned code is preserved under `experimental/ml/`. It has no verified checkpoint and no evidence that justifies selecting it for the submission path. It is labeled experimental, is not imported by the official package, and is excluded from official benchmark claims.
