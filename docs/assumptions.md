# Assumptions

These assumptions define the operating envelope of the official path. They are not claims of universal semiconductor-tool capability.

## Image geometry

- The reference and search images are grayscale 1000×1000 arrays.
- The reference is treated as 1 nm/pixel and the search as 10 nm/pixel.
- Area averaging by 10× therefore produces a 100×100 template in search-image pixels.
- A template top-left coordinate maps to its center by `(template_size - 1)/2`; for the default template this is `49.5` pixels.

## Challenge-scoped center prior

The challenge supplies a nominal return-to-center condition. The engine uses `(500, 500)` as a tie-break/fallback prior, not as a universal answer:

- candidates within the configured five-percent score tie can be ranked by distance to the prior;
- a weak, remote coarse peak can trigger a center fallback before the fine ROI search;
- a genuinely off-center target can be biased toward the prior or classified `OUT_OF_DISTRIBUTION`.

The benchmark and tests keep this assumption visible. A deployment without the challenge guarantee should remove or replace this prior and revalidate the method.

## Geometric mismatch

The official grid searches rotations from −3° to +3° and scales from 0.95× to 1.05×. Larger mismatch is outside the supported envelope; a low score or `OUT_OF_DISTRIBUTION` status is preferable to an unsupported-confidence claim.

## Confidence and status

Confidence combines correlation score, peak margin, and agreement with the center prior. It is a bounded heuristic from 0 to 1, not a calibrated probability. Calibration would require representative labeled validation data, especially real SEM data.

## Synthetic data

The generator is a physics-informed engineering approximation. Poisson shot noise, charging/edge effects, smooth gradients, scanline jitter, and geometric mismatch are simulated to exercise the pipeline. They are not a validated physical SEM simulator, and synthetic benchmark performance does not establish industrial generalization.
