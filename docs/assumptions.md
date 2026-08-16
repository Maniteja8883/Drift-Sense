# Assumptions

- The reference is 1000×1000 pixels at 1 nm/pixel and the search is 1000×1000 pixels at 10 nm/pixel. The reference therefore becomes a 100×100 search-space template.
- The challenge specification supplies a nominal search-frame center prior. The official tie-break selects the candidate closest to `(500, 500)` only when candidates are within 5% of the best score.
- This prior is not a universal localization capability. If the target is far from the center, the engine can report `OUT_OF_DISTRIBUTION`; a prior ablation is recorded in the benchmark artifacts.
- Rotation and scale are searched only over ±3° and 0.95–1.05. Larger geometric mismatch is outside the supported operating envelope.
- Confidence is a bounded heuristic built from score, peak margin, and prior agreement. It is not a calibrated probability.
- Dataset images are synthetic. “Edge charging,” “charging gradient,” and “scanline jitter” are engineering approximations, not validated SEM physics.

