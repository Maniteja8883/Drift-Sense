# Ground-truth leakage audit

The evaluator-facing entry point is `inference.py` and calls only `load_gray()` plus `LocalizationEngine.predict_arrays()`.

The inference path was inspected for the following prohibited dependencies:

- `benchmark_ground_truth.csv`: not imported or opened;
- pair index, filename conventions, or dataset metadata: not read;
- generator-only seeds or noise multipliers: not read;
- ground-truth coordinates: not passed to the engine;
- model weights/checkpoints: not required because the official path is classical;
- current working directory: not used to resolve package imports or inputs.

The only challenge-scoped prior used internally is the documented nominal search-frame center `(500, 500)`, applied as a tie-break/fallback under the stated center-prior assumption. This information is available from the challenge specification, not from the generated ground-truth files. The benchmark and generator are decoupled from prediction.

