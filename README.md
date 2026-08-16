# Drift-Sense

Deterministic semiconductor image localization for recovering a known inspection site under small stage drift, scale/rotation mismatch, and periodic visual ambiguity.

![Annotated demo result](examples/result.png)

## What is the official system?

There is exactly one official inference/benchmark path:

`src/drift_sense/pipeline.py` → `LocalizationEngine` → FFT-ZNCC coarse search → rotation/scale grid → center-prior tie-break → full-resolution refinement → subpixel peak refinement → confidence/status diagnostics.

The original PyTorch research stack is preserved under [`experimental/ml/`](experimental/ml/), but it is not imported or used for the reported results because no verified checkpoint or reproducible ML benchmark was present.

The official path is classical and requires no model weights. The inference script does not read generated ground truth, pair filenames, metadata, or benchmark outputs; the audit is recorded in [`docs/leakage_audit.md`](docs/leakage_audit.md).

## Why this problem is difficult

A known high-resolution reference must be localized in a larger, lower-resolution search image. Repeated semiconductor structures create near-equal correlation peaks, so a high score alone does not guarantee the correct site. Drift also changes scale, rotation, brightness, and local appearance.

## Quick start

Python 3.9+ is supported. The official path needs only NumPy and Pillow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py
python -m pytest -q
python scripts/run_benchmark.py
```

`requirements.txt` is intentionally a complete frozen environment capture as required by the submission contract; it includes the test/lint packages used to verify this checkout. The optional `requirements-dev.txt` and `requirements-ml.txt` files are supplementary.

## Submission-contract commands

Generate pairs independently of the benchmark:

```bash
python dataset_generator.py \
  --architecture DRAM \
  --num-pairs 3 \
  --output-dir ./dataset_sample \
  --seed 123
```

The output directory contains `reference_000.png`, `search_000.png`, and `benchmark_ground_truth.csv` (plus `manifest.csv` and metadata). The ground-truth CSV records `index`, `architecture`, `reference`, `search`, `gt_x`, `gt_y`, and the partition/noise metadata. Use `--architecture FinFET` for FinFET-only pairs or `both` for an alternating mixed set. The same seed produces the same generated files.

Run the public evaluator-facing API:

```bash
python inference.py \
  --reference examples/reference.png \
  --search examples/search.png
# stdout: one line containing exactly: x y
```

The script resolves its package path from its own repository location, so it also works from another directory:

```bash
cd /tmp
python /absolute/path/to/drift_sense/inference.py \
  --reference /absolute/path/to/drift_sense/examples/reference.png \
  --search /absolute/path/to/drift_sense/examples/search.png
```

Only the coordinate line goes to stdout. Diagnostics are opt-in via `--time` and go to stderr; the richer `scripts/run_demo.py` is for human-readable status/confidence output.

The strict challenge-compatible wrapper prints only coordinates:

```bash
python inference.py --reference examples/reference.png --search examples/search.png
```

The richer demo prints coordinates, score, heuristic confidence, status, latency, and writes `examples/result.png`.

## Demo output

The included example is deterministic for coordinates and score. A run on the checked-out example reports:

```json
{
  "x": 499.5,
  "y": 499.5,
  "confidence": 0.6781366495,
  "status": "SUCCESS",
  "score": 0.2881953716
}
```

The exact latency is machine-dependent; run the command to obtain the local value.

## Verified benchmark

The canonical command generates 30 in-distribution synthetic pairs (15 DRAM-like, 15 FinFET-like) and one separate adversarial periodic pair, then stores machine-readable and human-readable artifacts under [`results/`](results/).

```bash
python scripts/run_benchmark.py
```

The committed [`results/benchmark_summary.md`](results/benchmark_summary.md) and [`results/benchmark_30.json`](results/benchmark_30.json) are the source of truth for measured values. The benchmark reports Acc@1px/3px/5px, median/mean/P90/P95 error, P50/P90/P95/P99/max latency, method comparisons, status counts, and environment metadata. Results are scoped to this synthetic dataset and recorded CPU; they are not industrial SEM validation.

## Baseline comparison

The benchmark measures:

| Method | What it tests |
|---|---|
| `naive_single_scale` | One unwarped FFT-ZNCC map with raw argmax |
| `fft_zncc_center_tie` | Single-scale FFT-ZNCC plus the documented center tie-break |
| `fft_zncc_geometric` | Rotation/scale search with coarse coordinate output |
| `drift_sense_final` | Official geometric search plus fine refinement and diagnostics |

The measured table is regenerated in [`results/benchmark_30.csv`](results/benchmark_30.csv); no numbers are hard-coded in this README.

## Ambiguity and failure handling

The richer API returns `(x, y)`, a matching score, heuristic confidence, and one of:

- `SUCCESS`
- `AMBIGUOUS`
- `LOW_CONFIDENCE`
- `OUT_OF_DISTRIBUTION`

Ambiguity is based on separated near-equal peaks. The periodic adversarial sample is reported separately so it cannot inflate the in-distribution accuracy claim. The strict wrapper still prints coordinates for challenge integrations; applications should inspect the richer result status.

## Data and assumptions

The generator is a **physics-informed synthetic SEM corruption approximation**, not authentic or validated SEM noise. It simulates Poisson shot noise, an edge/charging approximation, smooth charging gradients, scanline jitter, and bounded rotation/scale mismatch. Details and limitations are in [`docs/benchmark_methodology.md`](docs/benchmark_methodology.md), [`docs/assumptions.md`](docs/assumptions.md), and [`docs/limitations.md`](docs/limitations.md).

The technical choices are supported and numbered in [`docs/references.md`](docs/references.md): [1] image formation/shot-noise context, [2] charging and image defects, [3] normalized cross-correlation, and [4] coarse-to-fine signal processing context.

The default coordinate geometry is 1 nm/pixel reference, 10 nm/pixel search, and a 100×100 search-space template. The 49.5 center offset is derived from `(100 - 1)/2` and tested in [`tests/test_geometry.py`](tests/test_geometry.py). The center prior is a challenge-scoped assumption, not universal capability.

## Reproducibility

- `requirements.txt` is the frozen `pip freeze` from the verified development environment used for the committed benchmark; `requirements-dev.txt` and `requirements-ml.txt` are supplementary engineering files.
- Dataset seed: `0xD1575317`.
- Dataset version: `synthetic-v2`.
- Dataset manifest, metadata, split/partition labels, and pair-level records are generated by the canonical command.
- Environment and Git commit are recorded in the JSON artifact where available.
- Tests cover geometry, FFT-ZNCC against a direct reference, constant/invalid inputs, NMS, subpixel refinement, periodic ambiguity, and pipeline dimension handling.

## Project structure

```text
src/drift_sense/       official package: geometry, matching, pipeline, diagnostics, dataset
baseline/              reusable FFT-ZNCC baseline
benchmark/              metric, comparison, and artifact generation code
scripts/                canonical demo and benchmark commands
tests/                  numerical and pipeline tests
docs/                   architecture, methodology, assumptions, limitations, audit
examples/               deterministic input pair and annotated output
results/                verified benchmark artifacts
experimental/ml/        retained PyTorch research implementation, not official
```

## Limitations and future work

The benchmark is synthetic, layouts are simplified, and no real SEM calibration or tool-side validation is included. The center prior can bias off-center targets, and periodic cases may remain fundamentally ambiguous. Future work should validate on labeled real SEM acquisitions, calibrate confidence, and only then reconsider whether the experimental ML path improves the official result.

## License

MIT; see [`LICENSE`](LICENSE).
