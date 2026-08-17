# Reproducibility guide for reviewers

This guide separates the fast evaluator smoke test from the longer evidence commands. All commands assume a fresh clone and work without editing source files.

## A. Fresh environment smoke test

```bash
git clone https://github.com/Maniteja8883/Drift-Sense.git
cd Drift-Sense
python -m venv fresh_env
source fresh_env/bin/activate                 # Windows: fresh_env\Scripts\activate
python -m pip install -r requirements.txt
python inference.py \
  --reference examples/reference.png \
  --search examples/search.png
```

Expected stdout is exactly one line with two finite numbers:

```text
499.5000 499.5000
```

The command is independent of the current working directory because `inference.py` resolves the package relative to its own location. To verify that explicitly:

```bash
cd /tmp
python /absolute/path/to/Drift-Sense/inference.py \
  --reference /absolute/path/to/Drift-Sense/examples/reference.png \
  --search /absolute/path/to/Drift-Sense/examples/search.png
```

## B. Generate a sample dataset

```bash
python dataset_generator.py \
  --architecture DRAM \
  --num-pairs 5 \
  --output-dir ./review_dataset \
  --seed 42
```

The generator accepts `DRAM`, `FinFET`, or `both`. It writes `reference_*.png`, `search_*.png`, `benchmark_ground_truth.csv`, `manifest.csv`, and metadata. The CSV begins with:

```text
index,architecture,reference,search,gt_x,gt_y
```

The seed makes the generated data deterministic. This generator is independent of inference and exists for development and benchmark creation.

## C. Run the tests

```bash
python -m pytest tests/ -v
```

The verified checkout passes 11 tests covering geometry, direct-vs-FFT ZNCC, invalid/constant inputs, NMS, subpixel refinement, periodic ambiguity, and pipeline validation.

## D. Run the benchmark

```bash
python scripts/run_benchmark.py
```

The command creates the deterministic `synthetic-v2` dataset using seed `0xD1575317`, evaluates the official method and diagnostic baselines, and writes:

- `results/benchmark_30.json` — metrics and environment metadata;
- `results/benchmark_30.csv` — method comparison table;
- `results/benchmark_summary.md` — human-readable summary.

The stored official in-distribution results are 80.00% Acc@1px, 90.00% Acc@3px, 100.00% Acc@5px, 0.513 px median error, and 2.611 px P90 error. The separate periodic adversarial case is reported as `AMBIGUOUS` and is not included in those accuracy figures.

## E. Verification checklist

- [x] No model weights or hidden downloads are required.
- [x] `inference.py` accepts only reference/search paths and prints only coordinates.
- [x] The public API works from another current working directory.
- [x] The generator records ground truth but inference never reads it.
- [x] `requirements.txt` is a complete frozen environment capture.
- [x] Benchmark metadata records software, hardware, latency method, and Git metadata.

## What is not claimed

The evidence is from a physics-informed synthetic corruption approximation, not from a calibrated real SEM tool. Latency is CPU- and machine-dependent. Confidence is heuristic rather than a calibrated probability. These boundaries are documented in [`assumptions.md`](assumptions.md) and [`limitations.md`](limitations.md).
