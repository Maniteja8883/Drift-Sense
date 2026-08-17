# Getting started

This is the short command card for the Drift-Sense submission.

## 1. Install

```bash
git clone https://github.com/Maniteja8883/Drift-Sense.git
cd Drift-Sense
python -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is the frozen environment used for the verified benchmark. No model download is required because the official path is classical.

## 2. Run the evaluator-facing API

```bash
python inference.py \
  --reference examples/reference.png \
  --search examples/search.png
```

Expected stdout:

```text
499.5000 499.5000
```

The wrapper prints only `x y`. Use the demo for diagnostics:

```bash
python scripts/run_demo.py
```

## 3. Generate independent pairs

```bash
python dataset_generator.py \
  --architecture DRAM \
  --num-pairs 5 \
  --output-dir ./test_data \
  --seed 42
```

The generator writes PNG pairs and `benchmark_ground_truth.csv` with the true center for every pair.

## 4. Verify and benchmark

```bash
python -m pytest tests/ -v
python scripts/run_benchmark.py
```

For the full judge-facing procedure, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
