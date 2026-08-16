# Final three-question audit

| Component | Problem solved | Evidence | Reproduction | Status |
|---|---|---|---|---|
| FFT-ZNCC | Gain/bias-robust template matching | reference-vs-FFT unit test | `pytest -q` | official |
| Geometric search | Small rotation/scale mismatch | benchmark comparison | `python scripts/run_benchmark.py` | official |
| Center prior | Challenge-defined tie resolution | benchmark and assumptions | benchmark JSON + `docs/assumptions.md` | official, constrained |
| Subpixel refinement | Fractional peak localization | fractional synthetic test | `pytest -q` | official |
| Ambiguity status | Surface periodic uncertainty | adversarial record/status | benchmark JSON | official diagnostic |
| Synthetic generator | Repeatable test data | manifest, seed, metadata | benchmark command | official benchmark support |
| PyTorch model | Research direction | no verified checkpoint/result | `experimental/ml/README.md` | experimental |
| `inference.py` | Public evaluator API | standalone stdout/exit-code tests | `python inference.py --reference ... --search ...` | official |
| `dataset_generator.py` | Independent pair creation | CLI and ground-truth schema | `python dataset_generator.py --architecture DRAM --num-pairs 3 --output-dir ./dataset` | official support |
| `requirements.txt` | Recreate verified environment | frozen `pip freeze` capture | `pip install -r requirements.txt` | submission artifact |
