# Submission-contract verification

These checks were run against the current checkout after the rebuild:

```bash
python -m pip freeze > /tmp/actual_freeze.txt
diff -u <(sort -f /tmp/actual_freeze.txt) <(sort -f requirements.txt)
python dataset_generator.py --architecture DRAM --num-pairs 3 --output-dir /tmp/drift_sense_sample --seed 123
python inference.py --reference examples/reference.png --search examples/search.png
python -m pytest -q
python scripts/run_demo.py
python scripts/run_benchmark.py
```

The frozen requirements comparison matched the committed `requirements.txt`. The standalone generator created three DRAM pairs and `benchmark_ground_truth.csv`. The evaluator-facing inference script exited successfully and printed exactly two finite coordinate tokens. The same inference command was run from `/tmp` with absolute script and image paths, and it produced the same coordinate-only stdout format. The clean environment installed `requirements.txt`, then passed the same standalone inference test and all 11 tests.

The official path is classical FFT-ZNCC, so model weights and a training script are not submission dependencies. The retained PyTorch code is explicitly experimental and cannot affect `inference.py`.

