# Benchmark methodology

Run:

```bash
python scripts/run_benchmark.py
```

The generator is independently callable for reviewer data creation:

```bash
python dataset_generator.py --architecture DRAM --num-pairs 10 --output-dir ./dataset --seed 123
```

It writes reference/search PNGs, `benchmark_ground_truth.csv`, `manifest.csv`, and `dataset_metadata.json`. The inference script never reads these files.

The command deterministically generates 30 in-distribution pairs (15 DRAM-like and 15 FinFET-like) plus one adversarial periodic pair. The default seed is `0xD1575317`, recorded in `dataset/dataset_metadata.json` and the committed result metadata.

The images are a physics-informed synthetic corruption approximation. It includes Poisson shot noise, an edge/charging approximation, a smooth charging gradient, scanline jitter, and a small rotation/scale mismatch. It is not calibrated against real SEM measurements.

Metrics use Euclidean error in search-image pixels against the generated manifest: Acc@1px, Acc@3px, Acc@5px, median/mean/P90/P95 error, and P50/P90/P95/P99/max inference latency. Latency is measured with `time.perf_counter` around in-memory inference; PNG IO is excluded. The official path is CPU-only and the result records Python, NumPy, Pillow, OS, CPU, CPU count, and Git commit when available.

The benchmark also measures single-scale, center-tie, geometric-search, and final methods. Ablations compare the final result with rounded coordinates and with rotation/scale search disabled. The adversarial periodic sample is reported separately; it is not folded into the in-distribution accuracy claim.
