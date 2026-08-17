# Limitations

## Evidence scope

- The benchmark uses synthetic layouts and corruption; no real SEM dataset or tool-side validation is included.
- DRAM-like and FinFET-like motifs are simplified engineering test patterns, not a production process design kit or calibrated wafer model.
- The 30-pair benchmark is reproducible evidence for this implementation, not proof of industrial generalization.

## Algorithmic boundaries

- The center prior can improve challenge-scoped near-ties but can bias a genuinely off-center target.
- Periodic layouts can remain fundamentally ambiguous. The system reports `AMBIGUOUS` when its peak evidence supports that conclusion, but it cannot manufacture information that is absent from the pixels.
- Rotation and scale outside ±3° and 0.95–1.05 are outside the configured search envelope.
- Confidence is heuristic, not a calibrated probability or a formal uncertainty interval.
- Invalid dimensions, non-finite values, and low-variance inputs are rejected or assigned weak evidence rather than silently accepted.

## Performance boundaries

- Reported latency depends on CPU, NumPy/FFT implementation, operating system, image dimensions, and process state.
- Benchmark latency measures in-memory prediction and excludes PNG input/output.
- The official path is CPU-oriented and does not claim GPU acceleration.

## Experimental ML boundary

The retained PyTorch implementation has no verified checkpoint and is not part of the official inference or benchmark path. It must not be described as a trained production model until a checkpoint, training evidence, and independent validation are available.

## Next validation steps

The highest-value future work is labeled real-SEM validation, confidence calibration on held-out tool conditions, broader geometric stress testing, and an explicit decision about whether any learned component improves the official evidence without harming reproducibility.
