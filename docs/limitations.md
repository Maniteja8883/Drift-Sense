# Limitations

- No real SEM dataset or tool-side validation is included.
- The synthetic layouts are simple DRAM-like and FinFET-like motifs, not a production process design kit or a calibrated wafer model.
- The center prior can rescue a near-tie under the challenge rule but can bias a prediction when the target is genuinely off-center.
- Periodic layouts can produce multiple visually indistinguishable peaks. The engine exposes `AMBIGUOUS` rather than treating a tie as proof of correctness.
- Latency depends on CPU, NumPy FFT implementation, BLAS, and image dimensions. Reported latency is hardware-specific.
- The ML implementation is experimental and has no verified checkpoint or official benchmark evidence.

