# Baseline report

The repository initially contained root-level classical modules (`common.py`, `inference.py`, `dataset_generator.py`, `evaluate.py`, `benchmark_30.py`) and a separate, substantial PyTorch stack (`drift_sense_model.py`, `losses_and_postprocess.py`, `train_eval.py`). The default requirements omitted PyTorch, no trained checkpoint was present, and the ML config imports were inconsistent with the classical config.

The initial README reported benchmark values that were not reproducible from the checked-out repository and described synthetic corruption as authentic SEM noise. The original generator placed one feature on an otherwise mostly empty canvas and always used the nominal center, so its periodic ambiguity claim was not supported by the data construction.

Final decision: the deterministic FFT-ZNCC coarse-to-fine path is official; the PyTorch stack is retained under `experimental/ml/` and is explicitly excluded from claims. The new generator labels its images synthetic and includes a separate adversarial periodic partition. The canonical command, tests, and stored result are the evidence path.

