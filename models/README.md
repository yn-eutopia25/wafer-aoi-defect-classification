# Local model artifacts

Trained weights and serialized classifiers are stored here locally and are excluded from Git. Typical artifacts include `.pt`, `.joblib`, `.onnx` and TensorRT engine files.

Record model configuration, input data fingerprint and evaluation split in the corresponding local model card before using a model for comparison.

Authorized formal model artifacts are distributed as a private Release attachment, not as Git objects. See [`../docs/PRIVATE_RELEASE.md`](../docs/PRIVATE_RELEASE.md) for the package name and integrity-check procedure.
