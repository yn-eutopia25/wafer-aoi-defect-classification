# Reproducibility protocol

## Data source of truth

`data/annotations/annotations.json` is the authoritative manual annotation file. `instances.csv`, `image_summary.csv`, patches and YOLO labels are derived artifacts and must be regenerated from the source when annotations change.

## Split policy

Split by source `image_id`, never by annotation or patch. All boxes, patches and tiles derived from one image must remain in the same split. Fix the random seed and persist a split lock containing the image IDs and input fingerprints.

## Model selection

Use the training set for fitting and the validation set for model, threshold and post-processing selection. Do not inspect the test set during tuning. Run the test evaluation only after the model and working point are frozen.

## Detection metrics

Report standard mAP50 and mAP50-95 together with per-class precision and recall. For very small particles, also report IoU@0.3 recall and center-hit recall. For diffuse large-area defects, report image-level presence metrics in addition to bounding-box metrics.

## Run manifest

For each formal experiment, retain locally:

- command line and resolved configuration;
- Python, PyTorch and Ultralytics versions;
- random seed and device;
- source data and split hashes;
- model checkpoint hash;
- validation or test designation.

These files may contain local paths or private data fingerprints and therefore are not committed to Git history. Authorized sanitized copies may be retained in the controlled private Release described in `PRIVATE_RELEASE.md`.
