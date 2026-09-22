# Local data layout

This directory is intentionally excluded from Git. Create the following folders locally as needed:

```text
data/
├── raw_original/     # Read-only source AOI images
├── images/           # Renamed copies
├── metadata/         # Manifests and patch metadata
├── annotations/      # annotations.json and CSV exports
├── derived/          # Splits, patches and detection datasets
└── patches/          # Optional patch exports
```

Do not commit real AOI images, annotation files, manifests or hashes. Use only data that you are authorized to process.

