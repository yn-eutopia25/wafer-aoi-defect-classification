# Project-specific rules

- `data/raw_original/`, `data/images/`, `data/annotations/` and all `backups/` directories contain private or historical material. Automation must not modify or commit them unless the user explicitly requests a local data correction.
- `data/annotations/annotations.json` is the only authoritative manual annotation source. CSV files are derived exports.
- Split all boxes, patches and tiles by source `image_id`. Do not allow one source image to cross train, validation and test sets.
- Do not regenerate or tune against the locked test split. Final test scripts require explicit confirmation and produce a lock record.
- Git history contains code and reproducibility documentation only. When the user explicitly authorizes it, normalized private images, current annotations, trained models and experiment outputs may be distributed as assets of a private GitHub Release. Never place those artifacts in Git history, and never include `backups/`, raw-original filenames, credentials or personal office deliverables.
