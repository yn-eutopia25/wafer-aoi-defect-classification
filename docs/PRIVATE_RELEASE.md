# Private data release

The Git repository stores source code and documentation only. Non-public project artifacts are distributed separately through the private GitHub Release tagged `private-data-v1.0.0`.

## Assets

| Asset | Contents |
|---|---|
| `aoi-private-images-v1.0.0.zip` | 200 real AOI NG images renamed as `AOI_NG_####` |
| `aoi-private-annotations-v1.0.0.zip` | Authoritative `annotations.json`, derived instance and image tables, and label schema |
| `aoi-private-models-v1.0.0.zip` | Formal detector, patch-classifier and hard-negative-guard artifacts |
| `aoi-private-experiments-v1.0.0.zip` | Derived datasets, figures, evaluation tables, HTML reports and training-run records |
| `RELEASE_MANIFEST.csv` | Archive sizes and SHA-256 hashes |
| `SHA256SUMS.txt` | Command-line integrity-check list |

The release excludes `data/raw_original/`, every `backups/` directory, virtual environments, temporary authoring workspaces, personal office deliverables and access credentials. Local absolute paths in experiment metadata are replaced in the release copy; local source files are not modified.

## Integrity check

After downloading the assets, compare their SHA-256 values with `SHA256SUMS.txt`.

```powershell
Get-FileHash .\aoi-private-images-v1.0.0.zip -Algorithm SHA256
Get-FileHash .\aoi-private-annotations-v1.0.0.zip -Algorithm SHA256
Get-FileHash .\aoi-private-models-v1.0.0.zip -Algorithm SHA256
Get-FileHash .\aoi-private-experiments-v1.0.0.zip -Algorithm SHA256
```

Each archive also contains an internal `MANIFEST.csv` with the relative path, byte size and SHA-256 hash of every included file.

## Access boundary

These assets are non-public and intended only for authorized project use. Do not mirror them to a public repository or detach them from the project context.
