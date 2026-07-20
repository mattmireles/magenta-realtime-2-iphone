# Evidence bundle

This directory does not contain the raw evidence itself — 11 GB of WAV, token,
and trace files do not belong in a git repository. It contains the exact,
hash-verified manifest of what backs every quantitative claim in the paper,
the record of what was published where, and the script for an additional
mirror.

- `evidence-manifest.json` — 139 files, each with its source path (on the
  authoring machine), destination path, sha256, size, and which
  `validation/results/system-paper/*.json` receipt it backs. Every hash in
  this file matches a hash already committed in this repository — nothing
  here is a new claim, it is a pointer to raw bytes for the claims already
  made.
- `release-asset-map.json` — maps each manifest entry to its flattened
  GitHub Release asset name (`/` replaced with `__`, since release assets
  are a flat namespace).
- `release-upload.log` — the upload transcript: 139/139 uploaded, 0 failures,
  verified against the release's asset count and spot-checked (including the
  largest file) by downloading and re-hashing after upload.
- `publish_to_huggingface.py` — re-verifies every file against its recorded
  hash, then uploads the same bundle to a Hugging Face dataset as an
  additional mirror. Run it after `hf auth login`; it never reads, stores, or
  transmits your token itself. Not required — the GitHub Release below is
  already the live, verified public copy.

## Public location

The evidence bundle is published at:
https://github.com/mattmireles/magenta-realtime-2-iphone/releases/tag/evidence-v1

## Reproducing the verification yourself

```
curl -sL -o f.wav "https://github.com/mattmireles/magenta-realtime-2-iphone/releases/download/evidence-v1/context12__a17-context12-soak.wav"
sha256sum f.wav
```

...should print the same hash recorded in
`validation/results/system-paper/a17pro/context12/context12-soak-manifest.json`
under `artifacts.wav.sha256`. The same pattern (asset name = `dest` from
`evidence-manifest.json` with `/` replaced by `__`) holds for every file.

## What stays private

- The signed, compiled iOS application binary (the shipping Crossfade
  product). Its hash is recorded (`signedExecutableSha256` / `executable`)
  but the binary itself is not published.
- Exploratory pilot/smoke runs that never became a numbered paper result.
- The single blinded human engineering listening check (not machine-hash-bound
  in any manifest; reported only as prose in the paper's limitations).
