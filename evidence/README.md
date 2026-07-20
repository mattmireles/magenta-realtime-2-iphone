# Evidence bundle

This directory does not contain the raw evidence itself — 11 GB of WAV, token,
and trace files do not belong in a git repository. It contains the exact,
hash-verified manifest of what backs every quantitative claim in the paper,
and the script that publishes it.

- `evidence-manifest.json` — 139 files, each with its source path (on the
  authoring machine), destination path in the public dataset, sha256, size,
  and which `validation/results/system-paper/*.json` receipt it backs. Every
  hash in this file matches a hash already committed in this repository —
  nothing here is a new claim, it is a pointer to raw bytes for the claims
  already made.
- `publish_to_huggingface.py` — re-verifies every file against its recorded
  hash, then uploads the bundle to a Hugging Face dataset. Run it after
  `hf auth login`; it never reads, stores, or transmits your token itself.

## Public dataset

The evidence bundle is published at:
https://huggingface.co/datasets/mattmireles/mrt2-three-clocks-evidence

## Reproducing the verification yourself

```
sha256sum <(curl -sL https://huggingface.co/datasets/mattmireles/mrt2-three-clocks-evidence/resolve/main/context12/a17-context12-soak.wav)
```

...should print the same hash recorded in
`validation/results/system-paper/a17pro/context12/context12-soak-manifest.json`
under `artifacts.wav.sha256`. The same pattern holds for every file in the
manifest.

## What stays private

- The signed, compiled iOS application binary (the shipping Crossfade
  product). Its hash is recorded (`signedExecutableSha256` / `executable`)
  but the binary itself is not published.
- Exploratory pilot/smoke runs that never became a numbered paper result.
- The single blinded human engineering listening check (not machine-hash-bound
  in any manifest; reported only as prose in the paper's limitations).
