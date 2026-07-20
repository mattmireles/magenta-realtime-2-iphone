#!/usr/bin/env python3
"""Upload the MRT2 Three Clocks system-paper evidence bundle to a Hugging Face dataset.

Usage:
    hf auth login   # one-time, run by a human; this script never touches your token
    python3 evidence/publish_to_huggingface.py --repo-id <your-hf-username>/mrt2-three-clocks-evidence

Every file this script uploads is listed in evidence-manifest.json together with the
sha256 that validation/results/system-paper/*.json already publishes. Before uploading,
this script re-hashes each local file and refuses to proceed if anything has drifted
from the receipt it is supposed to back — the public paper repo's hashes remain the
ground truth.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

DATASET_README = """---
license: cc-by-4.0
---

# MRT2 Three Clocks — System Paper Evidence

Raw device evidence backing *Throughput Is Not Liveness: Three Clocks for
GPU-Free Music Generation on iPhone* (Mireles, 2026).

Every file here is hash-bound to a machine-readable receipt in the public
paper repository: https://github.com/mattmireles/magenta-realtime-2-iphone
under `validation/results/system-paper/`. Re-hash any file in this dataset
with `sha256sum` and compare against the corresponding manifest to verify it
has not changed since publication.

## Layout

- `context12/` — the corrected A17 Pro sustained-throughput run (610 s foreground
  capture): PCM WAV, event trace, token capture, and summaries. Backs
  `a17pro/context12/context12-soak-manifest.json`.
- `crossover/` — three seeds (20260718, 271828, 1618033) of the 2x2 token-source
  x decoder-path crossover, plus the FLOAT32 pre-iSTFT split, corrected-DSP
  controls, and 12-frame context arms. Backs `crossover/seed-*.json` and
  `crossover/aggregate.json`. `decoder-context-probe-tokens.npy` is the token
  file underlying `crossover/decoder-context-probe.json`.
- `liveness/reset-factorial/` — all 12 arms (3 seeds x {off, kv-only,
  feedback-only, both}) of the frozen unrefreshed-liveness gate: WAV plus
  codebook-local token capture per arm. Backs `liveness/g5-manifest.json`.
- `liveness/judge-events/` — the 9 event-centered model-vote lineups (sealed
  clip mapping + vote result + the three candidate/baseline/corruption-control
  clips) that back the listening checks in `g5-manifest.json`. These are
  automated model votes, not human-subject data.
- `steering/` — the final post-ring steering run: WAV, event trace, token
  capture, paired-latency detector report, and device launch logs. Backs
  `steering/g6-manifest.json`.

## What is intentionally not included

- The signed, compiled iOS application binary (its hash is recorded in the
  manifests as `signedExecutableSha256` / `executable`, but the binary itself
  is the shipping product and stays private).
- Exploratory pilot/smoke runs and abandoned configurations that never became
  a numbered result in the paper.
- The single blinded human engineering listening check: it is not
  machine-hash-bound in any manifest and is reported only as prose in the
  paper's limitations.

## License

The underlying `mrt2_small` checkpoint is CC-BY-4.0 (Google DeepMind). Audio
and token captures derived from it are released under the same license.
"""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. mattmireles/mrt2-three-clocks-evidence")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parent / "evidence-manifest.json"),
        help="Path to evidence-manifest.json",
    )
    parser.add_argument("--private", action="store_true", help="Create the dataset as private (default: public)")
    parser.add_argument("--dry-run", action="store_true", help="Verify hashes and print the plan without uploading")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest["entries"]

    print(f"Verifying {len(entries)} files against their recorded sha256 before upload...")
    total_bytes = 0
    for i, entry in enumerate(entries, 1):
        src = Path(entry["source"])
        if not src.is_file():
            print(f"MISSING: {src} (expected at {entry['dest']})", file=sys.stderr)
            return 1
        actual = sha256_of(src)
        if actual != entry["sha256"]:
            print(f"HASH MISMATCH: {src}\n  expected {entry['sha256']}\n  actual   {actual}", file=sys.stderr)
            print("Refusing to upload a file that no longer matches the published receipt.", file=sys.stderr)
            return 1
        total_bytes += entry["size"]
        if i % 25 == 0 or i == len(entries):
            print(f"  verified {i}/{len(entries)}")
    print(f"All {len(entries)} files verified. Total size: {total_bytes / 1e9:.2f} GB.")

    if args.dry_run:
        print("Dry run only; nothing uploaded.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Creating (or reusing) dataset repo {args.repo_id} ...")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    readme_path = Path(__file__).parent / ".dataset-readme-staged.md"
    readme_path.write_text(DATASET_README)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    readme_path.unlink()

    for i, entry in enumerate(entries, 1):
        api.upload_file(
            path_or_fileobj=entry["source"],
            path_in_repo=entry["dest"],
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"  uploaded {i}/{len(entries)}: {entry['dest']}")

    print(f"Done. Dataset live at https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
