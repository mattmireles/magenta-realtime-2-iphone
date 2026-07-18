#!/usr/bin/env python3
"""Build five frozen, neutral-label audio lineups for the G3 blind vote."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


AUDIO_RATE_HZ = 48_000
HEAD_GUARD_SECONDS = 2.0
TAIL_GUARD_SECONDS = 0.5
EXCERPT_SECONDS = 24.0
DEFAULT_SEEDS = (1729, 31415, 271828, 1618033, 8675309)
ROLES = ("reference", "candidate", "known-bad")
LABELS = ("sample-a", "sample-b", "sample-c")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load(path: Path) -> np.ndarray:
  audio, rate = sf.read(path, dtype="float32", always_2d=True)
  if rate != AUDIO_RATE_HZ or audio.shape[1] != 2:
    raise ValueError(f"{path} must be 48 kHz stereo")
  minimum = int((HEAD_GUARD_SECONDS + EXCERPT_SECONDS + TAIL_GUARD_SECONDS) * rate)
  if audio.shape[0] < minimum:
    raise ValueError(f"{path} is too short for a {EXCERPT_SECONDS:g} s excerpt")
  return audio


def _excerpt(audio: np.ndarray, rng: random.Random) -> tuple[np.ndarray, float]:
  first = int(HEAD_GUARD_SECONDS * AUDIO_RATE_HZ)
  last = audio.shape[0] - int(
      (EXCERPT_SECONDS + TAIL_GUARD_SECONDS) * AUDIO_RATE_HZ
  )
  start = rng.randint(first, last)
  frames = int(EXCERPT_SECONDS * AUDIO_RATE_HZ)
  return audio[start : start + frames].copy(), start / AUDIO_RATE_HZ


def _rms(audio: np.ndarray) -> float:
  return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def build(args: argparse.Namespace) -> dict[str, Any]:
  sources = {
      "reference": args.reference,
      "candidate": args.candidate,
      "known-bad": args.known_bad,
  }
  audio = {role: _load(path) for role, path in sources.items()}
  lineups = []
  for seed in args.seeds:
    rng = random.Random(seed)
    excerpts: dict[str, np.ndarray] = {}
    starts: dict[str, float] = {}
    for role in ROLES:
      excerpts[role], starts[role] = _excerpt(audio[role], rng)

    target_rms = _rms(excerpts["reference"])
    gain = {
        role: target_rms / max(_rms(excerpts[role]), 1e-12) for role in ROLES
    }
    matched = {role: excerpts[role] * gain[role] for role in ROLES}
    shared_attenuation = min(
        1.0,
        0.98
        / max(float(np.max(np.abs(value))) for value in matched.values()),
    )

    labels = list(LABELS)
    rng.shuffle(labels)
    role_to_label = dict(zip(ROLES, labels, strict=True))
    upload_order = list(ROLES)
    rng.shuffle(upload_order)
    lineup_dir = args.output_dir / f"seed-{seed}"
    lineup_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for role in ROLES:
      label = role_to_label[role]
      path = lineup_dir / f"{label}.wav"
      rendered = np.clip(matched[role] * shared_attenuation, -1.0, 1.0)
      sf.write(path, rendered, AUDIO_RATE_HZ, subtype="PCM_16")
      outputs[role] = {
          "label": label,
          "sha256": _sha256(path),
          "sourceSha256": _sha256(sources[role]),
          "excerptStartSeconds": starts[role],
          "excerptDurationSeconds": EXCERPT_SECONDS,
          "gainBeforeSharedAttenuation": gain[role],
      }
    lineups.append({
        "seed": seed,
        "baselineLabel": role_to_label["reference"],
        "uploadOrder": [role_to_label[role] for role in upload_order],
        "roleToLabel": role_to_label,
        "sharedAttenuation": shared_attenuation,
        "clips": outputs,
    })

  manifest = {
      "schema": "mrt2-system-paper-listening-lineups-v1",
      "protocol": {
          "sampleRateHz": AUDIO_RATE_HZ,
          "channels": 2,
          "headGuardSeconds": HEAD_GUARD_SECONDS,
          "tailGuardSeconds": TAIL_GUARD_SECONDS,
          "excerptSeconds": EXCERPT_SECONDS,
          "gainMatching": "RMS to reference, then shared peak attenuation",
          "neutralLabels": list(LABELS),
      },
      "seeds": list(args.seeds),
      "lineups": lineups,
  }
  manifest_path = args.output_dir / "lineups-manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  return manifest


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument("--known-bad", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--seeds",
      type=lambda value: tuple(int(item) for item in value.split(",")),
      default=DEFAULT_SEEDS,
  )
  args = parser.parse_args()
  build(args)
  print(f"Wrote {args.output_dir / 'lineups-manifest.json'}")


if __name__ == "__main__":
  main()
