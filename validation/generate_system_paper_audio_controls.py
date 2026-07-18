#!/usr/bin/env python3
"""Generate deterministic channel-collapse and dropout audio controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("candidate", type=Path)
  parser.add_argument("output_dir", type=Path)
  parser.add_argument("--seconds", type=float, default=30.0)
  args = parser.parse_args()
  audio, sample_rate = sf.read(args.candidate, dtype="float32", always_2d=True)
  if audio.shape[1] != 2:
    raise ValueError("candidate must be stereo")
  frames = min(audio.shape[0], int(args.seconds * sample_rate))
  source = audio[-frames:].copy()
  args.output_dir.mkdir(parents=True, exist_ok=True)

  mono = source.mean(axis=1, keepdims=True)
  collapsed = np.repeat(mono, 2, axis=1)
  sf.write(args.output_dir / "channel-collapse.wav", collapsed, sample_rate, subtype="FLOAT")

  dropout = source.copy()
  dropout_frames = int(0.1 * sample_rate)
  period_frames = int(2.0 * sample_rate)
  for start in range(period_frames, frames, period_frames):
    dropout[start : start + dropout_frames] = 0.0
  sf.write(args.output_dir / "dropout-injection.wav", dropout, sample_rate, subtype="FLOAT")
  print(f"Wrote controls to {args.output_dir}")


if __name__ == "__main__":
  main()
