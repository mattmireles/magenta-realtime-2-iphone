#!/usr/bin/env python3
"""Compute the frozen G3 signal, semantic, and known-bad-control metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from magenta_rt import MagentaRT2Mlxfn
from magenta_rt import audio as mrt_audio


AUDIO_RATE_HZ = 48_000
DECODE_CHUNK_FRAMES = 46_080
TAIL_SECONDS = 30


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
  left = np.asarray(lhs, dtype=np.float64).reshape(-1)
  right = np.asarray(rhs, dtype=np.float64).reshape(-1)
  return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))


def _load(path: Path) -> tuple[np.ndarray, int]:
  audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
  return audio, int(sample_rate)


def _signal_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
  finite = np.isfinite(audio)
  safe = np.nan_to_num(audio)
  peak = float(np.max(np.abs(safe))) if safe.size else 0.0
  left_right = (
      float(np.corrcoef(safe[:, 0], safe[:, 1])[0, 1])
      if safe.shape[1] == 2
      and float(np.std(safe[:, 0])) > 0
      and float(np.std(safe[:, 1])) > 0
      else 0.0
  )
  boundary_indices = np.arange(DECODE_CHUNK_FRAMES, safe.shape[0], DECODE_CHUNK_FRAMES)
  if boundary_indices.size:
    boundary_jumps = np.max(
        np.abs(safe[boundary_indices] - safe[boundary_indices - 1]), axis=1
    )
    max_boundary_jump = float(np.max(boundary_jumps) / max(peak, 1e-12))
  else:
    max_boundary_jump = 0.0

  mono = safe.mean(axis=1)
  tail = mono[-min(mono.size, TAIL_SECONDS * sample_rate) :]
  envelope_hop = sample_rate // 100
  envelope_frames = tail.size // envelope_hop
  envelope = np.sqrt(
      np.mean(
          tail[: envelope_frames * envelope_hop].reshape(envelope_frames, envelope_hop) ** 2,
          axis=1,
      )
  )
  envelope -= np.mean(envelope)
  spectrum = np.abs(np.fft.rfft(envelope)) ** 2
  frequencies = np.fft.rfftfreq(envelope.size, envelope_hop / sample_rate)
  pulse_share = float(
      spectrum[(frequencies >= 4) & (frequencies <= 16)].sum()
      / (spectrum[frequencies > 0].sum() + 1e-12)
  )

  is_near_zero = np.max(np.abs(safe), axis=1) < 1e-8
  longest_zero_run = 0
  current_zero_run = 0
  for is_zero in is_near_zero:
    current_zero_run = current_zero_run + 1 if is_zero else 0
    longest_zero_run = max(longest_zero_run, current_zero_run)
  residual = safe[:, 0] - safe[:, 1] if safe.shape[1] == 2 else safe[:, 0]
  residual_ratio = float(
      np.sqrt(np.mean(residual**2)) / (np.sqrt(np.mean(safe**2)) + 1e-12)
  )
  return {
      "sampleRateHz": sample_rate,
      "channels": int(safe.shape[1]),
      "channelOrder": ["left", "right"] if safe.shape[1] == 2 else ["unknown"],
      "durationSeconds": safe.shape[0] / sample_rate,
      "finiteRatio": float(np.mean(finite)),
      "clippedSampleRatio": float(np.mean(np.abs(safe) >= 0.9999)),
      "maxChunkBoundaryAbsJump": max_boundary_jump,
      "leftRightCorrelation": left_right,
      "envelopePulseShare4To16Hz": pulse_share,
      "longestNearZeroRunSeconds": longest_zero_run / sample_rate,
      "interchannelResidualRMSRatio": residual_ratio,
      "peakAmplitude": peak,
  }


def _rejection_reasons(metrics: dict[str, Any]) -> list[str]:
  reasons: list[str] = []
  checks = (
      (metrics["finiteRatio"] < 1.0, "nonfinite"),
      (metrics["clippedSampleRatio"] > 1e-5, "clipping"),
      (metrics["maxChunkBoundaryAbsJump"] > 0.07, "chunk-boundary"),
      (metrics["leftRightCorrelation"] < 0.97, "stereo-correlation"),
      (metrics["promptAdherence"] < 0.30, "prompt-adherence"),
      (metrics["envelopePulseShare4To16Hz"] > 0.07, "periodic-envelope"),
      (metrics["longestNearZeroRunSeconds"] >= 0.05, "dropout"),
      (metrics["interchannelResidualRMSRatio"] < 1e-4, "channel-collapse"),
  )
  for failed, reason in checks:
    if failed:
      reasons.append(reason)
  return reasons


def _parse_named_path(value: str) -> tuple[str, Path]:
  name, separator, raw_path = value.partition("=")
  if not separator or not name:
    raise argparse.ArgumentTypeError("expected NAME=PATH")
  return name, Path(raw_path)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument("--control", action="append", type=_parse_named_path, default=[])
  parser.add_argument("--prompt", default="warm ambient texture")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  candidate, candidate_rate = _load(args.candidate)
  reference, reference_rate = _load(args.reference)
  if candidate_rate != AUDIO_RATE_HZ or reference_rate != AUDIO_RATE_HZ:
    raise ValueError("candidate and reference must be 48 kHz")

  embedding_model = MagentaRT2Mlxfn(size="mrt2_small")
  text_embedding = np.asarray(embedding_model.embed_style(args.prompt, use_mapper=False))

  def semantic_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    tail = audio[-min(audio.shape[0], TAIL_SECONDS * sample_rate) :]
    return np.asarray(
        embedding_model.embed_style(
            mrt_audio.Waveform(tail, sample_rate), use_mapper=False
        )
    )

  reference_embedding = semantic_embedding(reference, reference_rate)
  candidate_embedding = semantic_embedding(candidate, candidate_rate)
  objective = _signal_metrics(candidate, candidate_rate)
  objective["promptAdherence"] = _cosine(text_embedding, candidate_embedding)
  objective["embeddingSimilarityToReference"] = _cosine(
      candidate_embedding, reference_embedding
  )

  controls = []
  for control_id, path in args.control:
    audio, sample_rate = _load(path)
    metrics = _signal_metrics(audio, sample_rate)
    metrics["promptAdherence"] = _cosine(
        text_embedding, semantic_embedding(audio, sample_rate)
    )
    reasons = _rejection_reasons(metrics)
    controls.append({
        "id": control_id,
        "rejected": bool(reasons),
        "rejectionReasons": reasons,
        "metrics": metrics,
        "sha256": _sha256(path),
    })

  report = {
      "schema": "mrt2-system-paper-audio-analysis-v1",
      "protocol": {
          "prompt": args.prompt,
          "semanticTailSeconds": TAIL_SECONDS,
          "decodeChunkFrames": DECODE_CHUNK_FRAMES,
          "envelope": "10 ms RMS; FFT power share from 4 to 16 Hz",
      },
      "candidate": {"sha256": _sha256(args.candidate), "path": str(args.candidate)},
      "reference": {"sha256": _sha256(args.reference), "path": str(args.reference)},
      "objectiveMetrics": objective,
      "knownBadControls": controls,
  }
  if not all(math.isfinite(float(value)) for value in objective.values() if isinstance(value, (int, float))):
    raise ValueError("objective metrics contain non-finite values")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
