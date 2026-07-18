#!/usr/bin/env python3
"""Analyze one 600 s MRT2 token-by-decoder crossover replication.

The four WAVs form a 2x2 factorial design: MLX or Core ML generated tokens,
then MLX or Core ML plus the production DSP decoded those fixed tokens.  This
script deliberately reports effects and time series rather than assigning a
single causal label from one threshold crossing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SAMPLE_RATE_HZ = 48_000
ENVELOPE_HZ = 100
WINDOW_SECONDS = 30
PULSE_LIMIT = 0.07
DECODE_CHUNK_FRAMES = 46_080


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _pulse_share(envelope: np.ndarray) -> float:
  centered = np.asarray(envelope, dtype=np.float64) - float(np.mean(envelope))
  power = np.abs(np.fft.rfft(centered)) ** 2
  frequencies = np.fft.rfftfreq(centered.size, d=1.0 / ENVELOPE_HZ)
  positive = frequencies > 0
  band = (frequencies >= 4.0) & (frequencies <= 16.0)
  denominator = float(power[positive].sum())
  return float(power[band].sum() / denominator) if denominator > 0 else 0.0


def _longest_true_run(values: np.ndarray) -> int:
  indices = np.flatnonzero(np.diff(np.r_[False, values, False]))
  return int(np.max(indices[1::2] - indices[::2], initial=0))


def analyze_wav(path: Path) -> dict[str, Any]:
  audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
  if sample_rate != SAMPLE_RATE_HZ or audio.shape[1] != 2:
    raise ValueError(f"{path} must be 48 kHz stereo")
  safe = np.nan_to_num(audio)
  mono = safe.mean(axis=1)
  hop = sample_rate // ENVELOPE_HZ
  envelope_frames = mono.size // hop
  envelope = np.sqrt(
      np.mean(mono[: envelope_frames * hop].reshape(envelope_frames, hop) ** 2, axis=1)
  )
  window_envelopes = WINDOW_SECONDS * ENVELOPE_HZ
  windows = []
  for start in range(0, envelope.size, window_envelopes):
    chunk = envelope[start : start + window_envelopes]
    if chunk.size != window_envelopes:
      continue
    pulse = _pulse_share(chunk)
    windows.append({
        "startSeconds": start / ENVELOPE_HZ,
        "endSeconds": (start + chunk.size) / ENVELOPE_HZ,
        "envelopePulseShare4To16Hz": pulse,
        "exceedsFrozenLimit": pulse > PULSE_LIMIT,
    })
  boundary_indices = np.arange(DECODE_CHUNK_FRAMES, safe.shape[0], DECODE_CHUNK_FRAMES)
  peak = float(np.max(np.abs(safe))) if safe.size else 0.0
  boundary_jump = (
      float(np.max(np.max(np.abs(safe[boundary_indices] - safe[boundary_indices - 1]), axis=1))
            / max(peak, 1e-12))
      if boundary_indices.size else 0.0
  )
  near_zero = np.max(np.abs(safe), axis=1) < 1e-8
  pulses = [row["envelopePulseShare4To16Hz"] for row in windows]
  return {
      "path": str(path),
      "sha256": _sha256(path),
      "durationSeconds": safe.shape[0] / sample_rate,
      "finiteRatio": float(np.isfinite(audio).mean()),
      "clippedSampleRatio": float(np.mean(np.abs(safe) >= 0.9999)),
      "maxChunkBoundaryAbsJump": boundary_jump,
      "longestNearZeroRunSeconds": _longest_true_run(near_zero) / sample_rate,
      "tailPulseShare4To16Hz": pulses[-1],
      "medianWindowPulseShare4To16Hz": float(np.median(pulses)),
      "maxWindowPulseShare4To16Hz": float(np.max(pulses)),
      "windowsOverFrozenLimit": int(sum(value > PULSE_LIMIT for value in pulses)),
      "windowCount": len(windows),
      "windows": windows,
  }


def _read_summary(path: Path) -> dict[str, Any]:
  report = json.loads(path.read_text())
  windows = [row for row in report["windows"] if row["frames"] == 750]
  return {
      "path": str(path),
      "sha256": _sha256(path),
      "frames": int(report["frames"]),
      "completeWindows": len(windows),
      "firstWindow": windows[0],
      "lastWindow": windows[-1],
  }


def build(args: argparse.Namespace) -> dict[str, Any]:
  cells = {
      "mlxTokens_mlxDecoder": analyze_wav(args.mlx_mlx_wav),
      "mlxTokens_coremlDecoder": analyze_wav(args.mlx_coreml_wav),
      "coremlTokens_mlxDecoder": analyze_wav(args.coreml_mlx_wav),
      "coremlTokens_coremlDecoder": analyze_wav(args.coreml_coreml_wav),
  }
  if args.mlx_stft_core_dsp_wav is not None:
    cells["mlxTokens_mlxStftProductionDsp"] = analyze_wav(
        args.mlx_stft_core_dsp_wav
    )
  fixed_paths = {
      "mlxTokens_mlxStftFixedDsp": args.mlx_stft_fixed_dsp_wav,
      "mlxTokens_coremlFixedDsp": args.mlx_coreml_fixed_dsp_wav,
      "coremlTokens_mlxStftFixedDsp": args.coreml_mlx_stft_fixed_dsp_wav,
      "coremlTokens_coremlFixedDsp": args.coreml_coreml_fixed_dsp_wav,
  }
  supplied_fixed = [path is not None for path in fixed_paths.values()]
  if any(supplied_fixed) and not all(supplied_fixed):
    raise ValueError("all four fixed-DSP WAVs must be supplied together")
  fixed_cells = (
      {name: analyze_wav(path) for name, path in fixed_paths.items()}
      if all(supplied_fixed)
      else None
  )
  mm = cells["mlxTokens_mlxDecoder"]["tailPulseShare4To16Hz"]
  mc = cells["mlxTokens_coremlDecoder"]["tailPulseShare4To16Hz"]
  cm = cells["coremlTokens_mlxDecoder"]["tailPulseShare4To16Hz"]
  cc = cells["coremlTokens_coremlDecoder"]["tailPulseShare4To16Hz"]
  window_pulses = {
      name: np.asarray(
          [row["envelopePulseShare4To16Hz"] for row in cell["windows"]],
          dtype=np.float64,
      )
      for name, cell in cells.items()
  }
  token_effects = (
      window_pulses["coremlTokens_mlxDecoder"]
      - window_pulses["mlxTokens_mlxDecoder"]
  )
  decoder_effects = (
      window_pulses["mlxTokens_coremlDecoder"]
      - window_pulses["mlxTokens_mlxDecoder"]
  )
  interactions = (
      window_pulses["coremlTokens_coremlDecoder"]
      - window_pulses["coremlTokens_mlxDecoder"]
      - window_pulses["mlxTokens_coremlDecoder"]
      + window_pulses["mlxTokens_mlxDecoder"]
  )

  def summarize_effect(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "positiveWindows": int(np.sum(values > 0)),
        "windows": int(values.size),
    }

  h3_split = None
  if "mlxTokens_mlxStftProductionDsp" in window_pulses:
    batching_dsp = (
        window_pulses["mlxTokens_mlxStftProductionDsp"]
        - window_pulses["mlxTokens_mlxDecoder"]
    )
    coreml_graph = (
        window_pulses["mlxTokens_coremlDecoder"]
        - window_pulses["mlxTokens_mlxStftProductionDsp"]
    )
    h3_split = {
        "mlxWindowingAndProductionDspVsStreamingMlx": summarize_effect(batching_dsp),
        "coremlGraphVsMlxGraphAtProductionDsp": summarize_effect(coreml_graph),
    }

  fixed_dsp_effects = None
  if fixed_cells is not None:
    fixed_pulses = {
        name: np.asarray(
            [row["envelopePulseShare4To16Hz"] for row in cell["windows"]],
            dtype=np.float64,
        )
        for name, cell in fixed_cells.items()
    }
    fixed_dsp_effects = {
        "fixOnMlxTokensMlxGraph": summarize_effect(
            fixed_pulses["mlxTokens_mlxStftFixedDsp"]
            - window_pulses["mlxTokens_mlxStftProductionDsp"]
        ),
        "fixOnMlxTokensCoremlGraph": summarize_effect(
            fixed_pulses["mlxTokens_coremlFixedDsp"]
            - window_pulses["mlxTokens_coremlDecoder"]
        ),
        "coremlGraphVsMlxGraphAtFixedDsp": summarize_effect(
            fixed_pulses["mlxTokens_coremlFixedDsp"]
            - fixed_pulses["mlxTokens_mlxStftFixedDsp"]
        ),
    }

  context_paths = {
      "mlxTokens_mlxContext12FixedDsp": args.mlx_context12_fixed_dsp_wav,
      "mlxTokens_coremlContext12FixedDsp": args.coreml_context12_fixed_dsp_wav,
  }
  supplied_context = [path is not None for path in context_paths.values()]
  if any(supplied_context) and not all(supplied_context):
    raise ValueError("both 12-frame-context WAVs must be supplied together")
  context_cells = (
      {name: analyze_wav(path) for name, path in context_paths.items()}
      if all(supplied_context)
      else None
  )
  if context_cells is not None and fixed_cells is None:
    raise ValueError("12-frame-context WAVs require the four fixed-DSP controls")
  context_effects = None
  if context_cells is not None:
    context_pulses = {
        name: np.asarray(
            [row["envelopePulseShare4To16Hz"] for row in cell["windows"]],
            dtype=np.float64,
        )
        for name, cell in context_cells.items()
    }
    streaming = window_pulses["mlxTokens_mlxDecoder"]
    context_effects = {
        "mlxContext12VsStreamingMlx": summarize_effect(
            context_pulses["mlxTokens_mlxContext12FixedDsp"] - streaming
        ),
        "coremlContext12VsStreamingMlx": summarize_effect(
            context_pulses["mlxTokens_coremlContext12FixedDsp"] - streaming
        ),
        "coremlGraphVsMlxGraphAtContext12": summarize_effect(
            context_pulses["mlxTokens_coremlContext12FixedDsp"]
            - context_pulses["mlxTokens_mlxContext12FixedDsp"]
        ),
        "mlxContext12VsNoContextAtFixedDsp": summarize_effect(
            context_pulses["mlxTokens_mlxContext12FixedDsp"]
            - fixed_pulses["mlxTokens_mlxStftFixedDsp"]
        ),
        "coremlContext12VsNoContextAtFixedDsp": summarize_effect(
            context_pulses["mlxTokens_coremlContext12FixedDsp"]
            - fixed_pulses["mlxTokens_coremlFixedDsp"]
        ),
    }

  return {
      "schema": "mrt2-long-horizon-crossover-analysis-v1",
      "seed": int(args.seed),
      "protocol": {
          "prompt": args.prompt,
          "durationSeconds": 600,
          "temperature": 1.0,
          "topK": 40,
          "trajectoryRefreshSeconds": 10.0,
          "pulseWindowSeconds": WINDOW_SECONDS,
          "frozenPulseLimit": PULSE_LIMIT,
      },
      "tokens": {
          "mlx": _read_summary(args.mlx_token_summary),
          "coremlPort": _read_summary(args.coreml_token_summary),
      },
      "cells": cells,
      "fixedDspCells": fixed_cells,
      "context12Cells": context_cells,
      "tailPulseFactorialEffects": {
          "coremlTokenEffectAtMlxDecoder": cm - mm,
          "coremlDecoderEffectOnMlxTokens": mc - mm,
          "interaction": cc - cm - mc + mm,
      },
      "windowPulseFactorialEffects": {
          "coremlTokenEffectAtMlxDecoder": summarize_effect(token_effects),
          "coremlDecoderEffectOnMlxTokens": summarize_effect(decoder_effects),
          "interaction": summarize_effect(interactions),
      },
      "h3SplitWindowPulseEffects": h3_split,
      "fixedDspWindowPulseEffects": fixed_dsp_effects,
      "context12WindowPulseEffects": context_effects,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--prompt", default="warm ambient texture")
  parser.add_argument("--mlx-token-summary", type=Path, required=True)
  parser.add_argument("--coreml-token-summary", type=Path, required=True)
  parser.add_argument("--mlx-mlx-wav", type=Path, required=True)
  parser.add_argument("--mlx-coreml-wav", type=Path, required=True)
  parser.add_argument("--coreml-mlx-wav", type=Path, required=True)
  parser.add_argument("--coreml-coreml-wav", type=Path, required=True)
  parser.add_argument("--mlx-stft-core-dsp-wav", type=Path)
  parser.add_argument("--mlx-stft-fixed-dsp-wav", type=Path)
  parser.add_argument("--mlx-coreml-fixed-dsp-wav", type=Path)
  parser.add_argument("--coreml-mlx-stft-fixed-dsp-wav", type=Path)
  parser.add_argument("--coreml-coreml-fixed-dsp-wav", type=Path)
  parser.add_argument("--mlx-context12-fixed-dsp-wav", type=Path)
  parser.add_argument("--coreml-context12-fixed-dsp-wav", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = build(args)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
