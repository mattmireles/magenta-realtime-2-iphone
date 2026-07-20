# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify SpectroStream one-shot decode against frame-by-frame streaming decode.

This is the Phase 4 gate before decoder Core ML conversion. It uses the host CPU
RVQ lookup boundary, feeds the resulting ``[frames, 256]`` embeddings to the real
MLX SpectroStream decoder+iSTFT, and compares one-shot decoding to stepwise
decoding with the same layer state. The expected one-embedding decoder lookahead
means the first 1920-sample stepwise warmup frame is dropped before comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.flax as safetensors_flax
import soundfile as sf

from magenta_rt import paths
from magenta_rt.coreml.sampling import MRT2_RVQ_LEVELS, unique_token_to_raw_code


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIQUE_TOKENS_PATH = (
    REPO_ROOT / "Scratchpad" / "coreml_proof_fixtures" / "generated_tokens_unique.npy"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Scratchpad" / "coreml_proof_validation"
DEFAULT_REPORT_NAME = "spectrostream_streaming_decode_validation.json"
DEFAULT_SUMMARY_NAME = "spectrostream_streaming_decode_validation.md"
DEFAULT_ONESHOT_WAV_NAME = "spectrostream_decode_oneshot_aligned.wav"
DEFAULT_STREAMING_WAV_NAME = "spectrostream_decode_streaming_aligned.wav"
QUANTIZER_KEY = "params/soundstream/quantizer/embedding"
SPECTROSTREAM_SAMPLE_RATE = 48_000
SPECTROSTREAM_FRAME_SAMPLES = 1_920
STFT_FRAME_LENGTH = 960
STFT_FRAME_STEP = 480
STFT_FFT_LENGTH = 960


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
  """Return audio parity metrics for two same-shaped arrays."""
  delta = actual.astype(np.float64) - expected.astype(np.float64)
  rms_signal = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
  rms_error = float(np.sqrt(np.mean(delta ** 2)))
  snr_db = math.inf if rms_error == 0.0 else 20.0 * math.log10(rms_signal / rms_error)
  return {
      "shape": list(actual.shape),
      "max_abs_error": float(np.max(np.abs(delta))),
      "mean_abs_error": float(np.mean(np.abs(delta))),
      "rms_error": rms_error,
      "snr_db": snr_db,
      "log_spectral_distance_db": _log_spectral_distance(actual, expected),
  }


def _stft_magnitude(audio: np.ndarray) -> np.ndarray:
  """Return Hann-window STFT magnitudes for audio shaped ``[B, T, C]``."""
  audio64 = audio.astype(np.float64)
  window = np.hanning(STFT_FRAME_LENGTH)
  frames = []
  for batch in range(audio64.shape[0]):
    for channel in range(audio64.shape[2]):
      samples = audio64[batch, :, channel]
      channel_frames = []
      for start in range(0, samples.shape[0] - STFT_FRAME_LENGTH + 1, STFT_FRAME_STEP):
        frame = samples[start : start + STFT_FRAME_LENGTH] * window
        channel_frames.append(np.abs(np.fft.rfft(frame, n=STFT_FFT_LENGTH)))
      frames.append(np.stack(channel_frames, axis=0))
  return np.stack(frames, axis=0)


def _log_spectral_distance(actual: np.ndarray, expected: np.ndarray) -> float:
  """Compute RMS log-magnitude distance in dB between two PCM tensors."""
  actual_mag = _stft_magnitude(actual)
  expected_mag = _stft_magnitude(expected)
  delta_db = 20.0 * (
      np.log10(np.maximum(actual_mag, 1e-7))
      - np.log10(np.maximum(expected_mag, 1e-7))
  )
  return float(np.sqrt(np.mean(delta_db ** 2)))


def _load_decoder_codes(unique_tokens_path: Path, frames: int) -> np.ndarray:
  """Load unique fixture tokens and convert to SpectroStream decoder codes."""
  if not unique_tokens_path.exists():
    raise FileNotFoundError(
        f"Unique token fixture not found: {unique_tokens_path}. "
        "Run scripts/generate_mrt2_coreml_reference_fixtures.py first."
    )
  unique_tokens = np.load(unique_tokens_path).astype(np.int32)[:frames]
  if unique_tokens.ndim != 2 or unique_tokens.shape[1] != MRT2_RVQ_LEVELS:
    raise ValueError(f"Expected unique token shape [N, 12], got {unique_tokens.shape}")
  return unique_token_to_raw_code(unique_tokens).astype(np.int32)


def _load_codebooks(checkpoint_path: Path) -> np.ndarray:
  """Load SpectroStream quantizer embedding codebooks from the MRT2 checkpoint."""
  arrays = safetensors_flax.load_file(str(checkpoint_path))
  if QUANTIZER_KEY not in arrays:
    raise KeyError(f"Missing checkpoint key: {QUANTIZER_KEY}")
  return np.asarray(arrays[QUANTIZER_KEY], dtype=np.float32)


def _host_lookup_sum(codebooks: np.ndarray, decoder_codes: np.ndarray) -> np.ndarray:
  """Sum selected codebook rows on CPU for decoder codes shaped ``[T, 12]``."""
  levels = np.arange(decoder_codes.shape[1])[:, np.newaxis]
  selected = codebooks[levels, decoder_codes.T]
  return np.sum(selected, axis=0, dtype=np.float32)


def _build_spectrostream(checkpoint_path: Path):
  """Build and weight-load the MLX SpectroStream decoder."""
  import magenta_rt  # noqa: F401

  from magenta_rt.mlx import model
  from magenta_rt.mlx import spectrostream
  from magenta_rt.mlx.spectrostream.load_weights import load_spectrostream_weights

  exp = model.get_model_class("mrt2_small")()
  config = spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
      rvq_truncation_level=exp.spectrostream.rvq_truncation_level,
      use_unique_codes=False,
  )
  soundstream = config.make()
  load_spectrostream_weights(soundstream, checkpoint_path)
  return soundstream


def _decode_oneshot(soundstream, embeddings: np.ndarray) -> np.ndarray:
  """Decode all embeddings in one MLX SequenceLayers call."""
  import mlx.core as mx
  import sequence_layers.mlx as sl

  sequence = sl.Sequence(
      mx.array(embeddings[np.newaxis], dtype=mx.float32),
      mx.ones((1, embeddings.shape[0]), dtype=mx.bool_),
  )
  output = soundstream.embeddings_to_waveform_layer.layer(sequence)
  mx.eval(output.values)
  return np.asarray(output.values, dtype=np.float32)


def _decode_streaming(soundstream, embeddings: np.ndarray) -> np.ndarray:
  """Decode embeddings one frame at a time using persistent layer state."""
  import mlx.core as mx
  import sequence_layers.mlx as sl

  state = soundstream.embeddings_to_waveform_layer.get_initial_state(
      1,
      sl.ChannelSpec(shape=[embeddings.shape[-1]], dtype=mx.float32),
  )
  outputs = []
  for frame in embeddings:
    sequence = sl.Sequence(
        mx.array(frame.reshape(1, 1, -1), dtype=mx.float32),
        mx.ones((1, 1), dtype=mx.bool_),
    )
    output, state = soundstream.embeddings_to_waveform_layer.step(sequence, state)
    mx.eval(output.values)
    outputs.append(np.asarray(output.values, dtype=np.float32))
  return np.concatenate(outputs, axis=1)


def _boundary_jump_metrics(audio: np.ndarray) -> dict[str, Any]:
  """Measure sample discontinuities at 40 ms frame boundaries."""
  jumps = []
  for index in range(SPECTROSTREAM_FRAME_SAMPLES, audio.shape[1], SPECTROSTREAM_FRAME_SAMPLES):
    jumps.append(np.abs(audio[:, index] - audio[:, index - 1]))
  if not jumps:
    return {"boundary_count": 0, "max_abs_jump": None, "mean_abs_jump": None}
  values = np.stack(jumps, axis=0)
  return {
      "boundary_count": int(values.shape[0]),
      "max_abs_jump": float(np.max(values)),
      "mean_abs_jump": float(np.mean(values)),
  }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
  """Run one-shot vs streaming decode validation."""
  checkpoint_path = Path(args.checkpoint_path)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  decoder_codes = _load_decoder_codes(Path(args.unique_tokens_path), args.frames)
  codebooks = _load_codebooks(checkpoint_path)
  embeddings = _host_lookup_sum(codebooks, decoder_codes)
  soundstream = _build_spectrostream(checkpoint_path)

  oneshot = _decode_oneshot(soundstream, embeddings)
  streaming = _decode_streaming(soundstream, embeddings)
  lookahead_samples = streaming.shape[1] - oneshot.shape[1]
  if lookahead_samples < 0:
    raise ValueError(
        f"One-shot output is longer than streaming output: {oneshot.shape} vs {streaming.shape}"
    )
  aligned_streaming = streaming[:, lookahead_samples : lookahead_samples + oneshot.shape[1]]
  if aligned_streaming.shape != oneshot.shape:
    raise ValueError(
        "Aligned streaming shape does not match one-shot shape: "
        f"{aligned_streaming.shape} vs {oneshot.shape}"
    )

  oneshot_wav_path = output_dir / args.oneshot_wav_name
  streaming_wav_path = output_dir / args.streaming_wav_name
  sf.write(oneshot_wav_path, oneshot[0], SPECTROSTREAM_SAMPLE_RATE)
  sf.write(streaming_wav_path, aligned_streaming[0], SPECTROSTREAM_SAMPLE_RATE)

  return {
      "schema": "spectrostream-streaming-decode-validation-v1",
      "checkpoint_path": str(checkpoint_path),
      "frames": int(decoder_codes.shape[0]),
      "decoder_codes": {
          "shape": list(decoder_codes.shape),
          "min": int(np.min(decoder_codes)),
          "max": int(np.max(decoder_codes)),
      },
      "embeddings": {
          "shape": list(embeddings.shape),
          "source": "host CPU RVQ lookup sum from first 12 SpectroStream codebooks",
      },
      "oneshot": {
          "shape": list(oneshot.shape),
      },
      "streaming": {
          "raw_shape": list(streaming.shape),
          "aligned_shape": list(aligned_streaming.shape),
          "dropped_warmup_samples": int(lookahead_samples),
          "dropped_warmup_frames": float(lookahead_samples / SPECTROSTREAM_FRAME_SAMPLES),
      },
      "streaming_vs_oneshot": _metrics(aligned_streaming, oneshot),
      "boundary_jumps": {
          "oneshot": _boundary_jump_metrics(oneshot),
          "streaming_aligned": _boundary_jump_metrics(aligned_streaming),
      },
      "artifacts": {
          "oneshot_wav": str(oneshot_wav_path),
          "streaming_aligned_wav": str(streaming_wav_path),
      },
      "known_limits": [
          "This validates the MLX decoder+iSTFT streaming state, not Core ML decoder speed.",
          "The first 1920-sample stepwise warmup frame is dropped to align one-shot output.",
          "Generated WAVs are validation artifacts under Scratchpad, not tracked assets.",
      ],
  }


def _write_summary(report: dict[str, Any], summary_path: Path) -> None:
  """Write a short markdown summary beside the JSON report."""
  metrics = report["streaming_vs_oneshot"]
  lines = [
      "# SpectroStream Streaming Decode Validation",
      "",
      f"- Frames: {report['frames']}",
      f"- Embedding shape: `{report['embeddings']['shape']}`",
      f"- One-shot output shape: `{report['oneshot']['shape']}`",
      f"- Streaming raw output shape: `{report['streaming']['raw_shape']}`",
      f"- Dropped warmup samples: {report['streaming']['dropped_warmup_samples']}",
      f"- Streaming vs one-shot max error: {metrics['max_abs_error']:.10f}",
      f"- Streaming vs one-shot mean error: {metrics['mean_abs_error']:.10f}",
      f"- Streaming vs one-shot SNR: {metrics['snr_db']:.3f} dB",
      f"- Streaming vs one-shot log-spectral distance: {metrics['log_spectral_distance_db']:.6f} dB",
      f"- One-shot WAV: `{report['artifacts']['oneshot_wav']}`",
      f"- Streaming WAV: `{report['artifacts']['streaming_aligned_wav']}`",
      "",
      "Known limits:",
  ]
  lines.extend(f"- {limit}" for limit in report["known_limits"])
  lines.append("")
  summary_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
  """Parse CLI flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--checkpoint-path",
      default=str(paths.checkpoints_dir() / "mrt2_small.safetensors"),
  )
  parser.add_argument("--unique-tokens-path", default=str(DEFAULT_UNIQUE_TOKENS_PATH))
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
  parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
  parser.add_argument("--oneshot-wav-name", default=DEFAULT_ONESHOT_WAV_NAME)
  parser.add_argument("--streaming-wav-name", default=DEFAULT_STREAMING_WAV_NAME)
  parser.add_argument("--frames", type=int, default=25)
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  args = parse_args()
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  report = run_validation(args)
  report_path = output_dir / args.report_name
  summary_path = output_dir / args.summary_name
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  _write_summary(report, summary_path)
  print(f"Wrote {report_path}")
  print(f"Wrote {summary_path}")
  print(
      "Streaming vs one-shot max error "
      f"{report['streaming_vs_oneshot']['max_abs_error']:.10f}"
  )


if __name__ == "__main__":
  main()
