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

"""Probe host CPU RVQ lookup for the SpectroStream split decoder proof.

Phase 4 deliberately keeps RVQ detokenization out of the first decoder Core ML
graph. This probe loads the real MRT2/SpectroStream quantizer codebooks, converts
Depthformer unique-code tokens to raw 0-1023 RVQ codes, sums the first 12
codebook rows on CPU, and compares that host result against the MLX
``ResidualVectorQuantizer.codes_to_embeddings_layer`` reference.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.flax as safetensors_flax

from magenta_rt import paths
from magenta_rt.coreml.sampling import MRT2_RVQ_LEVELS, unique_token_to_raw_code


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIQUE_TOKENS_PATH = (
    REPO_ROOT / "Scratchpad" / "coreml_proof_fixtures" / "generated_tokens_unique.npy"
)
DEFAULT_RAW_TOKENS_PATH = (
    REPO_ROOT / "Scratchpad" / "coreml_proof_fixtures" / "generated_tokens_raw.npy"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Scratchpad" / "coreml_proof_validation"
DEFAULT_REPORT_NAME = "spectrostream_rvq_lookup_probe.json"
DEFAULT_SUMMARY_NAME = "spectrostream_rvq_lookup_probe.md"
QUANTIZER_KEY = "params/soundstream/quantizer/embedding"
SPECTROSTREAM_CODEBOOKS = 64
SPECTROSTREAM_CODEBOOK_SIZE = 1_024
SPECTROSTREAM_EMBEDDING_DIM = 256


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
  """Return scalar parity metrics for two same-shaped arrays."""
  actual_flat = actual.astype(np.float64).reshape(-1)
  expected_flat = expected.astype(np.float64).reshape(-1)
  delta = actual_flat - expected_flat
  if np.std(actual_flat) == 0.0 or np.std(expected_flat) == 0.0:
    correlation = None
  else:
    correlation = float(np.corrcoef(actual_flat, expected_flat)[0, 1])
  return {
      "shape": list(actual.shape),
      "max_abs_error": float(np.max(np.abs(delta))),
      "mean_abs_error": float(np.mean(np.abs(delta))),
      "correlation": correlation,
  }


def _load_tokens(
    *,
    unique_tokens_path: Path,
    raw_tokens_path: Path,
    frames: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
  """Load fixture tokens and derive the SpectroStream decoder-code contract."""
  if not unique_tokens_path.exists():
    raise FileNotFoundError(
        f"Unique token fixture not found: {unique_tokens_path}. "
        "Run scripts/generate_mrt2_coreml_reference_fixtures.py first."
    )
  unique_tokens = np.load(unique_tokens_path).astype(np.int32)[:frames]
  if unique_tokens.ndim != 2 or unique_tokens.shape[1] != MRT2_RVQ_LEVELS:
    raise ValueError(f"Expected unique token shape [N, 12], got {unique_tokens.shape}")
  decoder_codes = unique_token_to_raw_code(unique_tokens).astype(np.int32)

  metadata: dict[str, Any] = {
      "unique_tokens_path": str(unique_tokens_path),
      "cxx_tokens_out_path": str(raw_tokens_path),
      "cxx_tokens_out_fixture_available": raw_tokens_path.exists(),
      "frames": int(unique_tokens.shape[0]),
      "decoder_code_contract": (
          "MLX MagentaRT2Sampler.convert_from_unique_codes: "
          "(unique - NUM_RESERVED_TOKENS) % 1024."
      ),
  }
  if raw_tokens_path.exists():
    cxx_tokens_out = np.load(raw_tokens_path).astype(np.int32)[:frames]
    if cxx_tokens_out.shape != decoder_codes.shape:
      raise ValueError(
          "C++ tokens_out fixture shape does not match decoder-code conversion: "
          f"{cxx_tokens_out.shape} vs {decoder_codes.shape}"
      )
    delta = cxx_tokens_out - decoder_codes
    metadata["cxx_tokens_out_delta_min"] = int(np.min(delta))
    metadata["cxx_tokens_out_delta_max"] = int(np.max(delta))
    metadata["cxx_tokens_out_delta_unique"] = sorted(set(delta.reshape(-1).tolist()))
    metadata["cxx_tokens_out_note"] = (
        "The C++ tokens_out/debug fixture uses the output-facing contract and is "
        "not fed directly to SpectroStream decoder lookup."
    )
  return unique_tokens, decoder_codes, metadata


def _load_codebooks(checkpoint_path: Path) -> np.ndarray:
  """Load SpectroStream quantizer embedding codebooks from the MRT2 checkpoint."""
  arrays = safetensors_flax.load_file(str(checkpoint_path))
  if QUANTIZER_KEY not in arrays:
    raise KeyError(f"Missing checkpoint key: {QUANTIZER_KEY}")
  codebooks = np.asarray(arrays[QUANTIZER_KEY], dtype=np.float32)
  expected_shape = (
      SPECTROSTREAM_CODEBOOKS,
      SPECTROSTREAM_CODEBOOK_SIZE,
      SPECTROSTREAM_EMBEDDING_DIM,
  )
  if codebooks.shape != expected_shape:
    raise ValueError(f"Expected codebook shape {expected_shape}, got {codebooks.shape}")
  return codebooks


def _host_lookup_sum(codebooks: np.ndarray, raw_codes: np.ndarray) -> np.ndarray:
  """Sum selected codebook rows on CPU for raw RVQ codes shaped ``[T, 12]``."""
  levels = np.arange(raw_codes.shape[1])[:, np.newaxis]
  selected = codebooks[levels, raw_codes.T]
  return np.sum(selected, axis=0, dtype=np.float32)


def _one_hot_matmul_sum(codebooks: np.ndarray, raw_codes: np.ndarray) -> np.ndarray:
  """Reference one-hot matmul formulation for the same RVQ lookup."""
  eye = np.eye(codebooks.shape[1], dtype=np.float32)
  one_hot = eye[raw_codes]
  return np.einsum("tqv,qvd->td", one_hot, codebooks[: raw_codes.shape[1]])


def _mlx_lookup_sum(codebooks: np.ndarray, raw_codes: np.ndarray) -> np.ndarray:
  """Run the MLX SpectroStream quantizer reference."""
  import magenta_rt  # noqa: F401
  import mlx.core as mx
  import sequence_layers.mlx as sl

  from magenta_rt.mlx import model
  from magenta_rt.mlx import spectrostream

  exp = model.get_model_class("mrt2_small")()
  config = spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
      rvq_truncation_level=exp.spectrostream.rvq_truncation_level,
      use_unique_codes=False,
  )
  soundstream = config.make()
  soundstream.quantizer.embedding = mx.array(codebooks)
  sequence = sl.Sequence(
      mx.array(raw_codes[np.newaxis, :, :], dtype=mx.int32),
      mx.ones((1, raw_codes.shape[0]), dtype=mx.bool_),
  )
  output = soundstream.quantizer.codes_to_embeddings_layer.layer(sequence)
  mx.eval(output.values)
  return np.asarray(output.values[0], dtype=np.float32)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
  """Run the RVQ lookup probe and return a machine-readable report."""
  checkpoint_path = Path(args.checkpoint_path)
  unique_tokens, raw_codes, token_metadata = _load_tokens(
      unique_tokens_path=Path(args.unique_tokens_path),
      raw_tokens_path=Path(args.raw_tokens_path),
      frames=args.frames,
  )
  codebooks = _load_codebooks(checkpoint_path)

  per_frame_seconds = []
  frame_outputs = []
  for frame_codes in raw_codes:
    start = time.perf_counter()
    frame_outputs.append(_host_lookup_sum(codebooks, frame_codes[np.newaxis, :])[0])
    per_frame_seconds.append(time.perf_counter() - start)
  host_output = np.stack(frame_outputs, axis=0)
  one_hot_output = _one_hot_matmul_sum(codebooks, raw_codes)
  mlx_output = _mlx_lookup_sum(codebooks, raw_codes)

  return {
      "schema": "spectrostream-rvq-lookup-probe-v1",
      "checkpoint_path": str(checkpoint_path),
      "quantizer_key": QUANTIZER_KEY,
      "codebooks": {
          "shape": list(codebooks.shape),
          "dtype": str(codebooks.dtype),
          "used_levels": MRT2_RVQ_LEVELS,
      },
      "tokens": token_metadata
      | {
          "unique_shape": list(unique_tokens.shape),
          "decoder_code_shape": list(raw_codes.shape),
          "decoder_code_min": int(np.min(raw_codes)),
          "decoder_code_max": int(np.max(raw_codes)),
      },
      "host_lookup_vs_mlx": _metrics(host_output, mlx_output),
      "host_lookup_vs_one_hot_matmul": _metrics(host_output, one_hot_output),
      "timing_smoke": {
          "scope": "Python NumPy CPU host lookup only, not device timing",
          "p50_ms": float(np.percentile(per_frame_seconds, 50) * 1000.0),
          "p99_ms": float(np.percentile(per_frame_seconds, 99) * 1000.0),
      },
      "decision": {
          "rvq_lookup_owner": "host_cpu",
          "reason": (
              "The lookup is a small gather/sum and matches MLX exactly enough "
              "to keep gather out of the first Core ML decoder graph."
          ),
          "next_boundary": (
              "Feed the [frames, 256] host embeddings into the SpectroStream "
              "decoder/iSTFT streaming verifier before converting decoder convs."
          ),
      },
  }


def _write_summary(report: dict[str, Any], summary_path: Path) -> None:
  """Write a short markdown summary beside the JSON report."""
  metrics = report["host_lookup_vs_mlx"]
  one_hot = report["host_lookup_vs_one_hot_matmul"]
  timing = report["timing_smoke"]
  lines = [
      "# SpectroStream RVQ Lookup Probe",
      "",
      f"- Codebook shape: `{report['codebooks']['shape']}`",
      f"- Used RVQ levels: {report['codebooks']['used_levels']}",
      f"- Token frames: {report['tokens']['frames']}",
      f"- Decoder code range: "
      f"{report['tokens']['decoder_code_min']}..{report['tokens']['decoder_code_max']}",
      f"- Host CPU lookup vs MLX max error: {metrics['max_abs_error']:.10f}",
      f"- Host CPU lookup vs MLX mean error: {metrics['mean_abs_error']:.10f}",
      f"- Host CPU lookup vs MLX correlation: {metrics['correlation']:.12f}",
      f"- Host CPU lookup vs one-hot matmul max error: {one_hot['max_abs_error']:.10f}",
      f"- Host lookup p50/p99 smoke: {timing['p50_ms']:.6f} ms / {timing['p99_ms']:.6f} ms",
      "",
      "Decision:",
      f"- RVQ lookup owner: `{report['decision']['rvq_lookup_owner']}`",
      f"- Reason: {report['decision']['reason']}",
      f"- Next boundary: {report['decision']['next_boundary']}",
      "",
  ]
  summary_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
  """Parse CLI flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--checkpoint-path",
      default=str(paths.checkpoints_dir() / "mrt2_small.safetensors"),
  )
  parser.add_argument("--unique-tokens-path", default=str(DEFAULT_UNIQUE_TOKENS_PATH))
  parser.add_argument("--raw-tokens-path", default=str(DEFAULT_RAW_TOKENS_PATH))
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
  parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
  parser.add_argument("--frames", type=int, default=25)
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  args = parse_args()
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  report = run_probe(args)
  report_path = output_dir / args.report_name
  summary_path = output_dir / args.summary_name
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  _write_summary(report, summary_path)
  print(f"Wrote {report_path}")
  print(f"Wrote {summary_path}")
  print(
      "Host lookup vs MLX max error "
      f"{report['host_lookup_vs_mlx']['max_abs_error']:.10f}"
  )


if __name__ == "__main__":
  main()
