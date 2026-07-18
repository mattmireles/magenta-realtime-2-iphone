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

"""Convert a host-owned K/V carry MRT2 temporal-body proof to Core ML."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch

from mrt2_coreml.depthformer_wrapper import (
    MRT2_HEAD_DIM,
    MRT2_LOCAL_WINDOW_FRAMES,
    MRT2_MODEL_DIM,
    MRT2_TEMPORAL_HEADS,
)
from mrt2_coreml.temporal_body_wrapper import (
    TEMPORAL_SOURCE_DIM,
    TemporalBodyCoreMLCarryWrapper,
    TemporalBodyCoreMLStreamingCarryWrapper,
)
from mrt2_coreml.weight_compression import (
    WEIGHT_COMPRESSION_CHOICES,
    compress_weights,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "models"
DEFAULT_PACKAGE_TEMPLATE = "mrt2_temporal_body_carry_{frames:02d}.mlpackage"
DEFAULT_COMPILED_TEMPLATE = "mrt2_temporal_body_carry_{frames:02d}.mlmodelc"
DEFAULT_METADATA_TEMPLATE = (
    "mrt2_temporal_body_carry_{frames:02d}_export_metadata.json"
)
STREAMING_PACKAGE_TEMPLATE = "mrt2_temporal_body_streaming_carry_01.mlpackage"
STREAMING_COMPILED_TEMPLATE = "mrt2_temporal_body_streaming_carry_01.mlmodelc"
STREAMING_METADATA_TEMPLATE = (
    "mrt2_temporal_body_streaming_carry_01_export_metadata.json"
)
TEMPORAL_INPUT_NAME = "temporal_inputs"
SOURCE_INPUT_NAME = "source_encoded"
CACHE_VALID_BIAS_NAME = (
    TemporalBodyCoreMLStreamingCarryWrapper.cache_valid_bias_name
)
OUTPUT_NAME = "temporal_outputs"
DEPLOYMENT_TARGET = "iOS18"


def _ensure_coreml_runtime_path() -> None:
  """Give coremltools access to macOS helper tools when Codex PATH is thin."""
  path_parts = os.environ.get("PATH", "").split(os.pathsep)
  for required in ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]:
    if required not in path_parts:
      path_parts.append(required)
  os.environ["PATH"] = os.pathsep.join(path_parts)


def _git_commit() -> str:
  """Return the current commit hash or an explicit unavailable marker."""
  try:
    return subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
  except (OSError, subprocess.CalledProcessError) as exc:
    return f"unavailable: {exc}"


def _compile_model(package_path: Path, compiled_path: Path) -> dict[str, Any]:
  """Compile an ``.mlpackage`` with Xcode's Core ML compiler."""
  if compiled_path.exists():
    shutil.rmtree(compiled_path)
  try:
    output = subprocess.check_output(
        [
            "/usr/bin/xcrun",
            "coremlcompiler",
            "compile",
            str(package_path),
            str(compiled_path.parent),
        ],
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    default_compiled = (
        compiled_path.parent / package_path.with_suffix(".mlmodelc").name
    )
    if default_compiled.exists() and default_compiled != compiled_path:
      if compiled_path.exists():
        shutil.rmtree(compiled_path)
      default_compiled.rename(compiled_path)
    return {
        "ok": True,
        "output": output.strip(),
        "compiled_path": str(compiled_path) if compiled_path.exists() else None,
    }
  except (OSError, subprocess.CalledProcessError) as exc:
    return {
        "ok": False,
        "error": str(exc),
        "output": getattr(exc, "output", None),
        "compiled_path": None,
    }


def _format_template(template: str, args: argparse.Namespace) -> str:
  """Format artifact template fields for carry frame/history buckets."""
  if args.streaming:
    replacements = {
        DEFAULT_PACKAGE_TEMPLATE: STREAMING_PACKAGE_TEMPLATE,
        DEFAULT_COMPILED_TEMPLATE: STREAMING_COMPILED_TEMPLATE,
        DEFAULT_METADATA_TEMPLATE: STREAMING_METADATA_TEMPLATE,
    }
    template = replacements.get(template, template)
  formatted = template.format(
      frames=args.frames, history_length=args.history_length
  )
  if args.history_length == 0 or "history_length" in template:
    return formatted
  path = Path(formatted)
  return f"{path.stem}_h{args.history_length:02d}{path.suffix}"


def _variant_path(path: Path, variant: str) -> Path:
  """Append a compression variant without risking baseline overwrite."""
  if variant == "none":
    return path
  return path.with_name(f"{path.stem}_{variant}{path.suffix}")


def _artifact_bytes(path: Path) -> int | None:
  """Return recursive artifact bytes, or ``None`` when absent."""
  if not path.exists():
    return None
  if path.is_file():
    return path.stat().st_size
  return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _cache_tensor_types() -> list[ct.TensorType]:
  """Return ordinary Core ML tensor inputs for host-owned K/V caches."""
  cache_shape = (
      1,
      MRT2_LOCAL_WINDOW_FRAMES,
      MRT2_TEMPORAL_HEADS,
      MRT2_HEAD_DIM,
  )
  return [
      ct.TensorType(name=name, shape=cache_shape, dtype=np.float16)
      for name in TemporalBodyCoreMLCarryWrapper.cache_input_names()
  ]


def _output_tensor_types(args: argparse.Namespace) -> list[ct.TensorType]:
  """Return temporal output plus K/V update output declarations."""
  return [
      ct.TensorType(name=OUTPUT_NAME),
      *[
          ct.TensorType(name=name, dtype=np.float16)
          for name in TemporalBodyCoreMLCarryWrapper.cache_update_output_names()
      ],
  ]


def convert(args: argparse.Namespace) -> dict[str, Any]:
  """Trace, convert, save, optionally compile, and return export metadata."""
  if args.streaming and args.frames != 1:
    raise ValueError("--streaming requires --frames 1")
  _ensure_coreml_runtime_path()
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  package_path = _variant_path(
      output_dir / _format_template(args.package_template, args),
      args.weight_compression,
  )
  compiled_path = _variant_path(
      output_dir / _format_template(args.compiled_template, args),
      args.weight_compression,
  )
  metadata_path = _variant_path(
      output_dir / _format_template(args.metadata_template, args),
      args.weight_compression,
  )

  model = (
      TemporalBodyCoreMLStreamingCarryWrapper()
      if args.streaming
      else TemporalBodyCoreMLCarryWrapper(
          frame_count=args.frames,
          history_length=args.history_length,
      )
  ).eval()
  temporal_inputs = torch.zeros(
      (1, args.frames, MRT2_MODEL_DIM), dtype=torch.float32
  )
  source_encoded = torch.zeros(
      (1, args.frames, TEMPORAL_SOURCE_DIM), dtype=torch.float32
  )
  cache_inputs = [
      torch.zeros(
          (
              1,
              MRT2_LOCAL_WINDOW_FRAMES,
              MRT2_TEMPORAL_HEADS,
              MRT2_HEAD_DIM,
          ),
          dtype=torch.float16,
      )
      for _ in model.cache_input_names()
  ]

  trace_inputs: tuple[torch.Tensor, ...]
  if args.streaming:
    cache_valid_bias = torch.full(
        (1, 1, 1, TemporalBodyCoreMLStreamingCarryWrapper.attention_extent),
        -1e4,
        dtype=torch.float16,
    )
    cache_valid_bias[..., 0] = 0
    cache_valid_bias[..., -1] = 0
    trace_inputs = (
        temporal_inputs,
        source_encoded,
        cache_valid_bias,
        *cache_inputs,
    )
  else:
    trace_inputs = (temporal_inputs, source_encoded, *cache_inputs)

  start_trace = time.perf_counter()
  traced = torch.jit.trace(model, trace_inputs)
  trace_seconds = time.perf_counter() - start_trace

  stderr_buffer = io.StringIO()
  start_convert = time.perf_counter()
  with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    with contextlib.redirect_stderr(stderr_buffer):
      mlmodel = ct.convert(
          traced,
          convert_to="mlprogram",
          inputs=[
              ct.TensorType(
                  name=TEMPORAL_INPUT_NAME,
                  shape=(1, args.frames, MRT2_MODEL_DIM),
                  dtype=np.float32,
              ),
              ct.TensorType(
                  name=SOURCE_INPUT_NAME,
                  shape=(1, args.frames, TEMPORAL_SOURCE_DIM),
                  dtype=np.float32,
              ),
              *(
                  [
                      ct.TensorType(
                          name=CACHE_VALID_BIAS_NAME,
                          shape=(
                              1,
                              1,
                              1,
                              TemporalBodyCoreMLStreamingCarryWrapper.attention_extent,
                          ),
                          dtype=np.float16,
                      )
                  ]
                  if args.streaming
                  else []
              ),
              *_cache_tensor_types(),
          ],
          outputs=_output_tensor_types(args),
          compute_precision=ct.precision.FLOAT16,
          minimum_deployment_target=ct.target.iOS18,
      )
  convert_seconds = time.perf_counter() - start_convert
  mlmodel, compression_report = compress_weights(mlmodel, args.weight_compression)

  if package_path.exists():
    shutil.rmtree(package_path)
  mlmodel.save(str(package_path))

  compile_report = (
      _compile_model(package_path, compiled_path) if args.compile else None
  )
  metadata: dict[str, Any] = {
      "schema": (
          "mrt2-temporal-body-streaming-carry-coreml-export-v1"
          if args.streaming
          else "mrt2-temporal-body-carry-coreml-export-v1"
      ),
      "source_commit": _git_commit(),
      "frames": args.frames,
      "history_length": args.history_length,
      "wrapper": f"{model.__class__.__module__}.{model.__class__.__name__}",
      "boundary": (
          "host_owned_chronological_kv_ring_with_validity_bias"
          if args.streaming
          else "host_owned_kv_cache_inputs_and_update_outputs"
      ),
      "inputs": [
          {
              "name": TEMPORAL_INPUT_NAME,
              "shape": [1, args.frames, MRT2_MODEL_DIM],
              "dtype": "float32",
          },
          {
              "name": SOURCE_INPUT_NAME,
              "shape": [1, args.frames, TEMPORAL_SOURCE_DIM],
              "dtype": "float32",
          },
          *(
              [{
                  "name": CACHE_VALID_BIAS_NAME,
                  "shape": [
                      1,
                      1,
                      1,
                      TemporalBodyCoreMLStreamingCarryWrapper.attention_extent,
                  ],
                  "dtype": "float16",
                  "semantics": (
                      "sink + 41 chronological cache slots + current frame"
                  ),
              }]
              if args.streaming
              else []
          ),
          *[
              {
                  "name": name,
                  "shape": [
                      1,
                      MRT2_LOCAL_WINDOW_FRAMES,
                      MRT2_TEMPORAL_HEADS,
                      MRT2_HEAD_DIM,
                  ],
                  "dtype": "float16",
              }
              for name in TemporalBodyCoreMLCarryWrapper.cache_input_names()
          ],
      ],
      "outputs": [
          {
              "name": OUTPUT_NAME,
              "shape": [1, args.frames, MRT2_MODEL_DIM],
              "dtype": "Core ML selected",
          },
          *[
              {
                  "name": name,
                  "shape": [1, args.frames, MRT2_TEMPORAL_HEADS, MRT2_HEAD_DIM],
                  "dtype": "float16",
              }
              for name in (
                  TemporalBodyCoreMLCarryWrapper.cache_update_output_names()
              )
          ],
      ],
      "conversion": {
          "convert_to": "mlprogram",
          "compute_precision": "FLOAT16",
          "minimum_deployment_target": DEPLOYMENT_TARGET,
          "trace_seconds": trace_seconds,
          "convert_seconds": convert_seconds,
      },
      "weight_compression": compression_report,
      "artifacts": {
          "mlpackage": str(package_path),
          "mlmodelc": str(compiled_path) if compiled_path.exists() else None,
          "metadata": str(metadata_path),
          "mlpackage_bytes": _artifact_bytes(package_path),
          "mlmodelc_bytes": _artifact_bytes(compiled_path),
      },
      "warnings": {
          "python_warnings": [
              {
                  "category": warning.category.__name__,
                  "message": str(warning.message),
              }
              for warning in caught_warnings
          ],
          "stderr": stderr_buffer.getvalue().strip(),
      },
      "compile": compile_report,
      "known_limits": [
          *(
              [
                  (
                      "The host must keep cache entries chronological and"
                      " mutate all 48 arrays in lockstep."
                  ),
                  (
                      "Warmup validity is supplied as a static-shape additive"
                      " bias; after 41 frames all cache slots are valid."
                  ),
              ]
              if args.streaming
              else [
                  "This first carry proof uses a fixed no-wrap history_length"
                  " bucket."
              ]
          ),
          "Host code owns K/V cache lifetime and ring-buffer placement.",
          "The graph returns K/V update slices, not full copied cache tensors.",
          (
              "Conditioning encoder remains host-owned; source_encoded is an"
              " input."
          ),
          "Depth-body logits remain a separate Core ML package for this phase.",
      ],
  }
  metadata_path.write_text(
      json.dumps(metadata, indent=2, sort_keys=True) + "\n"
  )
  return metadata


def parse_args() -> argparse.Namespace:
  """Parse command-line flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--frames", type=int, default=2)
  parser.add_argument("--history-length", type=int, default=0)
  parser.add_argument(
      "--streaming",
      action="store_true",
      help=(
          "Export the one-frame chronological host-ring boundary with a"
          " validity-bias input."
      ),
  )
  parser.add_argument("--package-template", default=DEFAULT_PACKAGE_TEMPLATE)
  parser.add_argument("--compiled-template", default=DEFAULT_COMPILED_TEMPLATE)
  parser.add_argument("--metadata-template", default=DEFAULT_METADATA_TEMPLATE)
  parser.add_argument(
      "--compile", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument(
      "--weight-compression",
      choices=WEIGHT_COMPRESSION_CHOICES,
      default="none",
      help="Post-training Core ML constant-weight compression variant.",
  )
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  metadata = convert(parse_args())
  print(f"Saved {metadata['artifacts']['mlpackage']}")
  if metadata["artifacts"]["mlmodelc"] is not None:
    print(f"Compiled {metadata['artifacts']['mlmodelc']}")
  if metadata["compile"] and not metadata["compile"]["ok"]:
    print(f"Compile failed: {metadata['compile']['error']}")
  print(f"Wrote {metadata['artifacts']['metadata']}")


if __name__ == "__main__":
  main()
