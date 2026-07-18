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

"""Convert the MRT2 whole-frame in-graph depth rollout to Core ML.

This exports ``DepthBodyRolloutWrapper``: ONE prediction per frame that
samples all 12 RVQ levels inside the graph (host-supplied Gumbel noise keeps
RNG/seed host-owned). It replaces the 12-predictions-per-frame designs
(``mrt2_depth_body_logits`` 12x full pass, ``mrt2_depth_body_step`` 12x
stateful step), both of which are weight-bandwidth-doomed on phones: every
prediction streams the full depth weight set from DRAM, so 12 predictions
cost ~40 ms/frame regardless of FLOPs. This is the paper's §6.5 finding
(weight bytes / DRAM bandwidth is the per-call invariant on every compute
unit), which reshaped the sampling loop from twelve predictions per frame
into one.

Default compute precision is FLOAT16: per-call cost on LPDDR-limited devices
is weight bytes / memory bandwidth, so halving weight bytes halves depth
latency. Validate with validation/validate_depth_body_rollout.py (zero-noise
argmax token parity vs the FLOAT32 full-pass reference plus noisy-arm parity
vs the reference sampler) before bundling. FLOAT32 export is token-for-token
exact (0/900 mismatches); FLOAT16 is the ship precision and flips fp16
near-tie tokens without changing the sampling distribution.
"""

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

from mrt2_coreml.depth_body_wrapper import (
    DEPTH_ROLLOUT_TOP_K,
    DepthBodyRolloutWrapper,
)
from mrt2_coreml.depthformer_wrapper import (
    MRT2_CODEBOOK_SIZE,
    MRT2_MODEL_DIM,
    MRT2_RVQ_LEVELS,
)
from mrt2_coreml.weight_compression import (
    WEIGHT_COMPRESSION_CHOICES,
    compress_weights,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "models"
DEFAULT_PACKAGE_NAME = "mrt2_depth_body_rollout.mlpackage"
DEFAULT_COMPILED_NAME = "mrt2_depth_body_rollout.mlmodelc"
DEFAULT_METADATA_NAME = "mrt2_depth_body_rollout_export_metadata.json"
FRAME_INPUT_NAME = "temporal_frame"
NOISE_INPUT_NAME = "gumbel_noise"
INVERSE_TEMPERATURE_INPUT_NAME = "inverse_temperature"
CODES_OUTPUT_NAME = "sampled_codes"
FEEDBACK_OUTPUT_NAME = "temporal_feedback"
DEPLOYMENT_TARGET = "iOS18"
# FLOAT16 by default — set by DEVICE data, not the projection. The in-graph
# unroll streams the transformer LAYER weights once per level (only the
# to_logits slices stream once per frame), so FLOAT32 is ~750 MB/frame ≈
# 37 ms measured on iPhone 12 Pro — still over budget. FLOAT16 halves that:
# 12.7 ms/frame on iPhone 12 Pro, 8.4 ms on iPhone 15 Pro Max (zero
# underruns composed). FLOAT16 flips fp16 near-tie tokens (~40% of frames
# flip at least one level vs the exact chain; distribution unchanged) and
# passed the 2026-06-10 device quality gate: metrics in the clean cluster,
# clean spectrogram, zero sample-level clicks, judge-fail pattern shown to
# be shared with the token-exact FLOAT32 arm. Use --compute-precision
# FLOAT32 for the token-for-token validation reference (0/900 mismatches).
COMPUTE_PRECISION = "FLOAT16"


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
    default_compiled = compiled_path.parent / package_path.with_suffix(".mlmodelc").name
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


def convert(args: argparse.Namespace) -> dict[str, Any]:
  """Trace, convert, save, optionally compile, and return export metadata."""
  _ensure_coreml_runtime_path()
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  package_path = _variant_path(output_dir / args.package_name, args.weight_compression)
  compiled_path = _variant_path(
      output_dir / args.compiled_name, args.weight_compression
  )
  metadata_path = _variant_path(
      output_dir / args.metadata_name, args.weight_compression
  )
  compute_precision = getattr(ct.precision, args.compute_precision)

  model = DepthBodyRolloutWrapper().eval()
  temporal_frame = torch.zeros((1, 1, MRT2_MODEL_DIM), dtype=torch.float32)
  gumbel_noise = torch.zeros(
      (MRT2_RVQ_LEVELS, MRT2_CODEBOOK_SIZE), dtype=torch.float32
  )
  inverse_temperature = torch.ones((1,), dtype=torch.float32)

  start_trace = time.perf_counter()
  traced = torch.jit.trace(
      model, (temporal_frame, gumbel_noise, inverse_temperature)
  )
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
                  name=FRAME_INPUT_NAME,
                  shape=(1, 1, MRT2_MODEL_DIM),
                  dtype=np.float32,
              ),
              ct.TensorType(
                  name=NOISE_INPUT_NAME,
                  shape=(MRT2_RVQ_LEVELS, MRT2_CODEBOOK_SIZE),
                  dtype=np.float32,
              ),
              ct.TensorType(
                  name=INVERSE_TEMPERATURE_INPUT_NAME,
                  shape=(1,),
                  dtype=np.float32,
              ),
          ],
          outputs=[
              ct.TensorType(name=CODES_OUTPUT_NAME),
              ct.TensorType(name=FEEDBACK_OUTPUT_NAME),
          ],
          compute_precision=compute_precision,
          minimum_deployment_target=ct.target.iOS18,
      )
  convert_seconds = time.perf_counter() - start_convert
  mlmodel, compression_report = compress_weights(mlmodel, args.weight_compression)

  if package_path.exists():
    shutil.rmtree(package_path)
  mlmodel.save(str(package_path))

  compile_report = _compile_model(package_path, compiled_path) if args.compile else None
  metadata: dict[str, Any] = {
      "schema": "mrt2-depth-body-rollout-coreml-export-v1",
      "source_commit": _git_commit(),
      "wrapper": "mrt2_coreml.depth_body_wrapper.DepthBodyRolloutWrapper",
      "boundary": (
          "one prediction per frame; all 12 RVQ levels sampled in-graph via "
          "Gumbel-max over a static top-{topk} set (host supplies noise + "
          "inverse temperature, so determinism stays host-owned); per-level "
          "to_logits slices touch the projection weights once per frame; "
          "token-embedding feedback is an in-graph gather; no Core ML state"
      ).format(topk=DEPTH_ROLLOUT_TOP_K),
      "inputs": [
          {
              "name": FRAME_INPUT_NAME,
              "shape": [1, 1, MRT2_MODEL_DIM],
              "dtype": "float32",
              "contract": "temporal transformer output for the frame",
          },
          {
              "name": NOISE_INPUT_NAME,
              "shape": [MRT2_RVQ_LEVELS, MRT2_CODEBOOK_SIZE],
              "dtype": "float32",
              "contract": (
                  "per-level, per-code Gumbel(0,1) noise: -log(-log(u)), "
                  "u from the host's seeded RNG clamped to >= 1e-7"
              ),
          },
          {
              "name": INVERSE_TEMPERATURE_INPUT_NAME,
              "shape": [1],
              "dtype": "float32",
              "contract": "1 / max(0.05, sampling temperature)",
          },
      ],
      "outputs": [
          {
              "name": CODES_OUTPUT_NAME,
              "shape": [MRT2_RVQ_LEVELS],
              "dtype": "int32",
              "contract": "codebook-local codes; unique id = 6 + level*1024 + code",
          },
          {
              "name": FEEDBACK_OUTPUT_NAME,
              "shape": [1, MRT2_MODEL_DIM],
              "dtype": "Core ML selected",
              "contract": (
                  "mean of the 12 sampled token embeddings (x32 scale baked "
                  "in) — the next frame's temporal_inputs row"
              ),
          },
      ],
      "sampling": {
          "top_k": DEPTH_ROLLOUT_TOP_K,
          "top_k_scope": "static; selected on raw soft-capped logits before temperature",
          "method": "Gumbel-max over the top-k set (== top-k softmax sampling)",
      },
      "conversion": {
          "convert_to": "mlprogram",
          "compute_precision": args.compute_precision,
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
          "top_k is baked at export; the host's sampling.topK control is "
          "ignored by this graph.",
          "Sampling is Gumbel-max: identical DISTRIBUTION to the host "
          "inverse-CDF sampler, but token sequences differ for the same seed "
          "(the seed reproduces runs of THIS graph, not of older builds).",
      ],
  }
  metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
  return metadata


def parse_args() -> argparse.Namespace:
  """Parse command-line flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--package-name", default=DEFAULT_PACKAGE_NAME)
  parser.add_argument("--compiled-name", default=DEFAULT_COMPILED_NAME)
  parser.add_argument("--metadata-name", default=DEFAULT_METADATA_NAME)
  parser.add_argument(
      "--compute-precision",
      choices=("FLOAT16", "FLOAT32"),
      default=COMPUTE_PRECISION,
  )
  parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
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
