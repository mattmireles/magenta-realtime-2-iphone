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

"""Convert a no-wrap unrolled MRT2 temporal-body proof to Core ML.

The unrolled model keeps one Core ML model/state object and executes a fixed
number of temporal frames in one prediction. This tests whether later frames can
observe K/V state written by earlier frames without relying on separate
fixed-slot packages that cannot share ``MLState`` continuity by themselves.
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

from mrt2_coreml.depthformer_wrapper import (
    MRT2_HEAD_DIM,
    MRT2_LOCAL_WINDOW_FRAMES,
    MRT2_MODEL_DIM,
    MRT2_TEMPORAL_HEADS,
)
from mrt2_coreml.temporal_body_wrapper import (
    TEMPORAL_SOURCE_DIM,
    TemporalBodyCoreMLUnrolledWrapper,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "models"
DEFAULT_PACKAGE_TEMPLATE = "mrt2_temporal_body_unrolled_{frames:02d}.mlpackage"
DEFAULT_COMPILED_TEMPLATE = "mrt2_temporal_body_unrolled_{frames:02d}.mlmodelc"
DEFAULT_METADATA_TEMPLATE = (
    "mrt2_temporal_body_unrolled_{frames:02d}_export_metadata.json"
)
TEMPORAL_INPUT_NAME = "temporal_inputs"
SOURCE_INPUT_NAME = "source_encoded"
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


def _state_types() -> list[ct.StateType]:
  """Return Core ML state declarations for all temporal K/V buffers."""
  return [
      ct.StateType(
          wrapped_type=ct.TensorType(
              shape=(
                  1,
                  MRT2_LOCAL_WINDOW_FRAMES,
                  MRT2_TEMPORAL_HEADS,
                  MRT2_HEAD_DIM,
              ),
              dtype=np.float16,
          ),
          name=name,
      )
      for name in TemporalBodyCoreMLUnrolledWrapper.state_names()
  ]


def convert(args: argparse.Namespace) -> dict[str, Any]:
  """Trace, convert, save, optionally compile, and return export metadata."""
  _ensure_coreml_runtime_path()
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  package_path = output_dir / args.package_template.format(frames=args.frames)
  compiled_path = output_dir / args.compiled_template.format(frames=args.frames)
  metadata_path = output_dir / args.metadata_template.format(frames=args.frames)

  model = TemporalBodyCoreMLUnrolledWrapper(frame_count=args.frames).eval()
  temporal_inputs = torch.zeros((1, args.frames, MRT2_MODEL_DIM), dtype=torch.float32)
  source_encoded = torch.zeros((1, args.frames, TEMPORAL_SOURCE_DIM), dtype=torch.float32)

  start_trace = time.perf_counter()
  traced = torch.jit.trace(model, (temporal_inputs, source_encoded))
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
          ],
          outputs=[ct.TensorType(name=OUTPUT_NAME)],
          states=_state_types(),
          compute_precision=ct.precision.FLOAT16,
          minimum_deployment_target=ct.target.iOS18,
      )
  convert_seconds = time.perf_counter() - start_convert

  if package_path.exists():
    shutil.rmtree(package_path)
  mlmodel.save(str(package_path))

  compile_report = _compile_model(package_path, compiled_path) if args.compile else None
  metadata: dict[str, Any] = {
      "schema": "mrt2-temporal-body-unrolled-coreml-export-v1",
      "source_commit": _git_commit(),
      "frames": args.frames,
      "wrapper": (
          "magenta_rt.coreml.temporal_body_wrapper."
          "TemporalBodyCoreMLUnrolledWrapper"
      ),
      "boundary": "unrolled_temporal_body_outputs_from_host_temporal_inputs",
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
      ],
      "outputs": [
          {
              "name": OUTPUT_NAME,
              "shape": [1, args.frames, MRT2_MODEL_DIM],
              "dtype": "Core ML selected",
          }
      ],
      "states": [
          {
              "name": name,
              "shape": [1, MRT2_LOCAL_WINDOW_FRAMES, MRT2_TEMPORAL_HEADS, MRT2_HEAD_DIM],
              "dtype": "float16",
          }
          for name in TemporalBodyCoreMLUnrolledWrapper.state_names()
      ],
      "conversion": {
          "convert_to": "mlprogram",
          "compute_precision": "FLOAT16",
          "minimum_deployment_target": DEPLOYMENT_TARGET,
          "trace_seconds": trace_seconds,
          "convert_seconds": convert_seconds,
      },
      "artifacts": {
          "mlpackage": str(package_path),
          "mlmodelc": str(compiled_path) if compiled_path.exists() else None,
          "metadata": str(metadata_path),
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
          "Unrolls a fixed no-wrap frame count into one prediction.",
          "Graph size and conversion time grow with frame count.",
          "This is a proof of state read-after-write, not the final per-frame API.",
          "Conditioning encoder remains host-owned; source_encoded is an input.",
          "Depth-body logits remain a separate Core ML package for this phase.",
      ],
  }
  metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
  return metadata


def parse_args() -> argparse.Namespace:
  """Parse command-line flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  # The published MRT2TemporalBody.mlpackage is the 1-frame stateful export
  # (one prediction per 40 ms frame); larger unrolls are probe variants.
  parser.add_argument("--frames", type=int, default=1)
  parser.add_argument("--package-template", default=DEFAULT_PACKAGE_TEMPLATE)
  parser.add_argument("--compiled-template", default=DEFAULT_COMPILED_TEMPLATE)
  parser.add_argument("--metadata-template", default=DEFAULT_METADATA_TEMPLATE)
  parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
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
