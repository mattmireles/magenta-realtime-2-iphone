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

"""Convert the MRT2 Depthformer depth-body logits wrapper to Core ML.

SUPERSEDED (paper §6.5) — retained as a negative-result artifact. This exports
full-vocabulary depth logits for host-side sampling, which requires 12 Core ML
predictions per frame. On phones the depth path is weight-bandwidth-bound
(per-call cost ~= weight bytes / DRAM bandwidth), so 12 predictions cost
~40 ms/frame regardless of FLOPs — over the entire frame budget. The corrected
depth exporter is ``convert_depth_body_rollout.py``
(``DepthBodyRolloutWrapper``): all 12 RVQ levels sampled in ONE in-graph FP16
prediction from host-supplied Gumbel noise (12.7 ms/frame on A14). Keep this
script as the FLOAT32 full-pass reference the rollout validator gates against.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch

from mrt2_coreml.depth_body_wrapper import DepthBodyLogitsWrapper


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "models"
DEFAULT_PACKAGE_NAME = "mrt2_depth_body_logits.mlpackage"
DEFAULT_COMPILED_NAME = "mrt2_depth_body_logits.mlmodelc"
DEFAULT_METADATA_NAME = "mrt2_depth_body_logits_export_metadata.json"
DEFAULT_BATCH_SIZE = 1
INPUT_NAME = "depth_inputs"
OUTPUT_NAME = "depth_logits"
DEPLOYMENT_TARGET = "iOS18"
# FLOAT32 is the published default: the FLOAT16 depth export passed naive
# correlation checks but corrupted sampled tokens on device. See
# docs/validation-receipts.md before changing this.
COMPUTE_PRECISION = "FLOAT32"


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


def _compile_model(package_path: Path, compiled_path: Path) -> str:
  """Compile an ``.mlpackage`` with Xcode's Core ML compiler."""
  if compiled_path.exists():
    shutil.rmtree(compiled_path)
  compile_dir = compiled_path.parent
  output = subprocess.check_output(
      [
          "/usr/bin/xcrun",
          "coremlcompiler",
          "compile",
          str(package_path),
          str(compile_dir),
      ],
      cwd=REPO_ROOT,
      text=True,
      stderr=subprocess.STDOUT,
  )
  default_compiled = compile_dir / package_path.with_suffix(".mlmodelc").name
  if default_compiled.exists() and default_compiled != compiled_path:
    if compiled_path.exists():
      shutil.rmtree(compiled_path)
    default_compiled.rename(compiled_path)
  return output.strip()


def convert(args: argparse.Namespace) -> dict[str, Any]:
  """Trace, convert, save, and optionally compile the depth-body logits wrapper."""
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  package_path = output_dir / args.package_name
  compiled_path = output_dir / args.compiled_name
  metadata_path = output_dir / args.metadata_name
  compute_precision = getattr(ct.precision, args.compute_precision)

  batch_size = int(args.batch_size)
  if batch_size < 1:
    raise ValueError("--batch-size must be positive")

  model = DepthBodyLogitsWrapper().eval()
  input_shape = (batch_size, 12, 1024)
  output_shape = (batch_size, 12, 12294)
  example_input = torch.zeros(input_shape, dtype=torch.float32)
  start_trace = time.perf_counter()
  traced = torch.jit.trace(model, example_input)
  trace_seconds = time.perf_counter() - start_trace

  stderr_buffer = io.StringIO()
  caught_warnings: list[warnings.WarningMessage]
  start_convert = time.perf_counter()
  with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    with contextlib.redirect_stderr(stderr_buffer):
      mlmodel = ct.convert(
          traced,
          convert_to="mlprogram",
          inputs=[
              ct.TensorType(
                  name=INPUT_NAME,
                  shape=input_shape,
                  dtype=np.float32,
              )
          ],
          outputs=[ct.TensorType(name=OUTPUT_NAME)],
          compute_precision=compute_precision,
          minimum_deployment_target=ct.target.iOS18,
      )
  convert_seconds = time.perf_counter() - start_convert

  if package_path.exists():
    shutil.rmtree(package_path)
  mlmodel.save(str(package_path))

  compile_output = None
  compile_error = None
  if args.compile:
    try:
      compile_output = _compile_model(package_path, compiled_path)
    except (OSError, subprocess.CalledProcessError) as exc:
      compile_error = str(exc)

  metadata: dict[str, Any] = {
      "schema": "mrt2-depth-body-logits-coreml-export-v1",
      "source_commit": _git_commit(),
      "wrapper": "magenta_rt.coreml.depth_body_wrapper.DepthBodyLogitsWrapper",
      "boundary": "depth-body logits from fixed depth_inputs, not temporal cache",
      "conversion": {
          "convert_to": "mlprogram",
          "compute_precision": args.compute_precision,
          "minimum_deployment_target": DEPLOYMENT_TARGET,
          "trace_seconds": trace_seconds,
          "convert_seconds": convert_seconds,
      },
      "inputs": [
          {
              "name": INPUT_NAME,
              "shape": list(input_shape),
              "dtype": "float32",
          }
      ],
      "outputs": [
          {
              "name": OUTPUT_NAME,
              "shape": list(output_shape),
              "dtype": "fp16/fp32 Core ML selected",
          }
      ],
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
          "compile_error": compile_error,
          "compile_output": compile_output,
      },
      "known_limits": [
          "Exports the depth transformer body only.",
          "Temporal transformer output and previous RVQ embeddings remain host-provided inputs.",
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
  parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
  parser.add_argument(
      "--compute-precision",
      choices=("FLOAT16", "FLOAT32"),
      default=COMPUTE_PRECISION,
  )
  parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  metadata = convert(parse_args())
  print(f"Saved {metadata['artifacts']['mlpackage']}")
  if metadata["artifacts"]["mlmodelc"] is not None:
    print(f"Compiled {metadata['artifacts']['mlmodelc']}")
  if metadata["warnings"]["compile_error"]:
    print(f"Compile failed: {metadata['warnings']['compile_error']}")
  print(f"Wrote {metadata['artifacts']['metadata']}")


if __name__ == "__main__":
  main()
