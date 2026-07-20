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

"""Probe Core ML state updates for MRT2 temporal K/V cache tensors.

This is a Phase 3B feasibility probe, not the full temporal transformer export.
It answers one narrow question before the 12-layer graph is traced:

Can coremltools produce and run an iOS 18 ``mlprogram`` with an MRT2-shaped
``[1, 41, 8, 128]`` temporal K/V cache as ``ct.StateType`` and an in-place slice
update?

The probe also records two useful negative controls:

- whole-buffer ``copy_(cat(slice(cache), new))`` is rejected by the Torch
  frontend assignment pass;
- FP32 state is rejected by the Core ML backend because states only support
  FP16 on this toolchain.
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
from torch import nn


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "Scratchpad" / "coreml_proof_models"
DEFAULT_REPORT_DIR = REPO_ROOT / "Scratchpad" / "coreml_proof_validation"
DEFAULT_REPORT_NAME = "mrt2_temporal_kv_state_probe.json"
STATE_NAME = "temporal_layer_00_self_key_cache"
STATE_SHAPE = (1, 41, 8, 128)
NEW_KEY_SHAPE = (1, 1, 8, 128)


def _ensure_coreml_runtime_path() -> None:
  """Give coremltools access to macOS helper tools when Codex PATH is thin."""
  path_parts = os.environ.get("PATH", "").split(os.pathsep)
  for required in ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]:
    if required not in path_parts:
      path_parts.append(required)
  os.environ["PATH"] = os.pathsep.join(path_parts)


class WholeBufferCopyProbe(nn.Module):
  """Negative control: update state through full-buffer copy after concat."""

  def __init__(self, dtype: torch.dtype):
    super().__init__()
    self.register_buffer(STATE_NAME, torch.zeros(STATE_SHAPE, dtype=dtype))

  def forward(self, new_key: torch.Tensor) -> torch.Tensor:
    """Attempt a full-buffer copy update, which coremltools rejects."""
    cache = getattr(self, STATE_NAME)
    updated = torch.cat([cache[:, 1:], new_key.to(cache.dtype)], dim=1)
    cache.copy_(updated)
    return updated[:, -1].to(torch.float32)


class SliceUpdateProbe(nn.Module):
  """Positive control: update one fixed cache slice in place."""

  def __init__(self, dtype: torch.dtype):
    super().__init__()
    self.register_buffer(STATE_NAME, torch.zeros(STATE_SHAPE, dtype=dtype))

  def forward(self, new_key: torch.Tensor) -> torch.Tensor:
    """Write a new K/V slice into the Core ML state buffer."""
    cache = getattr(self, STATE_NAME)
    cache[:, 0:1] = new_key.to(cache.dtype)
    return cache[:, 0].to(torch.float32)


def _trace(module: nn.Module) -> torch.jit.ScriptModule:
  """Trace the probe with the fixed MRT2 K/V update input shape."""
  return torch.jit.trace(module.eval(), torch.zeros(NEW_KEY_SHAPE, dtype=torch.float32))


def _convert_probe(
    *,
    module: nn.Module,
    state_dtype: np.dtype,
    package_path: Path,
) -> tuple[ct.models.MLModel, dict[str, Any]]:
  """Convert a probe and return the Core ML model plus captured diagnostics."""
  if package_path.exists():
    shutil.rmtree(package_path)
  traced = _trace(module)
  stderr_buffer = io.StringIO()
  start = time.perf_counter()
  with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    with contextlib.redirect_stderr(stderr_buffer):
      mlmodel = ct.convert(
          traced,
          convert_to="mlprogram",
          inputs=[
              ct.TensorType(
                  name="new_key",
                  shape=NEW_KEY_SHAPE,
                  dtype=np.float32,
              )
          ],
          outputs=[ct.TensorType(name="updated_key")],
          states=[
              ct.StateType(
                  wrapped_type=ct.TensorType(
                      shape=STATE_SHAPE,
                      dtype=state_dtype,
                  ),
                  name=STATE_NAME,
              )
          ],
          compute_precision=ct.precision.FLOAT16,
          minimum_deployment_target=ct.target.iOS18,
      )
  convert_seconds = time.perf_counter() - start
  mlmodel.save(str(package_path))
  return mlmodel, {
      "convert_seconds": convert_seconds,
      "warnings": [
          {
              "category": warning.category.__name__,
              "message": str(warning.message),
          }
          for warning in caught_warnings
      ],
      "stderr": stderr_buffer.getvalue().strip(),
  }


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


def _attempt(name: str, fn) -> dict[str, Any]:
  """Run one probe attempt and capture exception text as data."""
  try:
    return {"ok": True, "name": name, "result": fn()}
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return {
        "ok": False,
        "name": name,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
  """Run conversion probes and return a machine-readable report."""
  _ensure_coreml_runtime_path()
  model_dir = Path(args.model_dir)
  model_dir.mkdir(parents=True, exist_ok=True)

  def whole_buffer_copy_fp16() -> dict[str, Any]:
    _convert_probe(
        module=WholeBufferCopyProbe(torch.float16),
        state_dtype=np.float16,
        package_path=model_dir / "temporal_kv_whole_buffer_copy_probe.mlpackage",
    )
    return {"unexpected": "whole-buffer copy conversion succeeded"}

  def slice_update_fp32() -> dict[str, Any]:
    _convert_probe(
        module=SliceUpdateProbe(torch.float32),
        state_dtype=np.float32,
        package_path=model_dir / "temporal_kv_slice_update_fp32_probe.mlpackage",
    )
    return {"unexpected": "FP32 state conversion succeeded"}

  def slice_update_fp16() -> dict[str, Any]:
    package_path = model_dir / "temporal_kv_slice_update_fp16_probe.mlpackage"
    compiled_path = model_dir / "temporal_kv_slice_update_fp16_probe.mlmodelc"
    mlmodel, diagnostics = _convert_probe(
        module=SliceUpdateProbe(torch.float16),
        state_dtype=np.float16,
        package_path=package_path,
    )
    state = mlmodel.make_state()
    prediction = mlmodel.predict(
        {"new_key": np.ones(NEW_KEY_SHAPE, dtype=np.float32)},
        state=state,
    )
    updated_key = np.asarray(prediction["updated_key"], dtype=np.float32)
    compile_report = _compile_model(package_path, compiled_path) if args.compile else None
    return {
        "package_path": str(package_path),
        "prediction_shape": list(updated_key.shape),
        "prediction_mean": float(np.mean(updated_key)),
        "conversion": diagnostics,
        "compile": compile_report,
    }

  return {
      "schema": "mrt2-temporal-kv-state-probe-v1",
      "state_name": STATE_NAME,
      "state_shape": list(STATE_SHAPE),
      "new_key_shape": list(NEW_KEY_SHAPE),
      "deployment_target": "iOS18",
      "compute_precision": "FLOAT16",
      "whole_buffer_copy_fp16": _attempt(
          "whole_buffer_copy_fp16",
          whole_buffer_copy_fp16,
      ),
      "slice_update_fp32_state": _attempt(
          "slice_update_fp32_state",
          slice_update_fp32,
      ),
      "slice_update_fp16_state": _attempt(
          "slice_update_fp16_state",
          slice_update_fp16,
      ),
  }


def parse_args() -> argparse.Namespace:
  """Parse command-line flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
  parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
  parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
  parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  args = parse_args()
  report = run_probe(args)
  report_dir = Path(args.report_dir)
  report_dir.mkdir(parents=True, exist_ok=True)
  report_path = report_dir / args.report_name
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"Wrote {report_path}")
  print(
      "FP16 slice update state ok: "
      f"{report['slice_update_fp16_state']['ok']}"
  )


if __name__ == "__main__":
  main()
