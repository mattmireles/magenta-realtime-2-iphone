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

"""Validate a host-owned K/V carry MRT2 temporal-body Core ML export."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch

from mrt2_coreml.depthformer_wrapper import (
    MRT2_HEAD_DIM,
    MRT2_LOCAL_WINDOW_FRAMES,
    MRT2_TEMPORAL_HEADS,
)
from mrt2_coreml.temporal_body_wrapper import TemporalBodyCoreMLCarryWrapper
from validate_temporal_body import (
    SOURCE_INPUT_NAME,
    TEMPORAL_INPUT_NAME,
    _load_tokens,
    _metrics,
    _temporal_mlx_fixture,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_TEMPLATE = (
    REPO_ROOT
    / "build"
    / "models"
    / "mrt2_temporal_body_carry_{frames:02d}.mlpackage"
)
DEFAULT_METADATA_TEMPLATE = (
    REPO_ROOT
    / "build"
    / "models"
    / "mrt2_temporal_body_carry_{frames:02d}_export_metadata.json"
)
DEFAULT_TOKENS_PATH = (
    REPO_ROOT / "fixtures" / "generated_tokens_unique.npy"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "validation"
DEFAULT_REPORT_TEMPLATE = "mrt2_temporal_body_carry_{frames:02d}_validation.json"
DEFAULT_SUMMARY_TEMPLATE = "mrt2_temporal_body_carry_{frames:02d}_validation.md"
OUTPUT_NAME = "temporal_outputs"


def _ensure_coreml_runtime_path() -> None:
  """Give coremltools access to macOS helper tools when Codex PATH is thin."""
  path_parts = os.environ.get("PATH", "").split(os.pathsep)
  for required in ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]:
    if required not in path_parts:
      path_parts.append(required)
  os.environ["PATH"] = os.pathsep.join(path_parts)


def _empty_cache_inputs(dtype: np.dtype = np.float16) -> dict[str, np.ndarray]:
  """Return zero host-owned K/V cache tensors keyed by Core ML input name."""
  shape = (
      1,
      MRT2_LOCAL_WINDOW_FRAMES,
      MRT2_TEMPORAL_HEADS,
      MRT2_HEAD_DIM,
  )
  return {
      name: np.zeros(shape, dtype=dtype)
      for name in TemporalBodyCoreMLCarryWrapper.cache_input_names()
  }


def validate(args: argparse.Namespace) -> dict[str, Any]:
  """Run carry validation and return a machine-readable report."""
  _ensure_coreml_runtime_path()
  model_path = Path(args.model_template.format(frames=args.frames))
  metadata_path = Path(args.metadata_template.format(frames=args.frames))
  if not model_path.exists():
    raise FileNotFoundError(
        f"Core ML package not found: {model_path}. "
        "Run scripts/convert_mrt2_temporal_body_carry_coreml.py first."
    )
  tokens = _load_tokens(Path(args.tokens_path), args.frames)
  temporal_inputs, source_encoded, mlx_output, _, _ = _temporal_mlx_fixture(tokens=tokens)

  pytorch_model = TemporalBodyCoreMLCarryWrapper(frame_count=args.frames).eval()
  torch_cache_inputs = [
      torch.zeros(
          (
              1,
              MRT2_LOCAL_WINDOW_FRAMES,
              MRT2_TEMPORAL_HEADS,
              MRT2_HEAD_DIM,
          ),
          dtype=torch.float16,
      )
      for _ in pytorch_model.cache_input_names()
  ]
  with torch.no_grad():
    pytorch_outputs = pytorch_model(
        torch.from_numpy(temporal_inputs),
        torch.from_numpy(source_encoded),
        *torch_cache_inputs,
    )
  pytorch_temporal = pytorch_outputs[0].detach().cpu().numpy().astype(np.float32)

  coreml_model = ct.models.MLModel(
      str(model_path),
      compute_units=ct.ComputeUnit.CPU_ONLY,
  )
  coreml_inputs: dict[str, Any] = {
      TEMPORAL_INPUT_NAME: temporal_inputs,
      SOURCE_INPUT_NAME: source_encoded,
      **_empty_cache_inputs(),
  }
  start = time.perf_counter()
  coreml_outputs = coreml_model.predict(coreml_inputs)
  predict_ms = (time.perf_counter() - start) * 1000.0
  coreml_temporal = np.asarray(coreml_outputs[OUTPUT_NAME], dtype=np.float32)

  update_shapes = {
      name: list(np.asarray(coreml_outputs[name]).shape)
      for name in TemporalBodyCoreMLCarryWrapper.cache_update_output_names()
  }
  update_finite = {
      name: bool(np.isfinite(np.asarray(coreml_outputs[name])).all())
      for name in TemporalBodyCoreMLCarryWrapper.cache_update_output_names()
  }

  export_metadata: dict[str, Any] | None = None
  if metadata_path.exists():
    export_metadata = json.loads(metadata_path.read_text())

  return {
      "schema": "mrt2-temporal-body-carry-coreml-validation-v1",
      "boundary": "host_owned_kv_cache_inputs_and_update_outputs",
      "frames": args.frames,
      "model_path": str(model_path),
      "metadata_path": str(metadata_path) if metadata_path.exists() else None,
      "cache_input_count": len(TemporalBodyCoreMLCarryWrapper.cache_input_names()),
      "cache_update_output_count": len(
          TemporalBodyCoreMLCarryWrapper.cache_update_output_names()
      ),
      "pytorch_vs_mlx": _metrics(pytorch_temporal, mlx_output),
      "coreml_vs_pytorch": _metrics(coreml_temporal, pytorch_temporal),
      "coreml_vs_mlx": _metrics(coreml_temporal, mlx_output),
      "cache_update_shapes": update_shapes,
      "cache_updates_all_finite": all(update_finite.values()),
      "timing_smoke": {
          "scope": "Python Core ML CPU_ONLY single carry predict, not device timing",
          "predict_ms": float(predict_ms),
      },
      "export_compile": None if export_metadata is None else export_metadata.get("compile"),
      "known_limits": [
          "This first carry proof uses empty host-owned caches and history_length=0.",
          "It validates a no-wrap burst, not full rolling host cache placement.",
          "Depth-body logits remain in a separate Core ML package.",
      ],
  }


def _write_summary(report: dict[str, Any], summary_path: Path) -> None:
  """Write a short markdown summary beside the JSON report."""
  cml_mlx = report["coreml_vs_mlx"]
  lines = [
      "# MRT2 Temporal Body Carry Core ML Validation",
      "",
      f"- Boundary: `{report['boundary']}`",
      f"- Frames: {report['frames']}",
      f"- Cache inputs: {report['cache_input_count']}",
      f"- Cache update outputs: {report['cache_update_output_count']}",
      f"- Core ML vs MLX max error: {cml_mlx['max_abs_error']:.10f}",
      f"- Core ML vs MLX mean error: {cml_mlx['mean_abs_error']:.10f}",
      f"- Core ML vs MLX correlation: {cml_mlx['correlation']:.12f}",
      f"- Core ML CPU_ONLY predict smoke: {report['timing_smoke']['predict_ms']:.3f} ms",
      f"- Cache updates all finite: {report['cache_updates_all_finite']}",
      "",
      "Known limits:",
  ]
  lines.extend(f"- {limit}" for limit in report["known_limits"])
  lines.append("")
  summary_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
  """Parse CLI flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frames", type=int, default=2)
  parser.add_argument("--model-template", default=str(DEFAULT_MODEL_TEMPLATE))
  parser.add_argument("--metadata-template", default=str(DEFAULT_METADATA_TEMPLATE))
  parser.add_argument("--tokens-path", default=str(DEFAULT_TOKENS_PATH))
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--report-template", default=DEFAULT_REPORT_TEMPLATE)
  parser.add_argument("--summary-template", default=DEFAULT_SUMMARY_TEMPLATE)
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  args = parse_args()
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  report = validate(args)
  report_path = output_dir / args.report_template.format(frames=args.frames)
  summary_path = output_dir / args.summary_template.format(frames=args.frames)
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  _write_summary(report, summary_path)
  print(f"Wrote {report_path}")
  print(f"Wrote {summary_path}")
  print(
      "Core ML vs MLX max error "
      f"{report['coreml_vs_mlx']['max_abs_error']:.10f}"
  )


if __name__ == "__main__":
  main()
