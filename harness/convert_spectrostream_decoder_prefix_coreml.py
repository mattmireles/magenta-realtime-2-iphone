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

"""Convert a SpectroStream decoder prefix to Core ML for ANE placement probes."""

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

from magenta_rt import paths
from magenta_rt.coreml.spectrostream_decoder_wrapper import (
    SPECTROSTREAM_EMBEDDING_DIM,
    SpectroStreamDecoderNCHWParallelPrefixWrapper,
    SpectroStreamDecoderPrefixWrapper,
)
from scripts.convert_spectrostream_decoder_conv_coreml import (
    DEFAULT_UNIQUE_TOKENS_PATH,
    INPUT_NAME,
    _build_spectrostream,
    _compile_model,
    _git_commit,
    _host_lookup_sum,
    _load_codebooks,
    _load_decoder_codes,
    _metrics,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Scratchpad" / "coreml_proof_models"
DEFAULT_REPORT_DIR = REPO_ROOT / "Scratchpad" / "coreml_proof_validation"
DEFAULT_PACKAGE_TEMPLATE = "spectrostream_decoder_prefix_{prefix:02d}_{frames:02d}.mlpackage"
DEFAULT_COMPILED_TEMPLATE = "spectrostream_decoder_prefix_{prefix:02d}_{frames:02d}.mlmodelc"
DEFAULT_METADATA_TEMPLATE = (
    "spectrostream_decoder_prefix_{prefix:02d}_{frames:02d}_export_metadata.json"
)
DEFAULT_REPORT_TEMPLATE = "spectrostream_decoder_prefix_{prefix:02d}_{frames:02d}_validation.json"
DEFAULT_SUMMARY_TEMPLATE = "spectrostream_decoder_prefix_{prefix:02d}_{frames:02d}_validation.md"
OUTPUT_NAME = "decoder_prefix"
DEPLOYMENT_TARGET = "iOS18"
COMPUTE_PRECISION = "FLOAT16"


def _format(template: str, args: argparse.Namespace) -> str:
  """Format artifact templates for the selected prefix and frame count."""
  prefix_layers = _selected_prefix_layers(args)
  return template.format(prefix=prefix_layers, frames=args.frames)


def _selected_prefix_layers(args: argparse.Namespace) -> int:
  """Return the number of decoder layers included in the selected wrapper."""
  if args.nchw_parallel_layer is not None:
    return int(args.nchw_parallel_layer) + 1
  return int(args.prefix_layers)


def convert(args: argparse.Namespace) -> dict[str, Any]:
  """Trace, convert, compile, and validate a decoder prefix package."""
  output_dir = Path(args.output_dir)
  report_dir = Path(args.report_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  report_dir.mkdir(parents=True, exist_ok=True)
  package_path = output_dir / _format(args.package_template, args)
  compiled_path = output_dir / _format(args.compiled_template, args)
  metadata_path = output_dir / _format(args.metadata_template, args)
  report_path = report_dir / _format(args.report_template, args)
  summary_path = report_dir / _format(args.summary_template, args)

  checkpoint_path = Path(args.checkpoint_path)
  decoder_codes = _load_decoder_codes(Path(args.unique_tokens_path), args.frames)
  embeddings = _host_lookup_sum(_load_codebooks(checkpoint_path), decoder_codes)
  soundstream = _build_spectrostream(checkpoint_path)
  layout_mode = "channel-last"
  if args.nchw_parallel_layer is None:
    wrapper = SpectroStreamDecoderPrefixWrapper.from_mlx_decoder(
        soundstream.decoder,
        layer_count=args.prefix_layers,
    ).eval()
  else:
    layout_mode = "nchw-parallel"
    wrapper = SpectroStreamDecoderNCHWParallelPrefixWrapper.from_mlx_decoder(
        soundstream.decoder,
        parallel_layer=args.nchw_parallel_layer,
    ).eval()

  example_input = torch.from_numpy(embeddings[np.newaxis]).to(torch.float32)
  with torch.no_grad():
    torch_prefix = wrapper(example_input).detach().cpu().numpy().astype(np.float32)

  start_trace = time.perf_counter()
  traced = torch.jit.trace(wrapper, example_input)
  trace_seconds = time.perf_counter() - start_trace

  compute_precision = getattr(ct.precision, args.compute_precision)
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
                  name=INPUT_NAME,
                  shape=(1, args.frames, SPECTROSTREAM_EMBEDDING_DIM),
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

  coreml_prefix = None
  coreml_error = None
  coreml_metrics = None
  if args.predict:
    try:
      coreml_model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY)
      coreml_prefix = np.asarray(
          coreml_model.predict({INPUT_NAME: embeddings[np.newaxis].astype(np.float32)})[
              OUTPUT_NAME
          ],
          dtype=np.float32,
      )
      coreml_metrics = _metrics(coreml_prefix, torch_prefix)
    except Exception as exc:  # Preserve Core ML predict failures as evidence.
      coreml_error = repr(exc)

  report: dict[str, Any] = {
      "schema": "spectrostream-decoder-prefix-coreml-export-v1",
      "source_commit": _git_commit(),
      "boundary": "host RVQ embeddings to intermediate SpectroStream decoder prefix tensor",
      "checkpoint_path": str(checkpoint_path),
      "frames": int(args.frames),
      "prefix_layers": _selected_prefix_layers(args),
      "layout_mode": layout_mode,
      "nchw_parallel_layer": (
          int(args.nchw_parallel_layer) if args.nchw_parallel_layer is not None else None
      ),
      "inputs": [
          {
              "name": INPUT_NAME,
              "shape": [1, args.frames, SPECTROSTREAM_EMBEDDING_DIM],
              "dtype": "float32",
          }
      ],
      "outputs": [
          {
              "name": OUTPUT_NAME,
              "shape": list(torch_prefix.shape),
              "dtype": "Core ML selected",
          }
      ],
      "conversion": {
          "convert_to": "mlprogram",
          "compute_precision": args.compute_precision,
          "minimum_deployment_target": DEPLOYMENT_TARGET,
          "trace_seconds": trace_seconds,
          "convert_seconds": convert_seconds,
          "coreml_predict_error": coreml_error,
      },
      "artifacts": {
          "mlpackage": str(package_path),
          "mlmodelc": str(compiled_path) if compiled_path.exists() else None,
          "metadata": str(metadata_path),
          "report": str(report_path),
          "summary": str(summary_path),
      },
      "torch_prefix": {
          "finite": bool(np.isfinite(torch_prefix).all()),
          "min": float(np.min(torch_prefix)),
          "max": float(np.max(torch_prefix)),
      },
      "coreml_vs_torch_prefix": coreml_metrics,
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
          "Prefix placement probe only; not a complete decoder replacement.",
          "RVQ lookup, decoder tail, iSTFT, and overlap-add remain outside this package.",
      ],
  }

  metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  _write_summary(report, summary_path)
  return report


def _write_summary(report: dict[str, Any], summary_path: Path) -> None:
  """Write a short markdown validation summary."""
  metrics = report["coreml_vs_torch_prefix"] or {}
  lines = [
      "# SpectroStream Decoder Prefix Core ML Probe",
      "",
      f"- Frames: {report['frames']}",
      f"- Prefix layers: {report['prefix_layers']}",
      f"- Layout mode: `{report['layout_mode']}`",
      f"- Output shape: `{report['outputs'][0]['shape']}`",
      f"- Torch output finite: {report['torch_prefix']['finite']}",
      f"- Torch output range: {report['torch_prefix']['min']:.6f} to {report['torch_prefix']['max']:.6f}",
  ]
  if metrics:
    lines.extend(
        [
            f"- Core ML vs Torch finite ratio: {metrics['finite_ratio']:.6f}",
            f"- Core ML vs Torch max error: {metrics['max_abs_error']:.10f}",
            f"- Core ML vs Torch mean error: {metrics['mean_abs_error']:.10f}",
            f"- Core ML vs Torch correlation: {metrics['correlation']:.12f}",
        ]
    )
  else:
    lines.append(f"- Core ML predict error: `{report['conversion']['coreml_predict_error']}`")
  lines.extend(
      [
          f"- MLPackage: `{report['artifacts']['mlpackage']}`",
          f"- MLMODELC: `{report['artifacts']['mlmodelc']}`",
          "",
          "Known limits:",
      ]
  )
  lines.extend(f"- {limit}" for limit in report["known_limits"])
  lines.append("")
  summary_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
  """Parse command-line flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
  parser.add_argument(
      "--checkpoint-path",
      default=str(paths.checkpoints_dir() / "mrt2_small.safetensors"),
  )
  parser.add_argument("--unique-tokens-path", default=str(DEFAULT_UNIQUE_TOKENS_PATH))
  parser.add_argument("--frames", type=int, default=5)
  parser.add_argument("--prefix-layers", type=int, default=4)
  parser.add_argument(
      "--nchw-parallel-layer",
      type=int,
      default=None,
      help=(
          "Select one ParallelChannels layer to run in NCHW internally. "
          "The exported prefix includes layers through this index."
      ),
  )
  parser.add_argument("--package-template", default=DEFAULT_PACKAGE_TEMPLATE)
  parser.add_argument("--compiled-template", default=DEFAULT_COMPILED_TEMPLATE)
  parser.add_argument("--metadata-template", default=DEFAULT_METADATA_TEMPLATE)
  parser.add_argument("--report-template", default=DEFAULT_REPORT_TEMPLATE)
  parser.add_argument("--summary-template", default=DEFAULT_SUMMARY_TEMPLATE)
  parser.add_argument(
      "--compute-precision",
      choices=("FLOAT16", "FLOAT32"),
      default=COMPUTE_PRECISION,
  )
  parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--predict", action=argparse.BooleanOptionalAction, default=True)
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  report = convert(parse_args())
  print(f"Saved {report['artifacts']['mlpackage']}")
  if report["artifacts"]["mlmodelc"] is not None:
    print(f"Compiled {report['artifacts']['mlmodelc']}")
  if report["warnings"]["compile_error"]:
    print(f"Compile failed: {report['warnings']['compile_error']}")
  print(f"Wrote {report['artifacts']['report']}")


if __name__ == "__main__":
  main()
