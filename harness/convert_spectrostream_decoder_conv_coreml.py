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

"""Convert the SpectroStream decoder conv boundary to a Core ML GPU baseline.

The exported graph starts after host CPU RVQ lookup and stops before host iSTFT.
This keeps the known gather and iSTFT fallback risks out of the Core ML graph,
matching the split recommended by ``README/Guides/RVQ-codec-decoder-guide.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import safetensors.flax as safetensors_flax
import torch

from magenta_rt import paths
from magenta_rt.coreml.sampling import MRT2_RVQ_LEVELS, unique_token_to_raw_code
from magenta_rt.coreml.spectrostream_decoder_wrapper import (
    SPECTROSTREAM_EMBEDDING_DIM,
    SPECTROSTREAM_NUM_BINS,
    SPECTROSTREAM_OUTPUT_CHANNELS,
    SpectroStreamDecoderConvWrapper,
    SpectroStreamDecoderNCHWParallelWrapper,
    apply_fp16_safe_rescale,
    count_torch_conv_layers,
    decoder_output_frames,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIQUE_TOKENS_PATH = (
    REPO_ROOT / "Scratchpad" / "coreml_proof_fixtures" / "generated_tokens_unique.npy"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Scratchpad" / "coreml_proof_models"
DEFAULT_REPORT_DIR = REPO_ROOT / "Scratchpad" / "coreml_proof_validation"
DEFAULT_PACKAGE_NAME = "spectrostream_decoder_conv_gpu.mlpackage"
DEFAULT_COMPILED_NAME = "spectrostream_decoder_conv_gpu.mlmodelc"
DEFAULT_METADATA_NAME = "spectrostream_decoder_conv_gpu_export_metadata.json"
DEFAULT_REPORT_NAME = "spectrostream_decoder_conv_gpu_validation.json"
DEFAULT_SUMMARY_NAME = "spectrostream_decoder_conv_gpu_validation.md"
QUANTIZER_KEY = "params/soundstream/quantizer/embedding"
INPUT_NAME = "decoder_embeddings"
OUTPUT_NAME = "decoder_stft"
COMPUTE_PRECISION = "FLOAT32"
DEFAULT_WARMUP_PREDICTIONS = 1
DEFAULT_TIMED_PREDICTIONS = 5


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
  return output.strip()


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
  """Return tensor parity metrics for two same-shaped arrays."""
  finite = np.isfinite(actual) & np.isfinite(expected)
  if not np.all(finite):
    return {
        "shape": list(actual.shape),
        "finite_count": int(np.sum(finite)),
        "total_count": int(actual.size),
        "finite_ratio": float(np.sum(finite) / actual.size),
        "max_abs_error": math.nan,
        "mean_abs_error": math.nan,
        "rms_error": math.nan,
        "snr_db": math.nan,
        "correlation": math.nan,
    }
  delta = actual.astype(np.float64) - expected.astype(np.float64)
  rms_signal = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
  rms_error = float(np.sqrt(np.mean(delta ** 2)))
  snr_db = math.inf if rms_error == 0.0 else 20.0 * math.log10(rms_signal / rms_error)
  flat_actual = actual.reshape(-1).astype(np.float64)
  flat_expected = expected.reshape(-1).astype(np.float64)
  corr = float(np.corrcoef(flat_actual, flat_expected)[0, 1])
  return {
      "shape": list(actual.shape),
      "finite_count": int(actual.size),
      "total_count": int(actual.size),
      "finite_ratio": 1.0,
      "max_abs_error": float(np.max(np.abs(delta))),
      "mean_abs_error": float(np.mean(np.abs(delta))),
      "rms_error": rms_error,
      "snr_db": snr_db,
      "correlation": corr,
  }


def _log_spectral_distance(actual: np.ndarray, expected: np.ndarray) -> float:
  """Compute RMS log-magnitude distance over complex pre-iSTFT bins."""
  if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
    return math.nan
  actual_complex = actual.astype(np.float64).reshape(actual.shape[:3] + (2, 2))
  expected_complex = expected.astype(np.float64).reshape(expected.shape[:3] + (2, 2))
  actual_mag = np.sqrt(np.sum(actual_complex ** 2, axis=-1))
  expected_mag = np.sqrt(np.sum(expected_complex ** 2, axis=-1))
  delta_db = 20.0 * (
      np.log10(np.maximum(actual_mag, 1e-7)) - np.log10(np.maximum(expected_mag, 1e-7))
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


def _decode_mlx_decoder(soundstream: object, embeddings: np.ndarray) -> np.ndarray:
  """Run the MLX decoder only, excluding inverse STFT."""
  import magenta_rt  # noqa: F401
  import mlx.core as mx
  import sequence_layers.mlx as sl

  sequence = sl.Sequence(
      mx.array(embeddings[np.newaxis], dtype=mx.float32),
      mx.ones((1, embeddings.shape[0]), dtype=mx.bool_),
  )
  output = soundstream.decoder.layer(sequence)
  mx.eval(output.values)
  return np.asarray(output.values, dtype=np.float32)


def _predict_coreml(
    package_path: Path,
    embeddings: np.ndarray,
    compute_units: ct.ComputeUnit,
    warmups: int,
    repeats: int,
) -> tuple[np.ndarray | None, str | None, dict[str, Any] | None]:
  """Run a Core ML prediction when the saved package can be loaded."""
  try:
    mlmodel = ct.models.MLModel(str(package_path), compute_units=compute_units)
    model_input = {INPUT_NAME: embeddings[np.newaxis].astype(np.float32)}
    prediction = None
    for _ in range(warmups):
      prediction = mlmodel.predict(model_input)
    timings = []
    for _ in range(repeats):
      start = time.perf_counter()
      prediction = mlmodel.predict(model_input)
      timings.append((time.perf_counter() - start) * 1000.0)
    timing = {
        "warmups": int(warmups),
        "repeats": int(repeats),
        "p50": float(np.percentile(timings, 50)),
        "p90": float(np.percentile(timings, 90)),
        "p99": float(np.percentile(timings, 99)),
        "min": float(np.min(timings)),
        "max": float(np.max(timings)),
        "all": [float(value) for value in timings],
    }
    return np.asarray(prediction[OUTPUT_NAME], dtype=np.float32), None, timing
  except Exception as exc:  # Core ML load/predict failures must be preserved.
    return None, repr(exc), None


def convert(args: argparse.Namespace) -> dict[str, Any]:
  """Trace, convert, compile, and validate the SpectroStream decoder baseline."""
  output_dir = Path(args.output_dir)
  report_dir = Path(args.report_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  report_dir.mkdir(parents=True, exist_ok=True)
  package_path = output_dir / args.package_name
  compiled_path = output_dir / args.compiled_name
  metadata_path = output_dir / args.metadata_name
  report_path = report_dir / args.report_name
  summary_path = report_dir / args.summary_name

  checkpoint_path = Path(args.checkpoint_path)
  decoder_codes = _load_decoder_codes(Path(args.unique_tokens_path), args.frames)
  codebooks = _load_codebooks(checkpoint_path)
  embeddings = _host_lookup_sum(codebooks, decoder_codes)
  soundstream = _build_spectrostream(checkpoint_path)
  mlx_stft = _decode_mlx_decoder(soundstream, embeddings)

  layout_mode = "channel-last"
  if args.nchw_parallel_layer is None:
    wrapper = SpectroStreamDecoderConvWrapper.from_mlx_decoder(soundstream.decoder).eval()
  else:
    layout_mode = "nchw-parallel"
    wrapper = SpectroStreamDecoderNCHWParallelWrapper.from_mlx_decoder(
        soundstream.decoder,
        parallel_layer=args.nchw_parallel_layer,
    ).eval()
  example_input = torch.from_numpy(embeddings[np.newaxis]).to(torch.float32)

  # FP16 path: rescale the hot mid-network BEFORE tracing so the converted
  # graph is FP16-overflow-safe (see apply_fp16_safe_rescale docstring; the
  # 2026-06-08 plain-FLOAT16 export hit finite_ratio=0.843). The transform is
  # exact in FP32, so the pytorch_vs_mlx parity below also validates it.
  fp16_rescale_info = None
  if args.fp16_rescale:
    if args.nchw_parallel_layer is None:
      raise ValueError("--fp16-rescale requires --nchw-parallel-layer")
    fp16_rescale_info = apply_fp16_safe_rescale(wrapper, example_input)

  with torch.no_grad():
    torch_stft = wrapper(example_input).detach().cpu().numpy().astype(np.float32)

  pytorch_metrics = _metrics(torch_stft, mlx_stft)
  pytorch_lsd = _log_spectral_distance(torch_stft, mlx_stft)

  start_trace = time.perf_counter()
  traced = torch.jit.trace(wrapper, example_input)
  trace_seconds = time.perf_counter() - start_trace

  compute_precision = getattr(ct.precision, args.compute_precision)
  stderr_buffer = io.StringIO()
  start_convert = time.perf_counter()
  conversion_error = None
  caught_warnings: list[warnings.WarningMessage] = []
  mlmodel = None
  try:
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
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )
  except Exception as exc:  # Preserve exact converter failure as proof data.
    conversion_error = repr(exc)
  convert_seconds = time.perf_counter() - start_convert

  if package_path.exists():
    shutil.rmtree(package_path)
  if mlmodel is not None:
    mlmodel.save(str(package_path))

  compile_output = None
  compile_error = None
  if mlmodel is not None and args.compile:
    try:
      compile_output = _compile_model(package_path, compiled_path)
    except (OSError, subprocess.CalledProcessError) as exc:
      compile_error = str(exc)

  coreml_stft = None
  coreml_error = None
  coreml_timing = None
  coreml_metrics = None
  coreml_lsd = None
  if mlmodel is not None and package_path.exists() and args.predict:
    coreml_stft, coreml_error, coreml_timing = _predict_coreml(
        package_path,
        embeddings,
        ct.ComputeUnit.CPU_AND_GPU,
        args.warmup_predictions,
        args.timed_predictions,
    )
    if coreml_stft is not None:
      coreml_metrics = _metrics(coreml_stft, mlx_stft)
      coreml_lsd = _log_spectral_distance(coreml_stft, mlx_stft)

  expected_output_shape = [
      1,
      decoder_output_frames(args.frames),
      SPECTROSTREAM_NUM_BINS,
      SPECTROSTREAM_OUTPUT_CHANNELS,
  ]
  report: dict[str, Any] = {
      "schema": "spectrostream-decoder-conv-coreml-export-v1",
      "source_commit": _git_commit(),
      "checkpoint_path": str(checkpoint_path),
      "boundary": "host RVQ embeddings to pre-iSTFT SpectroStream decoder tensor",
      "frames": int(args.frames),
      "decoder_codes": {
          "shape": list(decoder_codes.shape),
          "min": int(np.min(decoder_codes)),
          "max": int(np.max(decoder_codes)),
      },
      "inputs": [
          {
              "name": INPUT_NAME,
              "shape": [1, int(args.frames), SPECTROSTREAM_EMBEDDING_DIM],
              "dtype": "float32",
              "source": "host CPU RVQ lookup sum",
          }
      ],
      "outputs": [
          {
              "name": OUTPUT_NAME,
              "shape": expected_output_shape,
              "dtype": "float32 model boundary, converted with selected compute precision",
              "consumer": "host inverse STFT / overlap-add",
          }
      ],
      "wrapper": {
          "class": f"{type(wrapper).__module__}.{type(wrapper).__name__}",
          "layout_mode": layout_mode,
          "nchw_parallel_layer": (
              int(args.nchw_parallel_layer) if args.nchw_parallel_layer is not None else None
          ),
          "conv_layers": count_torch_conv_layers(wrapper),
          "weight_norm": "standard MRT2 SpectroStream config has global_weight_norm=False; no fusion needed",
          "fp16_rescale": fp16_rescale_info,
      },
      "pytorch_vs_mlx_decoder": {
          **pytorch_metrics,
          "log_spectral_distance_db": pytorch_lsd,
      },
      "coreml_vs_mlx_decoder": None
      if coreml_metrics is None
      else {
          **coreml_metrics,
          "log_spectral_distance_db": coreml_lsd,
          "predict_timing_ms_cpu_and_gpu": coreml_timing,
      },
      "conversion": {
          "convert_to": "mlprogram",
          "compute_precision": args.compute_precision,
          "minimum_deployment_target": "iOS18",
          "compute_units_for_predict": "CPU_AND_GPU",
          "trace_seconds": trace_seconds,
          "convert_seconds": convert_seconds,
          "conversion_error": conversion_error,
          "coreml_predict_error": coreml_error,
      },
      "artifacts": {
          "mlpackage": str(package_path) if package_path.exists() else None,
          "mlmodelc": str(compiled_path) if compiled_path.exists() else None,
          "metadata": str(metadata_path),
          "report": str(report_path),
          "summary": str(summary_path),
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
          "Exports a fixed chunk decoder conv baseline, not per-frame streaming Core ML state.",
          "RVQ lookup remains host CPU owned.",
          "Inverse STFT and PCM overlap state remain host owned.",
          "CPU_AND_GPU predict timing on Mac is only a local smoke signal; iPhone profiling is Phase 5 authority.",
      ],
  }

  metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  _write_summary(report, summary_path)
  return report


def _write_summary(report: dict[str, Any], summary_path: Path) -> None:
  """Write a short markdown summary beside the JSON report."""
  pytorch = report["pytorch_vs_mlx_decoder"]
  coreml = report["coreml_vs_mlx_decoder"]
  lines = [
      "# SpectroStream Decoder Conv Core ML Baseline",
      "",
      f"- Frames: {report['frames']}",
      f"- Boundary: {report['boundary']}",
      f"- Layout mode: `{report['wrapper']['layout_mode']}`",
      f"- Input shape: `{report['inputs'][0]['shape']}`",
      f"- Output shape: `{report['outputs'][0]['shape']}`",
      f"- Conv layers: `{report['wrapper']['conv_layers']}`",
      f"- Weight norm: {report['wrapper']['weight_norm']}",
      f"- PyTorch vs MLX max error: {pytorch['max_abs_error']:.10f}",
      f"- PyTorch vs MLX mean error: {pytorch['mean_abs_error']:.10f}",
      f"- PyTorch vs MLX SNR: {pytorch['snr_db']:.3f} dB",
      f"- PyTorch vs MLX log-spectral distance: {pytorch['log_spectral_distance_db']:.6f} dB",
  ]
  if coreml is None:
    lines.extend([
        f"- Core ML conversion error: `{report['conversion']['conversion_error']}`",
        f"- Core ML predict error: `{report['conversion']['coreml_predict_error']}`",
    ])
  else:
    lines.extend([
        f"- Core ML vs MLX max error: {coreml['max_abs_error']:.10f}",
        f"- Core ML vs MLX mean error: {coreml['mean_abs_error']:.10f}",
        f"- Core ML vs MLX SNR: {coreml['snr_db']:.3f} dB",
        f"- Core ML vs MLX log-spectral distance: {coreml['log_spectral_distance_db']:.6f} dB",
        "- Core ML CPU_AND_GPU predict smoke p50/p99: "
        f"{coreml['predict_timing_ms_cpu_and_gpu']['p50']:.3f} / "
        f"{coreml['predict_timing_ms_cpu_and_gpu']['p99']:.3f} ms",
    ])
  lines.extend([
      f"- MLPackage: `{report['artifacts']['mlpackage']}`",
      f"- MLMODELC: `{report['artifacts']['mlmodelc']}`",
      "",
      "Known limits:",
  ])
  lines.extend(f"- {limit}" for limit in report["known_limits"])
  lines.append("")
  summary_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
  """Parse command-line flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--checkpoint-path",
      default=str(paths.resolve_checkpoint("mrt2_small.safetensors")),
  )
  parser.add_argument("--unique-tokens-path", default=str(DEFAULT_UNIQUE_TOKENS_PATH))
  parser.add_argument("--frames", type=int, default=25)
  parser.add_argument(
      "--nchw-parallel-layer",
      type=int,
      default=None,
      help="Run one ParallelChannels layer in NCHW internally before resuming the decoder.",
  )
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
  parser.add_argument("--package-name", default=DEFAULT_PACKAGE_NAME)
  parser.add_argument("--compiled-name", default=DEFAULT_COMPILED_NAME)
  parser.add_argument("--metadata-name", default=DEFAULT_METADATA_NAME)
  parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
  parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
  parser.add_argument(
      "--compute-precision",
      choices=("FLOAT16", "FLOAT32"),
      default=COMPUTE_PRECISION,
  )
  parser.add_argument(
      "--fp16-rescale",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Rescale the hot mid-network (exact transform) so FLOAT16 "
          "conversion cannot overflow; required for ANE-resident decode. "
          "See apply_fp16_safe_rescale in spectrostream_decoder_wrapper.py."
      ),
  )
  parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--predict", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--warmup-predictions", type=int, default=DEFAULT_WARMUP_PREDICTIONS)
  parser.add_argument("--timed-predictions", type=int, default=DEFAULT_TIMED_PREDICTIONS)
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  report = convert(parse_args())
  print(f"Wrote {report['artifacts']['report']}")
  if report["artifacts"]["mlpackage"]:
    print(f"Saved {report['artifacts']['mlpackage']}")
  if report["artifacts"]["mlmodelc"]:
    print(f"Compiled {report['artifacts']['mlmodelc']}")
  if report["conversion"]["conversion_error"]:
    print(f"Convert failed: {report['conversion']['conversion_error']}")
  if report["conversion"]["coreml_predict_error"]:
    print(f"Core ML predict failed: {report['conversion']['coreml_predict_error']}")


if __name__ == "__main__":
  main()
