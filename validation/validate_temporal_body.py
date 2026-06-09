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

"""Validate a no-wrap unrolled MRT2 temporal-body Core ML export."""

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

from mrt2_coreml.sampling import (
    MRT2_RVQ_LEVELS,
    SamplingConfig,
    sample_rvq_frame_logits,
)
from mrt2_coreml.temporal_body_wrapper import (
    TemporalBodyCoreMLSlotWrapper,
    TemporalBodyCoreMLUnrolledWrapper,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
# Default layout: model packages downloaded from the Hugging Face repo
# (mattmireles/magenta-realtime-2-iphone) into ``models/`` at the repo root.
# The ``{frames}`` placeholder is kept for re-exported variants; the published
# MRT2TemporalBody.mlpackage is the frames=01 export under its public name.
DEFAULT_MODEL_TEMPLATE = REPO_ROOT / "models" / "MRT2TemporalBody.mlpackage"
DEFAULT_METADATA_TEMPLATE = (
    REPO_ROOT / "models" / "metadata" / "MRT2TemporalBody_export_metadata.json"
)
DEFAULT_TOKENS_PATH = (
    REPO_ROOT / "fixtures" / "generated_tokens_unique.npy"
)
DEFAULT_REFERENCE_NPZ_TEMPLATE = (
    REPO_ROOT / "fixtures" / "reference_temporal_unrolled.npz"
)
DEFAULT_DEPTH_BODY_MODEL_PATH = REPO_ROOT / "models" / "MRT2DepthBody.mlpackage"
DEFAULT_DEPTH_BODY_FP32_MODEL_PATH = (
    REPO_ROOT / "models" / "MRT2DepthBody_fp32_control.mlpackage"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "validation"
DEFAULT_REPORT_TEMPLATE = "mrt2_temporal_body_unrolled_{frames:02d}_validation.json"
DEFAULT_SUMMARY_TEMPLATE = "mrt2_temporal_body_unrolled_{frames:02d}_validation.md"
TEMPORAL_INPUT_NAME = "temporal_inputs"
SOURCE_INPUT_NAME = "source_encoded"
OUTPUT_NAME = "temporal_outputs"
DEPTH_BODY_INPUT_NAME = "depth_inputs"
DEPTH_BODY_OUTPUT_NAME = "depth_logits"


def _ensure_coreml_runtime_path() -> None:
  """Give coremltools access to macOS helper tools when Codex PATH is thin."""
  path_parts = os.environ.get("PATH", "").split(os.pathsep)
  for required in ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]:
    if required not in path_parts:
      path_parts.append(required)
  os.environ["PATH"] = os.pathsep.join(path_parts)


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


def _topk_report(actual: np.ndarray, expected: np.ndarray, ks: tuple[int, ...]) -> dict[str, Any]:
  """Report top-k agreement for full-vocabulary logits."""
  actual_flat = actual.reshape(-1, actual.shape[-1])
  expected_flat = expected.reshape(-1, expected.shape[-1])
  report: dict[str, Any] = {
      "top1_exact": float(
          np.mean(np.argmax(actual_flat, axis=-1) == np.argmax(expected_flat, axis=-1))
      ),
  }
  for k in ks:
    actual_topk = np.argpartition(actual_flat, -k, axis=-1)[:, -k:]
    expected_topk = np.argpartition(expected_flat, -k, axis=-1)[:, -k:]
    actual_sets = [set(row.tolist()) for row in actual_topk]
    expected_sets = [set(row.tolist()) for row in expected_topk]
    report[f"top{k}_set_exact"] = float(
        np.mean([a == e for a, e in zip(actual_sets, expected_sets, strict=True)])
    )
    expected_argmax = np.argmax(expected_flat, axis=-1)
    report[f"expected_argmax_in_actual_top{k}"] = float(
        np.mean([token in actual_sets[i] for i, token in enumerate(expected_argmax)])
    )
  return report


def _load_tokens(path: Path, frames: int) -> np.ndarray:
  """Load fixed unique-code fixture tokens shaped ``[frames, 12]``."""
  if not path.exists():
    raise FileNotFoundError(
        f"Fixture tokens not found: {path}. "
        "Run scripts/generate_mrt2_coreml_reference_fixtures.py first."
    )
  tokens = np.load(path).astype(np.int32)
  if tokens.ndim != 2 or tokens.shape[1] != MRT2_RVQ_LEVELS:
    raise ValueError(f"Expected token fixture shape [N, 12], got {tokens.shape}")
  return tokens[:frames]


def _load_reference_npz(
    path: Path,
    frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Load precomputed MLX reference tensors so validation runs without MLX.

  The ``.npz`` is produced by dumping ``_temporal_mlx_fixture`` outputs (see
  ``fixtures/`` in the Hugging Face repo). Arrays: ``temporal_inputs``
  ``[1, N, 1024]``, ``source_encoded`` ``[1, N, 256]``, ``temporal_outputs_mlx``
  ``[1, N, 1024]``, ``prefix_embeddings`` ``[N, 11, 1024]``,
  ``depth_logits_mlx`` ``[N, 12, 12294]``.
  """
  data = np.load(path)
  if data["temporal_inputs"].shape[1] < frames:
    raise ValueError(
        f"Reference npz {path} holds {data['temporal_inputs'].shape[1]} frames; "
        f"--frames {frames} requested."
    )
  return (
      data["temporal_inputs"][:, :frames].astype(np.float32),
      data["source_encoded"][:, :frames].astype(np.float32),
      data["temporal_outputs_mlx"][:, :frames].astype(np.float32),
      data["prefix_embeddings"][:frames].astype(np.float32),
      data["depth_logits_mlx"][:frames].astype(np.float32),
  )


def _temporal_mlx_fixture(
    *,
    tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Return teacher-forced temporal/depth fixtures from MLX."""
  import magenta_rt  # noqa: F401
  import mlx.core as mx
  import sequence_layers.mlx as sl

  from mrt2_coreml import paths
  from magenta_rt.mlx import load_weights as mlx_load_weights
  from magenta_rt.mlx import model
  from magenta_rt.mlx import spectrostream
  from magenta_rt.mlx import system

  musiccoca = [660, 1016, 295, 206, 857, 841, 391, 857, 619, 70, 401, 22]
  notes = [0] * 127 + [1]
  drums = [-1]
  cfg_tokens = [4, 4, 4]
  conditioning = (
      np.concatenate([musiccoca, notes, drums, cfg_tokens], axis=0).astype(np.int32)
      + 7
  )
  block = sl.Sequence(
      mx.array(conditioning.reshape(1, 1, -1), dtype=mx.int32),
      mx.array([[True]], dtype=mx.bool_),
  )
  constants = {
      "temperature": mx.array([1.3]),
      "top_k": mx.array([40]),
  }

  exp = model.get_model_class("mrt2_small")()
  exp.compute_dtype = mx.float32
  sampler_config = system.MagentaRT2Sampler.Config(
      depthformer=exp.depthformer_config(),
      spectrostream=spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
          rvq_truncation_level=exp.spectrostream.rvq_truncation_level,
          use_unique_codes=False,
      ),
  )
  sampler = sampler_config.make()
  original_convert = mlx_load_weights.convert_to_bf16
  mlx_load_weights.convert_to_bf16 = lambda module: None
  try:
    mlx_load_weights.load_weights(
        sampler,
        paths.checkpoints_dir() / "mrt2_small.safetensors",
        num_input_channels=exp.input_num_channels,
    )
  finally:
    mlx_load_weights.convert_to_bf16 = original_convert

  depth_sampler = sampler.layers[0]
  decoder = depth_sampler.decoder
  state = depth_sampler.get_initial_state(
      1,
      block.channel_spec,
      constants=constants,
      training=False,
  )

  temporal_inputs = []
  sources = []
  temporal_expected = []
  prefix_embeddings = []
  depth_logits = []
  for frame_tokens in tokens:
    encoder_state, sampler_previous_output, sampler_state, sampler_delay = state
    encoded, encoder_state = depth_sampler.encoder.body.step(
        block,
        encoder_state,
        training=False,
        constants=constants,
    )
    rng, previous_frame, temporal_state, step = sampler_state
    embedded_frame = decoder.embedder.layer(previous_frame)
    temporal_input = embedded_frame.apply_values(
        lambda v: mx.mean(v.astype(mx.float32), axis=-2)
    )
    temporal_output, temporal_state = decoder.temporal_body.step(
        temporal_input,
        temporal_state,
        training=False,
        constants=constants | {depth_sampler.conditioning_name: encoded},
    )
    mx.eval(encoded.values, temporal_input.values, temporal_output.values)
    temporal_inputs.append(np.asarray(temporal_input.values, dtype=np.float32))
    sources.append(np.asarray(encoded.values, dtype=np.float32))
    temporal_expected.append(np.asarray(temporal_output.values, dtype=np.float32))

    forced = frame_tokens.reshape(1, 1, MRT2_RVQ_LEVELS).astype(np.int32)
    prefix_tokens = forced[:, :, :-1]
    prefix_sequence = sl.Sequence(
        mx.array(prefix_tokens, dtype=mx.int32),
        mx.ones(prefix_tokens.shape[:2], dtype=mx.bool_),
    )
    prefix_embedding = decoder.embedder.layer(prefix_sequence).values.reshape(
        1,
        MRT2_RVQ_LEVELS - 1,
        -1,
    )
    depth_input = mx.concatenate(
        [temporal_output.values, prefix_embedding],
        axis=1,
    )
    logits = decoder.depth_body.layer(
        sl.Sequence(depth_input, mx.ones((1, MRT2_RVQ_LEVELS), dtype=mx.bool_))
    ).values
    logits = mx.tanh(logits / 30.0) * 30.0
    mx.eval(prefix_embedding, logits)
    prefix_embeddings.append(np.asarray(prefix_embedding, dtype=np.float32)[0])
    depth_logits.append(np.asarray(logits, dtype=np.float32)[0])

    depth_samples = sl.Sequence(
        mx.array(forced, dtype=mx.int32),
        mx.ones(forced.shape[:2], dtype=mx.bool_),
    )
    sampler_state = (rng, depth_samples, temporal_state, step + 1)
    state = (encoder_state, depth_samples, sampler_state, sampler_delay)

  return (
      np.concatenate(temporal_inputs, axis=1),
      np.concatenate(sources, axis=1),
      np.concatenate(temporal_expected, axis=1),
      np.stack(prefix_embeddings, axis=0),
      np.stack(depth_logits, axis=0),
  )


def _sample_depth_logits(logits: np.ndarray) -> np.ndarray:
  """Sample deterministic unique RVQ tokens from ``[frames, 12, vocab]`` logits."""
  level_major = np.transpose(logits, (1, 0, 2))[:, np.newaxis, :, :]
  return sample_rvq_frame_logits(
      level_major,
      SamplingConfig(temperature=0.0, top_k=None, seed=0),
  )


def _depth_body_report(
    *,
    model_path: Path,
    label: str,
    temporal_outputs: np.ndarray,
    prefix_embeddings: np.ndarray,
    mlx_depth_logits: np.ndarray,
) -> dict[str, Any]:
  """Run depth-body Core ML logits from temporal outputs plus prefix embeddings."""
  if not model_path.exists():
    return {
        "available": False,
        "label": label,
        "reason": f"Depth-body Core ML package not found: {model_path}",
    }
  depth_model = ct.models.MLModel(
      str(model_path),
      compute_units=ct.ComputeUnit.CPU_ONLY,
  )
  coreml_logits = []
  frame_seconds = []
  for frame_index in range(temporal_outputs.shape[1]):
    depth_input = np.concatenate(
        [
            temporal_outputs[:, frame_index : frame_index + 1],
            prefix_embeddings[frame_index : frame_index + 1],
        ],
        axis=1,
    ).astype(np.float32)
    start = time.perf_counter()
    logits = np.asarray(
        depth_model.predict({DEPTH_BODY_INPUT_NAME: depth_input})[DEPTH_BODY_OUTPUT_NAME],
        dtype=np.float32,
    )
    frame_seconds.append(time.perf_counter() - start)
    coreml_logits.append(logits[0])
  coreml_logits_np = np.stack(coreml_logits, axis=0)
  coreml_samples = _sample_depth_logits(coreml_logits_np)
  mlx_samples = _sample_depth_logits(mlx_depth_logits)
  sample_mismatches = coreml_samples != mlx_samples
  return {
      "available": True,
      "label": label,
      "model_path": str(model_path),
      "boundary": "unrolled_temporal_coreml_outputs_to_depth_body_logits",
      "coreml_vs_mlx": _metrics(coreml_logits_np, mlx_depth_logits),
      "topk_agreement_coreml_vs_mlx": _topk_report(coreml_logits_np, mlx_depth_logits, (5, 40)),
      "deterministic_argmax_sampling": {
          "sampled_unique_shape": list(coreml_samples.shape),
          "mismatch_count_vs_mlx": int(np.sum(sample_mismatches)),
          "total_tokens": int(sample_mismatches.size),
          "mismatch_indices": np.argwhere(sample_mismatches).tolist(),
      },
      "timing_smoke": {
          "scope": "Python Core ML CPU_ONLY depth-body predict per frame, not device timing",
          "p50_ms": float(np.percentile(frame_seconds, 50) * 1000.0),
          "p99_ms": float(np.percentile(frame_seconds, 99) * 1000.0),
      },
  }


def validate(args: argparse.Namespace) -> dict[str, Any]:
  """Run unrolled validation and return a machine-readable report."""
  _ensure_coreml_runtime_path()
  model_path = Path(args.model_template.format(frames=args.frames))
  metadata_path = Path(args.metadata_template.format(frames=args.frames))
  if not model_path.exists():
    raise FileNotFoundError(
        f"Core ML package not found: {model_path}. "
        "Run scripts/convert_mrt2_temporal_body_unrolled_coreml.py first."
    )
  reference_npz = Path(args.reference_npz.format(frames=args.frames))
  if reference_npz.exists():
    reference_source = f"precomputed npz: {reference_npz.name}"
    (
        temporal_inputs,
        source_encoded,
        mlx_output,
        prefix_embeddings,
        mlx_depth_logits,
    ) = _load_reference_npz(reference_npz, args.frames)
  else:
    reference_source = "live MLX teacher-forced reference"
    tokens = _load_tokens(Path(args.tokens_path), args.frames)
    (
        temporal_inputs,
        source_encoded,
        mlx_output,
        prefix_embeddings,
        mlx_depth_logits,
    ) = _temporal_mlx_fixture(tokens=tokens)

  pytorch_output = None
  if not args.skip_pytorch:
    pytorch_model = TemporalBodyCoreMLUnrolledWrapper(frame_count=args.frames).eval()
    with torch.no_grad():
      pytorch_output = (
          pytorch_model(
              torch.from_numpy(temporal_inputs),
              torch.from_numpy(source_encoded),
          )
          .detach()
          .cpu()
          .numpy()
          .astype(np.float32)
      )

  coreml_model = ct.models.MLModel(
      str(model_path),
      compute_units=ct.ComputeUnit.CPU_ONLY,
  )
  start = time.perf_counter()
  coreml_output = np.asarray(
      coreml_model.predict(
          {
              TEMPORAL_INPUT_NAME: temporal_inputs,
              SOURCE_INPUT_NAME: source_encoded,
          },
          state=coreml_model.make_state(),
      )[OUTPUT_NAME],
      dtype=np.float32,
  )
  predict_ms = (time.perf_counter() - start) * 1000.0

  export_metadata: dict[str, Any] | None = None
  if metadata_path.exists():
    export_metadata = json.loads(metadata_path.read_text())

  per_frame = []
  for frame_index in range(args.frames):
    frame_slice = slice(frame_index, frame_index + 1)
    per_frame.append(
        {
            "frame": frame_index,
            "pytorch_vs_mlx": None if pytorch_output is None else _metrics(
                pytorch_output[:, frame_slice],
                mlx_output[:, frame_slice],
            ),
            "coreml_vs_pytorch": None if pytorch_output is None else _metrics(
                coreml_output[:, frame_slice],
                pytorch_output[:, frame_slice],
            ),
            "coreml_vs_mlx": _metrics(
                coreml_output[:, frame_slice],
                mlx_output[:, frame_slice],
            ),
        }
    )
  zero_cache_control = None
  if args.frames > 1 and pytorch_output is not None:
    with torch.no_grad():
      zero_cache_frame = (
          TemporalBodyCoreMLSlotWrapper(slot_index=1).eval()(
              torch.from_numpy(temporal_inputs[:, 1:2]),
              torch.from_numpy(source_encoded[:, 1:2]),
          )
          .detach()
          .cpu()
          .numpy()
          .astype(np.float32)
      )
    zero_cache_control = {
        "frame": 1,
        "purpose": (
            "Negative control: run frame 1 through slot 1 with an empty cache. "
            "This should be worse than the unrolled frame-1 result if previous "
            "slot state matters."
        ),
        "pytorch_slot1_zero_cache_vs_mlx": _metrics(
            zero_cache_frame,
            mlx_output[:, 1:2],
        ),
        "pytorch_unrolled_frame1_vs_mlx": per_frame[1]["pytorch_vs_mlx"],
        "coreml_unrolled_frame1_vs_mlx": per_frame[1]["coreml_vs_mlx"],
    }

  known_limits = [
      "Unrolls a fixed no-wrap frame count into one prediction.",
      "Frame 1 proves read-after-write across slots inside this unrolled graph.",
      "This is not the final one-prediction-per-40-ms-frame runtime API.",
      "The conditioning encoder remains host-owned through source_encoded.",
      "Depth-body logits remain in a separate Core ML package.",
  ]
  if args.frames < 25:
    known_limits.append("This is not the 25-frame full temporal-plus-depth logits loop.")
  else:
    known_limits.append(
        "This is a 25-frame validation path, but still not a runtime-compatible per-frame API."
    )

  return {
      "schema": "mrt2-temporal-body-unrolled-coreml-validation-v1",
      "boundary": "temporal_body_unrolled_no_wrap",
      "frames": args.frames,
      "model_path": str(model_path),
      "metadata_path": str(metadata_path) if metadata_path.exists() else None,
      "state_count": len(TemporalBodyCoreMLUnrolledWrapper.state_names()),
      "reference_source": reference_source,
      "pytorch_vs_mlx": None if pytorch_output is None else _metrics(pytorch_output, mlx_output),
      "coreml_vs_pytorch": None if pytorch_output is None else _metrics(coreml_output, pytorch_output),
      "coreml_vs_mlx": _metrics(coreml_output, mlx_output),
      "per_frame": per_frame,
      "zero_cache_control": zero_cache_control,
      "temporal_plus_depth_logits": {
          "published": _depth_body_report(
              model_path=Path(args.depth_body_model_path),
              label="published depth body (FLOAT32)",
              temporal_outputs=coreml_output,
              prefix_embeddings=prefix_embeddings,
              mlx_depth_logits=mlx_depth_logits,
          ),
          "fp32_control": _depth_body_report(
              model_path=Path(args.depth_body_fp32_model_path),
              label="FLOAT32 depth-body control (optional, not published)",
              temporal_outputs=coreml_output,
              prefix_embeddings=prefix_embeddings,
              mlx_depth_logits=mlx_depth_logits,
          ),
      },
      "timing_smoke": {
          "scope": "Python Core ML CPU_ONLY single unrolled predict, not device timing",
          "predict_ms": float(predict_ms),
      },
      "export_compile": None if export_metadata is None else export_metadata.get("compile"),
      "known_limits": known_limits,
  }


def _write_summary(report: dict[str, Any], summary_path: Path) -> None:
  """Write a short markdown summary beside the JSON report."""
  cml_mlx = report["coreml_vs_mlx"]
  lines = [
      "# MRT2 Temporal Body Unrolled Core ML Validation",
      "",
      f"- Boundary: `{report['boundary']}`",
      f"- Frames: {report['frames']}",
      f"- State count: {report['state_count']}",
      f"- Core ML vs MLX max error: {cml_mlx['max_abs_error']:.10f}",
      f"- Core ML vs MLX mean error: {cml_mlx['mean_abs_error']:.10f}",
      f"- Core ML vs MLX correlation: {cml_mlx['correlation']:.12f}",
      f"- Core ML CPU_ONLY predict smoke: {report['timing_smoke']['predict_ms']:.3f} ms",
      "",
      "Per-frame Core ML vs MLX:",
  ]
  for frame in report["per_frame"]:
    metrics = frame["coreml_vs_mlx"]
    lines.append(
        f"- Frame {frame['frame']}: max {metrics['max_abs_error']:.10f}, "
        f"mean {metrics['mean_abs_error']:.10f}, "
        f"corr {metrics['correlation']:.12f}"
    )
  if report["zero_cache_control"] is not None:
    control = report["zero_cache_control"]
    metrics = control["pytorch_slot1_zero_cache_vs_mlx"]
    lines.extend(
        [
            "",
            "Frame-1 zero-cache control:",
            f"- PyTorch slot-1 zero-cache vs MLX max: {metrics['max_abs_error']:.10f}",
            f"- PyTorch slot-1 zero-cache vs MLX mean: {metrics['mean_abs_error']:.10f}",
            f"- PyTorch slot-1 zero-cache vs MLX corr: {metrics['correlation']:.12f}",
        ]
    )
  for key, label in (("published", "published (FLOAT32)"), ("fp32_control", "FLOAT32 control")):
    depth_report = report["temporal_plus_depth_logits"][key]
    lines.extend(["", f"Temporal Core ML + depth-body {label}:"])
    if not depth_report["available"]:
      lines.append(f"- Unavailable: {depth_report['reason']}")
      continue
    metrics = depth_report["coreml_vs_mlx"]
    samples = depth_report["deterministic_argmax_sampling"]
    lines.extend(
        [
            f"- Core ML vs MLX max: {metrics['max_abs_error']:.10f}",
            f"- Core ML vs MLX mean: {metrics['mean_abs_error']:.10f}",
            f"- Core ML vs MLX corr: {metrics['correlation']:.12f}",
            f"- Deterministic sample mismatches: "
            f"{samples['mismatch_count_vs_mlx']} / {samples['total_tokens']}",
        ]
    )
  lines.extend(["", "Known limits:"])
  lines.extend(f"- {limit}" for limit in report["known_limits"])
  lines.append("")
  summary_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
  """Parse CLI flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  # The published MRT2TemporalBody.mlpackage is the 1-frame stateful export, so
  # the default validates one frame. Re-exported multi-frame unrolled variants
  # can pass --frames N with a matching --model-template.
  parser.add_argument("--frames", type=int, default=1)
  parser.add_argument("--model-template", default=str(DEFAULT_MODEL_TEMPLATE))
  parser.add_argument("--metadata-template", default=str(DEFAULT_METADATA_TEMPLATE))
  parser.add_argument("--tokens-path", default=str(DEFAULT_TOKENS_PATH))
  parser.add_argument("--depth-body-model-path", default=str(DEFAULT_DEPTH_BODY_MODEL_PATH))
  parser.add_argument(
      "--depth-body-fp32-model-path",
      default=str(DEFAULT_DEPTH_BODY_FP32_MODEL_PATH),
  )
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--report-template", default=DEFAULT_REPORT_TEMPLATE)
  parser.add_argument("--summary-template", default=DEFAULT_SUMMARY_TEMPLATE)
  parser.add_argument(
      "--reference-npz",
      default=str(DEFAULT_REFERENCE_NPZ_TEMPLATE),
      help=(
          "Precomputed MLX reference tensors (.npz). When the file exists, the "
          "MLX stack is not needed; otherwise the reference is computed live "
          "(requires the magenta-realtime MLX backend and the mrt2_small "
          "checkpoint)."
      ),
  )
  parser.add_argument(
      "--skip-pytorch",
      action="store_true",
      help=(
          "Skip the PyTorch wrapper leg (which needs the mrt2_small.safetensors "
          "checkpoint). Core ML vs MLX-reference parity still runs."
      ),
  )
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
