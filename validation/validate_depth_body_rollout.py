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

"""Validate the MRT2 in-graph depth rollout Core ML model.

Gates ``mrt2_depth_body_rollout`` (exported by
scripts/convert_mrt2_depth_body_rollout_coreml.py) against the proven
FLOAT32 reference chain before it may replace the 12-call rollout:

1. ``argmax_frames``: zero Gumbel noise degenerates the in-graph sampler to
   pure argmax. Sequential AUTOREGRESSIVE frames (each frame's temporal input
   is derived from sampled embeddings, so divergence compounds) must match an
   out-of-graph reference rollout that drives the torch FLOAT32 full-pass
   wrapper with the host sampling algorithm. FLOAT32 export gate: zero token
   mismatches. FLOAT16 export: mismatches are reported (fp16 near-tie flips)
   and gated by ``--max-fp16-argmax-mismatch-rate``.
2. ``noise_frames``: real Gumbel noise + temperature. The Core ML graph must
   reproduce the reference sampler token for token when both consume the
   SAME noise (FLOAT32: exact; FLOAT16: same rate gate). This proves top-k
   masking, temperature scaling, noise addition, and embedding feedback.
3. ``feedback_parity``: the ``temporal_feedback`` output must equal the mean
   of the embedder rows of the sampled tokens (the next frame's
   ``temporal_inputs`` contract).

Frame inputs come from deterministic seeds, mirroring the
``random_frames_no_reset`` protocol in
scripts/validate_mrt2_depth_body_step_coreml.py. Timing numbers are CPU
predict smoke only — device latency is gated separately on the phone.
"""

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

from mrt2_coreml.depth_body_wrapper import (
    DepthBodyLogitsWrapper,
    DepthBodyRolloutWrapper,
    deterministic_depth_body_input,
    gumbel_topk_sample_reference,
)
from mrt2_coreml.depthformer_wrapper import (
    MRT2_CODEBOOK_SIZE,
    MRT2_MODEL_DIM,
    MRT2_RESERVED_TOKENS,
    MRT2_RVQ_LEVELS,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "build" / "models" / "mrt2_depth_body_rollout.mlpackage"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "validation"
DEFAULT_REPORT_NAME = "mrt2_depth_body_rollout_validation.json"
DEFAULT_SUMMARY_NAME = "mrt2_depth_body_rollout_validation.md"
FRAME_INPUT_NAME = "temporal_frame"
NOISE_INPUT_NAME = "gumbel_noise"
INVERSE_TEMPERATURE_INPUT_NAME = "inverse_temperature"
CODES_OUTPUT_NAME = "sampled_codes"
FEEDBACK_OUTPUT_NAME = "temporal_feedback"


def _ensure_coreml_runtime_path() -> None:
  """Give coremltools access to macOS helper tools when Codex PATH is thin."""
  path_parts = os.environ.get("PATH", "").split(os.pathsep)
  for required in ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]:
    if required not in path_parts:
      path_parts.append(required)
  os.environ["PATH"] = os.pathsep.join(path_parts)


def _gumbel_noise(seed: int | None) -> np.ndarray:
  """Per-level Gumbel(0,1) noise, or zeros for the argmax degenerate case."""
  if seed is None:
    return np.zeros((MRT2_RVQ_LEVELS, MRT2_CODEBOOK_SIZE), dtype=np.float32)
  rng = np.random.default_rng(seed)
  uniform = np.clip(
      rng.random((MRT2_RVQ_LEVELS, MRT2_CODEBOOK_SIZE), dtype=np.float32),
      1e-7,
      None,
  )
  return (-np.log(-np.log(uniform))).astype(np.float32)


def _reference_rollout(
    rollout: DepthBodyRolloutWrapper,
    full: DepthBodyLogitsWrapper,
    temporal_frame: np.ndarray,
    gumbel_noise: np.ndarray,
    inverse_temperature: float,
) -> tuple[list[int], np.ndarray]:
  """Out-of-graph FLOAT32 reference: 12 full passes + host sampling math."""
  depth_inputs = torch.zeros((1, MRT2_RVQ_LEVELS, MRT2_MODEL_DIM))
  depth_inputs[:, 0] = torch.from_numpy(temporal_frame).reshape(1, MRT2_MODEL_DIM)
  noise = torch.from_numpy(gumbel_noise)
  codes: list[int] = []
  embeddings: list[torch.Tensor] = []
  with torch.no_grad():
    for level in range(MRT2_RVQ_LEVELS):
      logits = full(depth_inputs)[0, level]
      start = MRT2_RESERVED_TOKENS + level * MRT2_CODEBOOK_SIZE
      code = gumbel_topk_sample_reference(
          logits[start : start + MRT2_CODEBOOK_SIZE],
          noise[level],
          inverse_temperature,
      )
      codes.append(code)
      embedding = getattr(rollout, rollout._embed_table_name(level))[code]
      embeddings.append(embedding)
      if level < MRT2_RVQ_LEVELS - 1:
        depth_inputs[:, level + 1] = embedding
  feedback = torch.mean(torch.stack(embeddings), dim=0, keepdim=True)
  return codes, feedback.numpy()


def _frames_report(
    model: ct.models.MLModel,
    rollout: DepthBodyRolloutWrapper,
    full: DepthBodyLogitsWrapper,
    frames: int,
    noise_seed_base: int | None,
    temperature: float,
) -> dict[str, Any]:
  """Autoregressive frame sequence: Core ML rollout vs the reference chain.

  Frame 0's temporal input is a deterministic pseudo-random vector; every
  later frame's temporal input is the REFERENCE chain's temporal_feedback,
  fed to both arms, so each frame isolates one prediction while the sequence
  still exercises feedback-driven inputs (real embedding rows, not noise).
  """
  inverse_temperature = 1.0 / max(0.05, temperature)
  temporal_frame = (
      deterministic_depth_body_input(seed=2026).numpy()[:, :1].astype(np.float32)
  )
  mismatches = 0
  total = 0
  feedback_max_error = 0.0
  per_frame_mismatches: list[int] = []
  # Within a frame the rollout is autoregressive, so ONE near-tie flip
  # cascades through every later level; the first divergent level is the
  # honest per-frame flip diagnostic, the raw mismatch count is not.
  first_divergence_levels: list[int | None] = []
  seconds: list[float] = []
  for frame_index in range(frames):
    noise_seed = None if noise_seed_base is None else noise_seed_base + frame_index
    noise = _gumbel_noise(noise_seed)
    reference_codes, reference_feedback = _reference_rollout(
        rollout, full, temporal_frame, noise, inverse_temperature
    )
    start = time.perf_counter()
    result = model.predict(
        {
            FRAME_INPUT_NAME: temporal_frame,
            NOISE_INPUT_NAME: noise,
            INVERSE_TEMPERATURE_INPUT_NAME: np.asarray(
                [inverse_temperature], dtype=np.float32
            ),
        }
    )
    seconds.append(time.perf_counter() - start)
    coreml_codes = np.asarray(result[CODES_OUTPUT_NAME], dtype=np.int64).reshape(-1)
    coreml_feedback = np.asarray(result[FEEDBACK_OUTPUT_NAME], dtype=np.float32)
    diverged = coreml_codes != np.asarray(reference_codes)
    frame_mismatches = int(np.sum(diverged))
    mismatches += frame_mismatches
    per_frame_mismatches.append(frame_mismatches)
    first_divergence_levels.append(
        int(np.argmax(diverged)) if frame_mismatches else None
    )
    total += MRT2_RVQ_LEVELS
    feedback_max_error = max(
        feedback_max_error,
        float(np.max(np.abs(coreml_feedback - reference_feedback))),
    )
    # Both arms continue from the REFERENCE feedback so later frames stay
    # comparable even if an earlier frame diverged on an fp16 near-tie.
    temporal_frame = reference_feedback.reshape(1, 1, MRT2_MODEL_DIM)
  return {
      "frames": frames,
      "temperature": temperature,
      "noise": "zero (argmax degenerate)" if noise_seed_base is None else "gumbel",
      "token_mismatches_vs_reference": mismatches,
      "total_tokens": total,
      "mismatch_rate": mismatches / total if total else 0.0,
      "per_frame_mismatches": per_frame_mismatches,
      "frames_with_any_divergence": sum(
          1 for level in first_divergence_levels if level is not None
      ),
      "first_divergence_levels": first_divergence_levels,
      "temporal_feedback_max_abs_error": feedback_max_error,
      "rollout_p50_ms": float(np.percentile(seconds, 50) * 1000.0),
      "timing_scope": "Python Core ML predict smoke, not device timing",
  }


def validate(args: argparse.Namespace) -> dict[str, Any]:
  """Run validation and return the report dictionary."""
  _ensure_coreml_runtime_path()
  model_path = Path(args.model_path)
  if not model_path.exists():
    raise FileNotFoundError(
        f"Rollout model not found: {model_path}. "
        "Run scripts/convert_mrt2_depth_body_rollout_coreml.py first."
    )
  model = ct.models.MLModel(
      str(model_path), compute_units=getattr(ct.ComputeUnit, args.compute_units)
  )
  precision = "FLOAT32" if args.float32 else "FLOAT16"
  rollout = DepthBodyRolloutWrapper().eval()
  full = DepthBodyLogitsWrapper().eval()

  report: dict[str, Any] = {
      "schema": "mrt2-depth-body-rollout-coreml-validation-v1",
      "model_path": str(model_path),
      "compute_units": args.compute_units,
      "declared_precision": precision,
      "argmax_frames": _frames_report(
          model, rollout, full, args.frames, noise_seed_base=None, temperature=1.0
      ),
      "noise_frames_t10": _frames_report(
          model, rollout, full, args.frames, noise_seed_base=5000, temperature=1.0
      ),
      "noise_frames_t13": _frames_report(
          model, rollout, full, args.frames, noise_seed_base=9000, temperature=1.3
      ),
  }
  arms = ["argmax_frames", "noise_frames_t10", "noise_frames_t13"]
  if args.float32:
    token_pass = all(
        report[arm]["token_mismatches_vs_reference"] == 0 for arm in arms
    )
    criteria = "FLOAT32: zero token mismatches vs reference on all arms"
  else:
    token_pass = all(
        report[arm]["mismatch_rate"] <= args.max_fp16_mismatch_rate for arm in arms
    )
    criteria = (
        "FLOAT16: token mismatch rate <= "
        f"{args.max_fp16_mismatch_rate} per arm (fp16 near-tie flips sample "
        "an equally-likely top-k neighbor; distribution unchanged)"
    )
  feedback_pass = all(
      report[arm]["temporal_feedback_max_abs_error"] <= args.max_feedback_error
      for arm in arms
  )
  report["gate"] = {
      "tokens": bool(token_pass),
      "temporal_feedback": bool(feedback_pass),
      "passed": bool(token_pass and feedback_pass),
      "criteria": criteria
      + f"; temporal_feedback max |err| <= {args.max_feedback_error}",
  }
  return report


def _write_summary(report: dict[str, Any], summary_path: Path) -> None:
  """Write a short markdown summary beside the JSON report."""
  lines = [
      "# MRT2 Depth-Body In-Graph Rollout Core ML Validation",
      "",
      f"- Model: {report['model_path']} ({report['declared_precision']}, "
      f"{report['compute_units']})",
      f"- Gate: {'PASS' if report['gate']['passed'] else 'FAIL'} — "
      f"{report['gate']['criteria']}",
  ]
  for arm in ("argmax_frames", "noise_frames_t10", "noise_frames_t13"):
    data = report[arm]
    lines.append(
        f"- {arm}: {data['token_mismatches_vs_reference']} / "
        f"{data['total_tokens']} token mismatches "
        f"(rate {data['mismatch_rate']:.4f}), feedback max |err| "
        f"{data['temporal_feedback_max_abs_error']:.6f}, p50 "
        f"{data['rollout_p50_ms']:.2f} ms (CPU smoke)"
    )
  lines.append("")
  summary_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
  """Parse CLI flags."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
  parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
  parser.add_argument("--frames", type=int, default=25)
  parser.add_argument(
      "--compute-units",
      choices=("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"),
      default="CPU_ONLY",
  )
  parser.add_argument(
      "--float32",
      action="store_true",
      help="Gate as a FLOAT32 export: zero token mismatches required.",
  )
  parser.add_argument("--max-fp16-mismatch-rate", type=float, default=0.02)
  parser.add_argument("--max-feedback-error", type=float, default=0.05)
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  args = parse_args()
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  report = validate(args)
  report_path = output_dir / args.report_name
  summary_path = output_dir / args.summary_name
  report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  _write_summary(report, summary_path)
  print(f"Wrote {report_path}")
  print(f"Wrote {summary_path}")
  print(f"Gate: {'PASS' if report['gate']['passed'] else 'FAIL'}")
  if not report["gate"]["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
