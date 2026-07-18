#!/usr/bin/env python3
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

"""Assemble the predeclared MRT2 weight-compression falsification ladder."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "build" / "models"
RESULT_ROOT = REPO_ROOT / "validation" / "results"
OUTPUT_JSON = RESULT_ROOT / "MRT2WeightCompressionLadder.json"
OUTPUT_MD = RESULT_ROOT / "MRT2WeightCompressionLadder.md"
VARIANTS = ("int8-linear", "palettize-6bit", "palettize-4bit")


def _read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def _artifact_bytes(path: Path) -> int:
  return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _weight_sha256(path: Path) -> str:
  weights = sorted(path.rglob("weight.bin"))
  if len(weights) != 1:
    raise ValueError(f"expected one weight.bin under {path}, found {len(weights)}")
  digest = hashlib.sha256()
  with weights[0].open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _relative(path: Path) -> str:
  return str(path.relative_to(REPO_ROOT))


def _temporal_entry(variant: str, baseline_bytes: int) -> dict[str, Any]:
  package = MODEL_ROOT / f"mrt2_temporal_body_streaming_carry_01_{variant}.mlpackage"
  receipt_path = (
    RESULT_ROOT / f"MRT2TemporalBodyStreamingCarry_{variant}_validation.json"
  )
  receipt = _read_json(receipt_path)
  metrics = receipt["coreml_vs_reference"]
  finite_passed = metrics["finite_ratio"] == 1.0
  parity_passed = (
    finite_passed
    and metrics["correlation"] >= 0.999
    and metrics["max_abs_error"] <= 2.5
  )
  artifact_bytes = _artifact_bytes(package)
  return {
    "component": "temporal",
    "variant": variant,
    "artifact": _relative(package),
    "artifact_bytes": artifact_bytes,
    "fraction_of_uncompressed": artifact_bytes / baseline_bytes,
    "weight_sha256": _weight_sha256(package),
    "receipt": _relative(receipt_path),
    "finite_gate": {"passed": finite_passed, "finite_ratio": metrics["finite_ratio"]},
    "parity_gate": {
      "passed": parity_passed,
      "criteria": "64 steps, 23 post-wrap; correlation >= 0.999 and max_abs_error <= 2.5",
      "correlation": metrics["correlation"],
      "max_abs_error": metrics["max_abs_error"],
      "mean_abs_error": metrics["mean_abs_error"],
    },
    "device_latency_gate": {
      "status": "not_run_due_to_parity_gate" if not parity_passed else "required"
    },
    "blind_audio_gate": {
      "status": "not_run_due_to_parity_gate" if not parity_passed else "required"
    },
    "disposition": "rejected_at_parity_gate" if not parity_passed else "advance",
  }


def _depth_entry(variant: str, baseline_bytes: int) -> dict[str, Any]:
  package = MODEL_ROOT / f"mrt2_depth_body_rollout_{variant}.mlpackage"
  receipt_path = RESULT_ROOT / f"MRT2DepthBodyRollout_{variant}_validation.json"
  receipt = _read_json(receipt_path)
  arms = ("argmax_frames", "noise_frames_t10", "noise_frames_t13")
  finite_passed = all(
    math.isfinite(receipt[arm]["temporal_feedback_max_abs_error"]) for arm in arms
  )
  parity_passed = finite_passed and receipt["gate"]["passed"]
  artifact_bytes = _artifact_bytes(package)
  return {
    "component": "depth",
    "variant": variant,
    "artifact": _relative(package),
    "artifact_bytes": artifact_bytes,
    "fraction_of_uncompressed": artifact_bytes / baseline_bytes,
    "weight_sha256": _weight_sha256(package),
    "receipt": _relative(receipt_path),
    "finite_gate": {"passed": finite_passed},
    "parity_gate": {
      "passed": parity_passed,
      "criteria": receipt["gate"]["criteria"],
      "arms": {
        arm: {
          "mismatch_rate": receipt[arm]["mismatch_rate"],
          "temporal_feedback_max_abs_error": receipt[arm][
            "temporal_feedback_max_abs_error"
          ],
        }
        for arm in arms
      },
    },
    "device_latency_gate": {
      "status": "not_run_due_to_parity_gate" if not parity_passed else "required"
    },
    "blind_audio_gate": {
      "status": "not_run_due_to_parity_gate" if not parity_passed else "required"
    },
    "disposition": "rejected_at_parity_gate" if not parity_passed else "advance",
  }


def _markdown(report: dict[str, Any]) -> str:
  lines = [
    "# MRT2 Weight-Compression Ladder",
    "",
    "The ladder follows the predeclared early-stop order: finite output, "
    "deterministic reference parity, device measurement, then blind audio. "
    "A failed parity arm was not installed on either phone and was not "
    "rendered for listening.",
    "",
    "| Component | Variant | Size (MiB) | Baseline | Key parity result | Disposition |",
    "| --- | --- | ---: | ---: | --- | --- |",
  ]
  for entry in report["candidates"]:
    if entry["component"] == "temporal":
      parity = entry["parity_gate"]
      result = (
        f"corr {parity['correlation']:.6f}; max |err| {parity['max_abs_error']:.3f}"
      )
    else:
      arms = entry["parity_gate"]["arms"]
      result = "mismatch " + "/".join(
        f"{arms[name]['mismatch_rate']:.3f}"
        for name in ("argmax_frames", "noise_frames_t10", "noise_frames_t13")
      )
    lines.append(
      f"| {entry['component']} | {entry['variant']} | "
      f"{entry['artifact_bytes'] / 2**20:.1f} | "
      f"{entry['fraction_of_uncompressed']:.3f}x | {result} | "
      f"{entry['disposition']} |"
    )
  lines.extend(
    [
      "",
      "## Decision",
      "",
      report["decision"]["summary"],
      "",
      "The uncompressed temporal graph remains the selected system artifact. "
      "The existing FP16 depth graph remains selected under its previously "
      "documented distributional and device-listening acceptance boundary. "
      "This experiment does not support a compression-causes-speedup claim: "
      "no compressed candidate passed the prerequisite parity gate, so device "
      "latency, placement, DRAM estimates, and listening were deliberately not "
      "measured for those candidates.",
      "",
    ]
  )
  return "\n".join(lines)


def main() -> None:
  temporal_baseline = MODEL_ROOT / "mrt2_temporal_body_streaming_carry_01.mlpackage"
  depth_baseline = MODEL_ROOT / "mrt2_depth_body_rollout_f16.mlpackage"
  temporal_bytes = _artifact_bytes(temporal_baseline)
  depth_bytes = _artifact_bytes(depth_baseline)
  candidates = [
    *(_temporal_entry(variant, temporal_bytes) for variant in VARIANTS),
    *(_depth_entry(variant, depth_bytes) for variant in VARIANTS),
  ]
  report = {
    "schema": "mrt2-weight-compression-ladder-v1",
    "gate_order": ["finite", "deterministic_parity", "device", "blind_audio"],
    "baselines": {
      "temporal": {
        "artifact": _relative(temporal_baseline),
        "artifact_bytes": temporal_bytes,
        "weight_sha256": _weight_sha256(temporal_baseline),
        "receipt": "validation/results/MRT2TemporalBodyStreamingCarry_validation.json",
      },
      "depth": {
        "artifact": _relative(depth_baseline),
        "artifact_bytes": depth_bytes,
        "weight_sha256": _weight_sha256(depth_baseline),
        "receipt": "validation/results/MRT2DepthBodyRollout_f16_validation.json",
        "acceptance_boundary": "prior distributional plus device-listening gate",
      },
    },
    "candidates": candidates,
    "decision": {
      "selected": "uncompressed_temporal_plus_existing_fp16_depth",
      "compressed_variant_selected": False,
      "summary": (
        "All six compressed component candidates remained finite but failed "
        "their declared deterministic-reference gate. The experiment stops "
        "there and retains the uncompressed temporal plus existing FP16 "
        "depth configuration."
      ),
    },
  }
  OUTPUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  OUTPUT_MD.write_text(_markdown(report))
  print(f"Wrote {_relative(OUTPUT_JSON)}")
  print(f"Wrote {_relative(OUTPUT_MD)}")


if __name__ == "__main__":
  main()
