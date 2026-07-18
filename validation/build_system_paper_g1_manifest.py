#!/usr/bin/env python3
"""Build a public G1 manifest from private Crossfade device receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from validation.verify_system_paper_gate import G1_SCHEMA
except ModuleNotFoundError:  # Direct `python validation/...py` execution.
    from verify_system_paper_gate import G1_SCHEMA


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILES = {
    "temporal": ("mrt2_temporal_body_streaming_carry_01.mlmodelc/weights/weight.bin"),
    "depth": "mrt2_depth_body_rollout.mlmodelc/weights/weight.bin",
    "decoder": "spectrostream_decoder_conv_nchw.mlmodelc/weights/weight.bin",
}
TRACE_MODELS = {
    "temporal": "mrt2_temporal_body_streaming_carry_01",
    "depth": "mrt2_depth_body_rollout",
    "decoder": "spectrostream_decoder_conv_nchw",
}
STATE_ARTIFACTS = (
    REPO_ROOT / "fixtures" / "temporal_streaming_carry_64.npz",
    REPO_ROOT / "fixtures" / "temporal_streaming_carry_64_temporal_inputs_f32.bin",
    REPO_ROOT / "fixtures" / "temporal_streaming_carry_64_source_encoded_f32.bin",
    REPO_ROOT / "fixtures" / "temporal_streaming_carry_64_reference_outputs_f32.bin",
    REPO_ROOT
    / "validation"
    / "results"
    / "MRT2TemporalBodyStreamingCarry_validation.json",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical(path: Path, root: Path) -> str:
    try:
        return f"crossfade-private/{path.resolve().relative_to(root.resolve())}"
    except ValueError:
        return path.name


def _trace_check_map(trace: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    checks = trace.get("checks", {}).get(key, [])
    return {
        item["model"]: item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("model"), str)
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    matrix = _load(args.cold_matrix)
    trace = _load(args.trace_dir / "placement-evidence.json")
    if matrix.get("allPassed") is not True or matrix.get("passedLaunches") != 10:
        raise ValueError("cold-launch matrix must contain exactly 10 passing launches")
    if trace.get("ane_residency_proven") is not True:
        raise ValueError("trace placement evidence must pass")

    model_hashes = {
        family: _sha256(args.model_root / path) for family, path in MODEL_FILES.items()
    }
    policy = {family: "cpuAndNeuralEngine" for family in model_hashes}
    launches = []
    for item in matrix["launches"]:
        placement = item["placement"]
        state = item["stateProof"]
        log_path = args.crossfade_repo / item["log"]
        launches.append(
            {
                "runId": f"cold-{int(item['launch']):02d}",
                "coldLaunch": True,
                "durationSeconds": item["durationSeconds"],
                "modelSha256": model_hashes,
                "computePolicy": policy,
                "temporalComputePlan": {
                    "passed": placement["passed"] == "true",
                    "cpuOperationCount": int(placement["cpu_ops"]),
                    "gpuOperationCount": int(placement["gpu_ops"]),
                    "aneOperationCount": int(placement["ane_ops"]),
                    "cpuEstimatedCostWeight": float(placement["cpu_cost"]),
                    "gpuEstimatedCostWeight": float(placement["gpu_cost"]),
                    "aneEstimatedCostWeight": float(placement["ane_cost"]),
                },
                "stateProofPassed": state["passed"] == "true",
                "fixtureProofPassed": state["fixture_passed"] == "true",
                "artifactSha256": {
                    _logical(log_path, args.crossfade_repo): _sha256(log_path)
                },
            }
        )

    ane_checks = _trace_check_map(trace, "required_models_have_ane_predictions")
    runtime_checks = _trace_check_map(
        trace, "required_runtime_models_have_coreml_predictions"
    )
    trace_hashes = {
        _logical(path, args.crossfade_repo): _sha256(path)
        for path in (
            args.trace_dir / "placement-evidence.json",
            args.trace_dir / "coreml-os-signpost.xml",
            args.trace_dir / "ane-hw-intervals.xml",
            args.trace_dir / "metal-gpu-intervals.xml",
        )
    }
    first_state = matrix["launches"][0]["stateProof"]
    # The signed host intentionally logs proof metrics to three decimals.
    # Convert rounded values into conservative bounds rather than presenting
    # them as higher-precision measurements.
    rounding_half_unit = 0.0005
    state_artifact_hashes = {
        str(path.relative_to(REPO_ROOT)): _sha256(path) for path in STATE_ARTIFACTS
    }
    manifest = {
        "schema": G1_SCHEMA,
        "device": {
            "modelIdentifier": args.model_identifier,
            "chip": args.chip,
            "osVersion": args.os_version,
        },
        "deviceRole": args.device_role,
        "crossfadeSourceCommit": matrix["sourceCommit"],
        "launches": launches,
        "traceEvidence": {
            "aneModelFamilies": {
                family: {
                    "anePredictionCount": ane_checks[model]["ane_prediction_count"],
                    "anePredictionTotalNs": ane_checks[model][
                        "ane_prediction_total_ns"
                    ],
                }
                for family, model in TRACE_MODELS.items()
                if family in {"temporal", "decoder"}
            },
            "runtimeModelFamilies": {
                family: {
                    "coremlPredictionCount": runtime_checks[model][
                        "coreml_prediction_count"
                    ],
                    "coremlPredictionTotalNs": runtime_checks[model][
                        "coreml_prediction_total_ns"
                    ],
                }
                for family, model in TRACE_MODELS.items()
            },
            "appGpuIntervalCount": len(
                trace["checks"]["app_absent_from_gpu_intervals"][
                    "matching_gpu_processes"
                ]
            ),
            "appGpuTotalNs": trace["checks"]["app_absent_from_gpu_intervals"][
                "app_gpu_total_ns"
            ],
            "modelSha256": model_hashes,
            "artifactSha256ByPath": trace_hashes,
        },
        "stateEvidence": {
            "artifactSha256": model_hashes["temporal"],
            "freshWarmedMaxAbsDelta": max(
                0.0, float(first_state["max_diff"]) - rounding_half_unit
            ),
            "streamingFrames": int(first_state["fixture_frames"]),
            "windowFrames": 41,
            "correlation": max(
                0.0,
                min(
                    float(item["stateProof"]["fixture_correlation"])
                    for item in matrix["launches"]
                )
                - rounding_half_unit,
            ),
            "maxAbsoluteError": max(
                float(item["stateProof"]["fixture_max_error"])
                for item in matrix["launches"]
            )
            + rounding_half_unit,
            "meanAbsoluteError": max(
                float(item["stateProof"]["fixture_mean_error"])
                for item in matrix["launches"]
            )
            + rounding_half_unit,
            # The in-app gate tests the unrounded value for exact finiteness.
            "finiteRatio": 1.0,
            "loggedMetricDecimals": 3,
            "boundsIncludeLoggingRounding": True,
            "artifactSha256ByPath": state_artifact_hashes,
        },
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfade-repo", type=Path, required=True)
    parser.add_argument("--cold-matrix", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--device-role", choices=("primary", "cross-device"), required=True
    )
    parser.add_argument("--model-identifier", required=True)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--os-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
