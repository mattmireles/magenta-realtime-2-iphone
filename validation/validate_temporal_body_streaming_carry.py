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

"""Validate the one-model host-owned MRT2 temporal streaming boundary."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

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
    TEMPORAL_SINKS,
    TEMPORAL_SOURCE_DIM,
    TemporalAttentionState,
    TemporalBodyCoreMLStreamingCarryWrapper,
    TemporalBodyStepWrapper,
    TemporalLayerState,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = (
    REPO_ROOT / "build" / "models" / "mrt2_temporal_body_streaming_carry_01.mlpackage"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "validation"
    / "results"
    / "MRT2TemporalBodyStreamingCarry_validation.json"
)
DEFAULT_SUMMARY = DEFAULT_REPORT.with_suffix(".md")
DEFAULT_FIXTURE = REPO_ROOT / "fixtures" / "temporal_streaming_carry_64.npz"
DEVICE_FIXTURE_STEMS = {
    "temporal_inputs": "temporal_streaming_carry_64_temporal_inputs_f32.bin",
    "source_encoded": "temporal_streaming_carry_64_source_encoded_f32.bin",
    "reference_outputs": "temporal_streaming_carry_64_reference_outputs_f32.bin",
}
NEGATIVE_BIAS = np.float16(-1e4)


def make_valid_bias(valid_history: int) -> np.ndarray:
    """Return sink + 41-cache + current-frame additive attention bias."""
    if not 0 <= valid_history <= MRT2_LOCAL_WINDOW_FRAMES:
        raise ValueError(f"valid_history must be in [0, {MRT2_LOCAL_WINDOW_FRAMES}]")
    extent = TEMPORAL_SINKS + MRT2_LOCAL_WINDOW_FRAMES + 1
    bias = np.full((1, 1, 1, extent), NEGATIVE_BIAS, dtype=np.float16)
    bias[..., 0] = 0
    if valid_history:
        bias[..., 1 : 1 + valid_history] = 0
    bias[..., -1] = 0
    return bias


def empty_cache_arrays() -> dict[str, np.ndarray]:
    """Return the 48 chronological FP16 host cache arrays."""
    shape = (
        1,
        MRT2_LOCAL_WINDOW_FRAMES,
        MRT2_TEMPORAL_HEADS,
        MRT2_HEAD_DIM,
    )
    return {
        name: np.zeros(shape, dtype=np.float16)
        for name in TemporalBodyCoreMLStreamingCarryWrapper.cache_input_names()
    }


def apply_cache_updates(
    caches: Mapping[str, np.ndarray],
    updates: Mapping[str, np.ndarray],
    valid_history: int,
) -> None:
    """Append one update to every chronological cache without allocating."""
    input_names = TemporalBodyCoreMLStreamingCarryWrapper.cache_input_names()
    output_names = TemporalBodyCoreMLStreamingCarryWrapper.cache_update_output_names()
    for input_name, output_name in zip(input_names, output_names, strict=True):
        cache = caches[input_name]
        update = np.asarray(updates[output_name], dtype=np.float16)
        if valid_history < MRT2_LOCAL_WINDOW_FRAMES:
            cache[:, valid_history : valid_history + 1] = update
        else:
            cache[:, :-1] = cache[:, 1:]
            cache[:, -1:] = update


def _quantize_attention_state(state: TemporalAttentionState) -> TemporalAttentionState:
    """Mirror the FP16 Core ML cache boundary in the readable reference."""
    return TemporalAttentionState(
        key_cache=state.key_cache.to(torch.float16).to(torch.float32),
        value_cache=state.value_cache.to(torch.float16).to(torch.float32),
        mask=state.mask,
        step=state.step,
    )


def _quantize_state(
    state: tuple[TemporalLayerState, ...],
) -> tuple[TemporalLayerState, ...]:
    return tuple(
        TemporalLayerState(
            self_attention=_quantize_attention_state(layer.self_attention),
            cross_attention=_quantize_attention_state(layer.cross_attention),
        )
        for layer in state
    )


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    correlation = np.corrcoef(candidate.reshape(-1), reference.reshape(-1))[0, 1]
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "correlation": float(correlation),
        "finite_ratio": float(np.isfinite(candidate).mean()),
    }


def _fixture_inputs(steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    temporal = rng.normal(0, 0.15, (steps, 1, 1, MRT2_MODEL_DIM)).astype(np.float32)
    source = rng.normal(0, 0.15, (steps, 1, 1, TEMPORAL_SOURCE_DIM)).astype(np.float32)
    return temporal, source


def _reference_outputs(
    temporal: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    model = TemporalBodyStepWrapper().eval()
    state = model.initial_state()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for temporal_step, source_step in zip(temporal, source, strict=True):
            output, state = model(
                torch.from_numpy(temporal_step),
                torch.from_numpy(source_step),
                state,
            )
            state = _quantize_state(state)
            outputs.append(output.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=1)


def _pytorch_streaming_outputs(
    temporal: np.ndarray,
    source: np.ndarray,
) -> tuple[np.ndarray, bool]:
    model = TemporalBodyCoreMLStreamingCarryWrapper().eval()
    caches = empty_cache_arrays()
    outputs: list[np.ndarray] = []
    changed_after_warmup = False
    with torch.no_grad():
        for step, (temporal_step, source_step) in enumerate(
            zip(temporal, source, strict=True)
        ):
            valid_history = min(step, MRT2_LOCAL_WINDOW_FRAMES)
            torch_inputs = [
                torch.from_numpy(temporal_step),
                torch.from_numpy(source_step),
                torch.from_numpy(make_valid_bias(valid_history)),
                *[torch.from_numpy(caches[name]) for name in model.cache_input_names()],
            ]
            result = model(*torch_inputs)
            output = result[0].detach().cpu().numpy().astype(np.float32)
            update_map = {
                name: value.detach().cpu().numpy()
                for name, value in zip(
                    model.cache_update_output_names(), result[1:], strict=True
                )
            }
            if step == 1:
                fresh_inputs = [
                    torch.from_numpy(temporal_step),
                    torch.from_numpy(source_step),
                    torch.from_numpy(make_valid_bias(0)),
                    *[
                        torch.zeros_like(torch.from_numpy(caches[name]))
                        for name in model.cache_input_names()
                    ],
                ]
                fresh_output = model(*fresh_inputs)[0].detach().cpu().numpy()
                changed_after_warmup = not np.array_equal(output, fresh_output)
            apply_cache_updates(caches, update_map, valid_history)
            outputs.append(output)
    return np.concatenate(outputs, axis=1), changed_after_warmup


def _coreml_streaming_outputs(
    model_path: Path,
    temporal: np.ndarray,
    source: np.ndarray,
) -> tuple[np.ndarray, list[float], bool]:
    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    caches = empty_cache_arrays()
    outputs: list[np.ndarray] = []
    timings: list[float] = []
    changed_after_warmup = False
    for step, (temporal_step, source_step) in enumerate(
        zip(temporal, source, strict=True)
    ):
        valid_history = min(step, MRT2_LOCAL_WINDOW_FRAMES)
        inputs: dict[str, Any] = {
            "temporal_inputs": temporal_step,
            "source_encoded": source_step,
            TemporalBodyCoreMLStreamingCarryWrapper.cache_valid_bias_name: (
                make_valid_bias(valid_history)
            ),
            **caches,
        }
        started = time.perf_counter()
        result = model.predict(inputs)
        timings.append((time.perf_counter() - started) * 1_000)
        output = np.asarray(result["temporal_outputs"], dtype=np.float32)
        if step == 1:
            fresh = dict(inputs)
            fresh[TemporalBodyCoreMLStreamingCarryWrapper.cache_valid_bias_name] = (
                make_valid_bias(0)
            )
            for name, cache in caches.items():
                fresh[name] = np.zeros_like(cache)
            fresh_output = np.asarray(
                model.predict(fresh)["temporal_outputs"], dtype=np.float32
            )
            changed_after_warmup = not np.array_equal(output, fresh_output)
        apply_cache_updates(caches, result, valid_history)
        outputs.append(output)
    return np.concatenate(outputs, axis=1), timings, changed_after_warmup


def _write_device_fixture(
    fixture: Path,
    *,
    temporal: np.ndarray,
    source: np.ndarray,
    reference: np.ndarray,
) -> dict[str, str]:
    """Write raw little-endian Float32 arrays readable without an NPZ library on iOS."""
    arrays = {
        "temporal_inputs": temporal,
        "source_encoded": source,
        "reference_outputs": reference,
    }
    paths: dict[str, str] = {}
    for name, array in arrays.items():
        path = fixture.parent / DEVICE_FIXTURE_STEMS[name]
        np.asarray(array, dtype="<f4").tofile(path)
        paths[name] = str(path)
    metadata_path = fixture.parent / "temporal_streaming_carry_64_device_fixture.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "mrt2-temporal-streaming-device-fixture-v1",
                "dtype": "little-endian-float32",
                "steps": int(temporal.shape[0]),
                "window_frames": MRT2_LOCAL_WINDOW_FRAMES,
                "arrays": {
                    name: {
                        "path": Path(path).name,
                        "shape": list(arrays[name].shape),
                        "byte_count": int(arrays[name].size * np.dtype("<f4").itemsize),
                    }
                    for name, path in paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    paths["metadata"] = str(metadata_path)
    return paths


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps <= MRT2_LOCAL_WINDOW_FRAMES:
        raise ValueError(
            f"--steps must exceed the {MRT2_LOCAL_WINDOW_FRAMES}-frame window"
        )
    temporal, source = _fixture_inputs(args.steps, args.seed)
    reference = _reference_outputs(temporal, source)
    pytorch_output, pytorch_state_read = _pytorch_streaming_outputs(temporal, source)
    report: dict[str, Any] = {
        "schema": "mrt2-temporal-body-streaming-carry-validation-v1",
        "steps": args.steps,
        "window_frames": MRT2_LOCAL_WINDOW_FRAMES,
        "wrapped_steps": args.steps - MRT2_LOCAL_WINDOW_FRAMES,
        "seed": args.seed,
        "cache_input_count": len(
            TemporalBodyCoreMLStreamingCarryWrapper.cache_input_names()
        ),
        "cache_update_output_count": len(
            TemporalBodyCoreMLStreamingCarryWrapper.cache_update_output_names()
        ),
        "pytorch_streaming_vs_reference": _metrics(pytorch_output, reference),
        "pytorch_fresh_vs_warmed_diverged": pytorch_state_read,
    }
    if not args.skip_coreml:
        if not args.model.exists():
            raise FileNotFoundError(args.model)
        coreml_output, timings, coreml_state_read = _coreml_streaming_outputs(
            args.model, temporal, source
        )
        report.update(
            {
                "model_path": str(args.model),
                "coreml_vs_reference": _metrics(coreml_output, reference),
                "coreml_vs_pytorch_streaming": _metrics(coreml_output, pytorch_output),
                "coreml_fresh_vs_warmed_diverged": coreml_state_read,
                "coreml_cpu_only_timing_ms": {
                    "p50": float(np.percentile(timings, 50)),
                    "p99": float(np.percentile(timings, 99)),
                    "iterations": len(timings),
                    "scope": "Mac CPU_ONLY validation only; not device evidence",
                },
            }
        )
    if args.write_fixture:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.fixture,
            temporal_inputs=temporal,
            source_encoded=source,
            reference_outputs=reference,
        )
        report["fixture_path"] = str(args.fixture)
        report["device_fixture_paths"] = _write_device_fixture(
            args.fixture,
            temporal=temporal,
            source=source,
            reference=reference,
        )
    return report


def _write_summary(report: dict[str, Any], path: Path) -> None:
    coreml = report.get("coreml_vs_reference")
    lines = [
        "# MRT2 Temporal Body Streaming Carry Validation",
        "",
        f"- Steps: {report['steps']} (window: {report['window_frames']}; wrapped: {report['wrapped_steps']})",
        "- Boundary: 48 ordinary cache tensors in, 48 one-frame updates out",
        f"- PyTorch streaming vs reference correlation: {report['pytorch_streaming_vs_reference']['correlation']:.12f}",
        f"- PyTorch fresh vs warmed diverged: {report['pytorch_fresh_vs_warmed_diverged']}",
    ]
    if coreml:
        lines.extend(
            [
                f"- Core ML vs reference correlation: {coreml['correlation']:.12f}",
                f"- Core ML vs reference max error: {coreml['max_abs_error']:.10f}",
                f"- Core ML finite ratio: {coreml['finite_ratio']:.6f}",
                f"- Core ML fresh vs warmed diverged: {report['coreml_fresh_vs_warmed_diverged']}",
            ]
        )
    lines.extend(
        [
            "",
            "This receipt crosses the 41-frame window and therefore exercises host-ring wraparound, not only no-wrap startup.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--skip-coreml", action="store_true")
    parser.add_argument("--write-fixture", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_summary(report, args.summary)
    print(f"Wrote {args.report}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
