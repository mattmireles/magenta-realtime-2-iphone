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

"""Run the MRT2 long-horizon token/decoder crossover.

This harness localizes late-horizon periodic audio by separating the token
generator from the decoder/DSP implementation. It can produce token streams
from either the in-process MLX reference or the exact compiled Core ML graphs
used by ``CrossfadeGenerationRuntime`` and decode either token stream through:

* the MLX SpectroStream streaming decoder; or
* the shipped FP16 Core ML decoder plus the production C++ iSTFT/overlap-add.

The four cells distinguish model/trajectory failures from decoder/DSP failures.
All token files use codebook-local values shaped ``[frames, 12]`` in ``0..1023``.
For an exact 600-second decoder output, generate 15,001 token frames: 15,000
audible frames plus the one-frame SpectroStream lookahead.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "Scratchpad" / "system_paper_models"
DEFAULT_REFERENCE_MODELS_DIR = REPO_ROOT / "Scratchpad" / "coreml_proof_models"
DEFAULT_CHECKPOINT = (
    Path.home()
    / "Documents"
    / "Magenta"
    / "magenta-rt-v2"
    / "checkpoints"
    / "mrt2_small.safetensors"
)
DEFAULT_SOURCE_CONDITIONING = DEFAULT_REFERENCE_MODELS_DIR / "warm.bin"
DEFAULT_SECONDS = 600
DEFAULT_SEED = 20_260_718
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 40
DEFAULT_REFRESH_SECONDS = 10.0
RESET_POLICY_AUTO = "auto"
RESET_POLICY_OFF = "off"
RESET_POLICY_BOTH = "both"
RESET_POLICY_KV_ONLY = "kv-only"
RESET_POLICY_FEEDBACK_ONLY = "feedback-only"
RESET_POLICIES = (
    RESET_POLICY_AUTO,
    RESET_POLICY_OFF,
    RESET_POLICY_BOTH,
    RESET_POLICY_KV_ONLY,
    RESET_POLICY_FEEDBACK_ONLY,
)
SAMPLE_RATE = 48_000
TOKEN_RATE = 25
RVQ_LEVELS = 12
CODEBOOK_SIZE = 1_024
RESERVED_TOKENS = 6
MODEL_DIM = 1_024
DECODER_EMBEDDING_DIM = 256
TEMPORAL_WINDOW_FRAMES = 41
TEMPORAL_ATTENTION_EXTENT = 43
DECODER_INPUT_FRAMES = 25
DECODER_STRIDE_FRAMES = 24
DECODER_STFT_FRAMES = 96
STFT_BINS = 480
STFT_CHANNELS = 4
STFT_HOP_SAMPLES = 480
STFT_FRAME_LENGTH = 960
UINT64_MASK = (1 << 64) - 1


def _sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write stable, readable JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _refresh_interval_frames(refresh_seconds: float) -> int | None:
    """Map a nonnegative refresh duration to frames; zero means no refresh."""
    if refresh_seconds < 0:
        raise ValueError("refresh seconds must be nonnegative")
    if refresh_seconds == 0:
        return None
    return max(1, int(round(refresh_seconds * TOKEN_RATE)))


def _resolve_reset_policy(refresh_seconds: float, requested: str) -> str:
    """Resolve the explicit state component reset at each refresh boundary."""
    if requested not in RESET_POLICIES:
        raise ValueError(f"unknown reset policy: {requested}")
    if requested == RESET_POLICY_AUTO:
        return RESET_POLICY_OFF if refresh_seconds == 0 else RESET_POLICY_BOTH
    if refresh_seconds == 0 and requested != RESET_POLICY_OFF:
        raise ValueError("refresh-seconds 0 requires reset policy off or auto")
    if refresh_seconds > 0 and requested == RESET_POLICY_OFF:
        raise ValueError("periodic refresh requires a non-off reset policy")
    return requested


class SplitMix64FloatGenerator:
    """Bit-for-bit port of ``CrossfadeSeededGenerator.nextUnitFloat``."""

    def __init__(self, seed: int):
        self.state = int(seed) & UINT64_MASK
        if self.state == 0:
            self.state = 0x9E37_79B9_7F4A_7C15

    def next_unit_float(self) -> np.float32:
        """Return the next 24-bit uniform value with Swift Float rounding."""
        self.state = (self.state + 0x9E37_79B9_7F4A_7C15) & UINT64_MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58_476D_1CE4_E5B9) & UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D0_49BB_1331_11EB) & UINT64_MASK
        value ^= value >> 31
        return np.float32(value >> 40) / np.float32(1 << 24)

    def gumbel(self, shape: tuple[int, ...]) -> np.ndarray:
        """Return Swift ``logf``-equivalent Gumbel noise."""
        output = np.empty(shape, dtype=np.float32)
        flat = output.reshape(-1)
        floor = np.float32(1e-7)
        for index in range(flat.size):
            uniform = max(self.next_unit_float(), floor)
            flat[index] = np.float32(-np.log(np.float32(-np.log(uniform))))
        return output


def _cache_names() -> list[str]:
    """Return the 48 temporal cache stems in runtime order."""
    kinds = ("self_key", "self_value", "cross_key", "cross_value")
    return [
        f"temporal_layer_{layer:02d}_{kind}_cache"
        for layer in range(12)
        for kind in kinds
    ]


def _valid_bias(valid_history: int) -> np.ndarray:
    """Build the exact streaming-carry attention validity bias."""
    bounded = min(max(int(valid_history), 0), TEMPORAL_WINDOW_FRAMES)
    bias = np.full((1, 1, 1, TEMPORAL_ATTENTION_EXTENT), -10_000, dtype=np.float16)
    bias[..., 0] = 0
    if bounded:
        bias[..., 1 : bounded + 1] = 0
    bias[..., -1] = 0
    return bias


def _append_cache_update(
    cache: np.ndarray, update: np.ndarray, valid_history: int
) -> None:
    """Apply the Swift host-cache append/shift operation in place."""
    dense = np.asarray(update, dtype=np.float16).reshape(1, 1, 8, 128)
    if valid_history >= TEMPORAL_WINDOW_FRAMES:
        cache[:, :-1] = cache[:, 1:]
        destination = TEMPORAL_WINDOW_FRAMES - 1
    else:
        destination = valid_history
    cache[:, destination : destination + 1] = dense


def summarize_tokens(
    tokens: np.ndarray,
    *,
    window_seconds: float = 30.0,
    token_rate: int = TOKEN_RATE,
) -> dict[str, Any]:
    """Summarize entropy, collapse, and short-cycle behavior over time."""
    values = np.asarray(tokens, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != RVQ_LEVELS:
        raise ValueError(f"Expected token shape [N, {RVQ_LEVELS}], got {values.shape}")
    if values.size and (int(values.min()) < 0 or int(values.max()) >= CODEBOOK_SIZE):
        raise ValueError("Token values must be codebook-local integers in 0..1023")
    window_frames = max(1, int(round(window_seconds * token_rate)))
    windows = []
    for start in range(0, values.shape[0], window_frames):
        chunk = values[start : start + window_frames]
        if not chunk.size:
            continue
        level_entropy = []
        dominant_share = []
        for level in range(RVQ_LEVELS):
            counts = np.bincount(chunk[:, level], minlength=CODEBOOK_SIZE)
            probabilities = counts[counts > 0].astype(np.float64) / chunk.shape[0]
            level_entropy.append(float(-np.sum(probabilities * np.log2(probabilities))))
            dominant_share.append(float(counts.max() / chunk.shape[0]))
        period_matches = {}
        for period in range(1, 9):
            if chunk.shape[0] <= period:
                period_matches[str(period)] = None
            else:
                period_matches[str(period)] = float(
                    np.mean(np.all(chunk[period:] == chunk[:-period], axis=1))
                )
        level_change_fraction = (
            np.mean(chunk[1:] != chunk[:-1], axis=1, dtype=np.float32)
            if chunk.shape[0] > 1
            else np.zeros(1, dtype=np.float32)
        )
        pulse_share = None
        if level_change_fraction.size >= token_rate:
            centered = level_change_fraction - float(level_change_fraction.mean())
            power = np.abs(np.fft.rfft(centered)) ** 2
            frequencies = np.fft.rfftfreq(centered.size, d=1.0 / token_rate)
            positive = frequencies > 0
            band = (frequencies >= 4.0) & (frequencies <= 12.5)
            denominator = float(power[positive].sum())
            pulse_share = (
                float(power[band].sum() / denominator) if denominator > 0 else 0.0
            )
        windows.append(
            {
                "startSeconds": start / token_rate,
                "endSeconds": min(start + window_frames, values.shape[0]) / token_rate,
                "frames": int(chunk.shape[0]),
                "meanLevelEntropyBits": float(np.mean(level_entropy)),
                "minLevelEntropyBits": float(np.min(level_entropy)),
                "maxDominantTokenShare": float(np.max(dominant_share)),
                "distinctFrameRatio": float(
                    np.unique(chunk, axis=0).shape[0] / chunk.shape[0]
                ),
                "meanAdjacentLevelChangeFraction": float(
                    np.mean(level_change_fraction)
                ),
                "shortPeriodExactFrameMatch": period_matches,
                "levelChangePulseShare4To12_5Hz": pulse_share,
            }
        )
    return {
        "schema": "mrt2-token-trajectory-summary-v1",
        "frames": int(values.shape[0]),
        "secondsAt25Hz": float(values.shape[0] / token_rate),
        "windowSeconds": float(window_seconds),
        "windows": windows,
    }


def _save_token_run(
    output_dir: Path,
    *,
    source: str,
    seed: int,
    prompt: str,
    seconds: int,
    refresh_seconds: float,
    tokens: np.ndarray,
    elapsed_seconds: float,
    temporal_reset_count: int,
    inputs: dict[str, Path],
    reset_policy: str | None = None,
    kv_reset_count: int | None = None,
    feedback_reset_count: int | None = None,
) -> Path:
    """Save tokens, trajectory summary, and a hash-linked manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / f"{source}-seed-{seed}-tokens.npy"
    np.save(token_path, np.asarray(tokens, dtype=np.int16))
    summary_path = output_dir / f"{source}-seed-{seed}-token-summary.json"
    _write_json(summary_path, summarize_tokens(tokens))
    manifest_path = output_dir / f"{source}-seed-{seed}-manifest.json"
    resolved_policy = reset_policy or (
        RESET_POLICY_OFF if refresh_seconds == 0 else RESET_POLICY_BOTH
    )
    resolved_kv_count = (
        int(kv_reset_count)
        if kv_reset_count is not None
        else int(
            temporal_reset_count
            if resolved_policy in {RESET_POLICY_BOTH, RESET_POLICY_KV_ONLY}
            else 0
        )
    )
    resolved_feedback_count = (
        int(feedback_reset_count)
        if feedback_reset_count is not None
        else int(
            temporal_reset_count
            if resolved_policy in {RESET_POLICY_BOTH, RESET_POLICY_FEEDBACK_ONLY}
            else 0
        )
    )
    _write_json(
        manifest_path,
        {
            "schema": "mrt2-long-horizon-token-run-v1",
            "tokenSource": source,
            "seed": int(seed),
            "prompt": str(prompt),
            "temperature": DEFAULT_TEMPERATURE,
            "topK": DEFAULT_TOP_K,
            "trajectoryRefreshSeconds": float(refresh_seconds),
            "refreshMode": "off" if refresh_seconds == 0 else "periodic",
            "resetPolicy": resolved_policy,
            "temporalResetCount": int(temporal_reset_count),
            "kvResetCount": resolved_kv_count,
            "feedbackResetCount": resolved_feedback_count,
            "rngResetCount": 0,
            "absoluteStepResetCount": 0,
            "requestedAudibleSeconds": int(seconds),
            "tokenFrames": int(tokens.shape[0]),
            "elapsedSeconds": float(elapsed_seconds),
            "tokens": {"path": str(token_path), "sha256": _sha256(token_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "inputs": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in inputs.items()
            },
        },
    )
    return manifest_path


def generate_coreml_tokens(args: argparse.Namespace) -> Path:
    """Generate tokens through the exact shipped Core ML graph boundary."""
    import coremltools as ct

    models_dir = Path(args.models_dir)
    reference_models_dir = Path(args.reference_models_dir)
    temporal_path = models_dir / "mrt2_temporal_body_streaming_carry_01.mlmodelc"
    depth_path = models_dir / "mrt2_depth_body_rollout.mlmodelc"
    embedder_path = models_dir / "mrt2_depth_embedder_f32.bin"
    source_path = Path(args.source_conditioning)
    for path in (temporal_path, depth_path, embedder_path, source_path):
        if not path.exists():
            raise FileNotFoundError(path)
    del reference_models_dir
    temporal = ct.models.CompiledMLModel(
        str(temporal_path), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    depth = ct.models.CompiledMLModel(
        str(depth_path), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    source = np.fromfile(source_path, dtype="<f4").reshape(1, 1, DECODER_EMBEDDING_DIM)
    embedder = np.fromfile(embedder_path, dtype="<f4").reshape(-1, MODEL_DIM)
    sos_feedback = embedder[np.zeros(RVQ_LEVELS, dtype=np.int64)].mean(
        axis=0, dtype=np.float32
    )
    cache_names = _cache_names()
    caches = {
        name: np.zeros((1, TEMPORAL_WINDOW_FRAMES, 8, 128), dtype=np.float16)
        for name in cache_names
    }
    previous_feedback: np.ndarray | None = None
    valid_history = 0
    rng = SplitMix64FloatGenerator(args.seed)
    refresh_frames = _refresh_interval_frames(args.refresh_seconds)
    reset_policy = _resolve_reset_policy(args.refresh_seconds, args.reset_policy)
    temporal_reset_count = 0
    kv_reset_count = 0
    feedback_reset_count = 0
    token_frames = int(round(args.seconds * TOKEN_RATE)) + 1
    tokens = np.empty((token_frames, RVQ_LEVELS), dtype=np.int32)
    started = time.perf_counter()
    for frame in range(token_frames):
        if refresh_frames is not None and frame > 0 and frame % refresh_frames == 0:
            if reset_policy in {RESET_POLICY_BOTH, RESET_POLICY_KV_ONLY}:
                for cache in caches.values():
                    cache.fill(0)
                valid_history = 0
                kv_reset_count += 1
            if reset_policy in {RESET_POLICY_BOTH, RESET_POLICY_FEEDBACK_ONLY}:
                previous_feedback = None
                feedback_reset_count += 1
            temporal_reset_count += 1
        temporal_inputs: dict[str, np.ndarray] = {
            "temporal_inputs": (
                sos_feedback if previous_feedback is None else previous_feedback
            )
            .reshape(1, 1, MODEL_DIM)
            .astype(np.float32, copy=False),
            "source_encoded": source,
            "cache_valid_bias": _valid_bias(valid_history),
        }
        temporal_inputs.update({f"{name}_in": caches[name] for name in cache_names})
        temporal_output = temporal.predict(temporal_inputs)
        temporal_frame = np.asarray(
            temporal_output["temporal_outputs"], dtype=np.float32
        ).reshape(1, 1, MODEL_DIM)
        for name in cache_names:
            _append_cache_update(
                caches[name], temporal_output[f"{name}_updates"], valid_history
            )
        valid_history = min(valid_history + 1, TEMPORAL_WINDOW_FRAMES)
        depth_output = depth.predict(
            {
                "temporal_frame": temporal_frame,
                "gumbel_noise": rng.gumbel((RVQ_LEVELS, CODEBOOK_SIZE)),
                "inverse_temperature": np.array(
                    [1.0 / max(0.05, args.temperature)], dtype=np.float32
                ),
            }
        )
        frame_tokens = np.asarray(
            depth_output["sampled_codes"], dtype=np.int32
        ).reshape(-1)
        if frame_tokens.shape != (RVQ_LEVELS,):
            raise ValueError(f"Unexpected sampled_codes shape {frame_tokens.shape}")
        tokens[frame] = frame_tokens
        previous_feedback = np.asarray(
            depth_output["temporal_feedback"], dtype=np.float32
        ).reshape(MODEL_DIM)
        if frame % 250 == 0:
            print(f"coreml frame {frame}/{token_frames}", flush=True)
    elapsed = time.perf_counter() - started
    return _save_token_run(
        Path(args.output_dir),
        source="coreml-port",
        seed=args.seed,
        prompt=args.prompt,
        seconds=args.seconds,
        refresh_seconds=args.refresh_seconds,
        tokens=tokens,
        elapsed_seconds=elapsed,
        temporal_reset_count=temporal_reset_count,
        reset_policy=reset_policy,
        kv_reset_count=kv_reset_count,
        feedback_reset_count=feedback_reset_count,
        inputs={
            "temporalModelMetadata": temporal_path / "metadata.json",
            "depthModelMetadata": depth_path / "metadata.json",
            "depthEmbedder": embedder_path,
            "sourceConditioning": source_path,
        },
    )


def _build_mlx_depth_sampler(checkpoint_path: Path):
    """Build the unquantized in-process MLX reference sampler."""
    import magenta_rt  # noqa: F401
    import mlx.core as mx

    from magenta_rt.mlx import load_weights as mlx_load_weights
    from magenta_rt.mlx import model
    from magenta_rt.mlx import spectrostream
    from magenta_rt.mlx import system

    experiment = model.get_model_class("mrt2_small")()
    experiment.compute_dtype = mx.float32
    sampler = system.MagentaRT2Sampler.Config(
        depthformer=experiment.depthformer_config(),
        spectrostream=spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
            rvq_truncation_level=experiment.spectrostream.rvq_truncation_level,
            use_unique_codes=False,
        ),
        int16_outputs=False,
    ).make()
    mlx_load_weights.load_weights(
        sampler, checkpoint_path, num_input_channels=experiment.input_num_channels
    )
    return sampler


def generate_mlx_tokens(args: argparse.Namespace) -> Path:
    """Generate tokens with the clean MLX reference at the same horizon."""
    import magenta_rt  # noqa: F401  (installs vendored sequence_layers import path)
    import mlx.core as mx
    import sequence_layers.mlx as sl

    checkpoint_path = Path(args.checkpoint)
    source_path = Path(args.source_conditioning)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    sampler = _build_mlx_depth_sampler(checkpoint_path)
    depth_sampler = sampler.layers[0]
    decoder = depth_sampler.decoder
    source = np.fromfile(source_path, dtype="<f4").reshape(1, 1, DECODER_EMBEDDING_DIM)
    encoded = sl.Sequence(
        mx.array(source, dtype=mx.float32), mx.ones((1, 1), dtype=mx.bool_)
    )
    constants = {
        "temperature": mx.array([args.temperature], dtype=mx.float32),
        "top_k": mx.array([args.top_k], dtype=mx.int32),
        depth_sampler.conditioning_name: encoded,
    }
    dummy = sl.Sequence(
        mx.zeros((1, 1, 144), dtype=mx.int32), mx.ones((1, 1), dtype=mx.bool_)
    )
    outer_state = depth_sampler.get_initial_state(
        1, dummy.channel_spec, constants=constants, training=False
    )
    decoder_state = outer_state[2]
    rng, previous_frame, temporal_state, step = decoder_state
    rng = mx.stack([mx.random.key(int(args.seed))])
    decoder_state = (rng, previous_frame, temporal_state, step)
    fresh_state = decoder.get_initial_state(
        1, previous_frame.channel_spec, constants=constants, training=False
    )
    refresh_frames = _refresh_interval_frames(args.refresh_seconds)
    reset_policy = _resolve_reset_policy(args.refresh_seconds, args.reset_policy)
    temporal_reset_count = 0
    kv_reset_count = 0
    feedback_reset_count = 0
    token_frames = int(round(args.seconds * TOKEN_RATE)) + 1
    tokens = np.empty((token_frames, RVQ_LEVELS), dtype=np.int32)
    started = time.perf_counter()
    for frame in range(token_frames):
        if refresh_frames is not None and frame > 0 and frame % refresh_frames == 0:
            rng, previous, temporal, step = decoder_state
            _, initial_previous, initial_temporal, _ = fresh_state
            if reset_policy in {RESET_POLICY_BOTH, RESET_POLICY_FEEDBACK_ONLY}:
                previous = initial_previous
                feedback_reset_count += 1
            if reset_policy in {RESET_POLICY_BOTH, RESET_POLICY_KV_ONLY}:
                temporal = initial_temporal
                kv_reset_count += 1
            decoder_state = (rng, previous, temporal, step)
            temporal_reset_count += 1
        output, decoder_state, _ = decoder.step_with_emits(
            dummy, decoder_state, training=False, constants=constants
        )
        mx.eval(output.values, decoder_state)
        unique = np.asarray(output.values, dtype=np.int32).reshape(RVQ_LEVELS)
        levels = np.arange(RVQ_LEVELS, dtype=np.int32)
        tokens[frame] = unique - RESERVED_TOKENS - levels * CODEBOOK_SIZE
        if frame % 250 == 0:
            print(f"mlx frame {frame}/{token_frames}", flush=True)
    elapsed = time.perf_counter() - started
    return _save_token_run(
        Path(args.output_dir),
        source="mlx",
        seed=args.seed,
        prompt=args.prompt,
        seconds=args.seconds,
        refresh_seconds=args.refresh_seconds,
        tokens=tokens,
        elapsed_seconds=elapsed,
        temporal_reset_count=temporal_reset_count,
        reset_policy=reset_policy,
        kv_reset_count=kv_reset_count,
        feedback_reset_count=feedback_reset_count,
        inputs={"checkpoint": checkpoint_path, "sourceConditioning": source_path},
    )


def _load_raw_tokens(path: Path) -> np.ndarray:
    """Load and validate a local-code token trajectory."""
    tokens = np.asarray(np.load(path), dtype=np.int32)
    summarize_tokens(tokens, window_seconds=30.0)
    return tokens


def _load_codebooks(path: Path) -> np.ndarray:
    """Load the 12-level SpectroStream codebook resource."""
    values = np.fromfile(path, dtype="<f4")
    expected = RVQ_LEVELS * CODEBOOK_SIZE * DECODER_EMBEDDING_DIM
    if values.size != expected:
        raise ValueError(f"Expected {expected} codebook values, got {values.size}")
    return values.reshape(RVQ_LEVELS, CODEBOOK_SIZE, DECODER_EMBEDDING_DIM)


def _lookup_embeddings(codebooks: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Apply the production host CPU RVQ lookup and sum."""
    levels = np.arange(RVQ_LEVELS)[:, np.newaxis]
    selected = codebooks[levels, tokens.T]
    return np.sum(selected, axis=0, dtype=np.float32)


def _render_core_library(*, legacy_dual_window: bool = False) -> ctypes.CDLL:
    """Compile and load the production RenderCore as a temporary dylib."""
    source = REPO_ROOT / "Sources" / "CrossfadeRuntimeCore" / "RenderCore.cpp"
    include = REPO_ROOT / "Sources" / "CrossfadeRuntimeCore" / "include"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    mode = "legacy-dual" if legacy_dual_window else "trained-hann"
    output = Path(tempfile.gettempdir()) / f"libcrossfade-render-{digest}-{mode}.dylib"
    if not output.exists():
        command = [
            "/usr/bin/clang++",
            "-std=c++20",
            "-dynamiclib",
            "-O2",
            "-I",
            str(include),
            str(source),
            "-framework",
            "Accelerate",
            "-framework",
            "AudioToolbox",
        ]
        if legacy_dual_window:
            command.append("-DCROSSFADE_LEGACY_DUAL_SYNTHESIS_WINDOW=1")
        command += ["-o", str(output)]
        subprocess.run(command, check=True)
    library = ctypes.CDLL(str(output))
    library.render_core_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    library.render_core_create.restype = ctypes.c_void_p
    library.render_core_destroy.argtypes = [ctypes.c_void_p]
    library.render_core_render_decoder_stft.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    library.render_core_render_decoder_stft.restype = ctypes.c_uint32
    return library


def decode_coreml(args: argparse.Namespace) -> Path:
    """Decode either token source with the shipped Core ML decoder and host DSP."""
    import coremltools as ct
    import soundfile as sf

    tokens_path = Path(args.tokens)
    tokens = _load_raw_tokens(tokens_path)
    models_dir = Path(args.models_dir)
    codebooks_path = models_dir / "spectrostream_rvq_codebooks_12_f32.bin"
    decoder_path = models_dir / "spectrostream_decoder_conv_nchw.mlmodelc"
    codebooks = _load_codebooks(codebooks_path)
    decoder = ct.models.CompiledMLModel(
        str(decoder_path), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    library = _render_core_library(legacy_dual_window=args.legacy_dual_window_dsp)
    render_core = library.render_core_create(2, SAMPLE_RATE * 2)
    if not render_core:
        raise RuntimeError("render_core_create failed")
    output_path = Path(args.output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows = 0
    rendered_frames = 0
    decoder_stride = DECODER_STRIDE_FRAMES - args.decoder_context_frames
    crop_stft_frames = args.decoder_context_frames * 4
    started = time.perf_counter()
    try:
        with sf.SoundFile(
            output_path, mode="w", samplerate=SAMPLE_RATE, channels=2, subtype="FLOAT"
        ) as output:
            for start in range(
                0, tokens.shape[0] - DECODER_INPUT_FRAMES + 1, decoder_stride
            ):
                token_window = tokens[start : start + DECODER_INPUT_FRAMES]
                embeddings = _lookup_embeddings(codebooks, token_window)
                prediction = decoder.predict(
                    {"decoder_embeddings": embeddings[np.newaxis]}
                )
                stft = np.ascontiguousarray(
                    prediction["decoder_stft"], dtype=np.float32
                )
                expected_shape = (1, DECODER_STFT_FRAMES, STFT_BINS, STFT_CHANNELS)
                if stft.shape != expected_shape:
                    raise ValueError(f"Unexpected decoder_stft shape {stft.shape}")
                if windows > 0 and crop_stft_frames:
                    stft = np.ascontiguousarray(
                        stft[:, crop_stft_frames:], dtype=np.float32
                    )
                rendered_stft_frames = int(stft.shape[1])
                capacity = rendered_stft_frames * STFT_HOP_SAMPLES
                pcm = np.empty((capacity, 2), dtype=np.float32)
                rendered = int(
                    library.render_core_render_decoder_stft(
                        render_core,
                        stft.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        rendered_stft_frames,
                        pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        capacity,
                        2,
                    )
                )
                if rendered != capacity:
                    raise RuntimeError(
                        f"RenderCore produced {rendered}/{capacity} frames"
                    )
                output.write(pcm)
                windows += 1
                rendered_frames += rendered
                if windows % 25 == 0:
                    print(f"coreml decoder window {windows}", flush=True)
    finally:
        library.render_core_destroy(render_core)
    report_path = output_path.with_suffix(".json")
    _write_json(
        report_path,
        {
            "schema": "mrt2-crossover-decode-v1",
            "tokenPath": str(tokens_path),
            "tokenSha256": _sha256(tokens_path),
            "decoder": "shipped-coreml-fp16-plus-production-rendercore",
            "decoderMetadataSha256": _sha256(decoder_path / "metadata.json"),
            "renderCoreSha256": _sha256(
                REPO_ROOT / "Sources" / "CrossfadeRuntimeCore" / "RenderCore.cpp"
            ),
            "dspWindow": (
                "legacy-dual-normalized-hann"
                if args.legacy_dual_window_dsp
                else "trained-periodic-hann"
            ),
            "codebooksSha256": _sha256(codebooks_path),
            "windows": windows,
            "decoderContextFrames": int(args.decoder_context_frames),
            "decoderStrideFrames": int(decoder_stride),
            "renderedFrames": rendered_frames,
            "durationSeconds": rendered_frames / SAMPLE_RATE,
            "elapsedSeconds": time.perf_counter() - started,
            "wav": {"path": str(output_path), "sha256": _sha256(output_path)},
        },
    )
    return report_path


def decode_mlx_stft_core_dsp(args: argparse.Namespace) -> Path:
    """Decode with the MLX pre-iSTFT network and production C++ DSP.

    This fifth arm splits the original H3 bucket.  It keeps the exact 25-frame
    window/24-frame stride and RenderCore overlap state used by the phone while
    replacing only the FP16 Core ML decoder graph with the FLOAT32 MLX decoder.
    """
    import magenta_rt  # noqa: F401  (installs vendored sequence_layers path)
    import mlx.core as mx
    import sequence_layers.mlx as sl
    import soundfile as sf

    tokens_path = Path(args.tokens)
    tokens = _load_raw_tokens(tokens_path)
    models_dir = Path(args.models_dir)
    checkpoint_path = Path(args.checkpoint)
    codebooks_path = models_dir / "spectrostream_rvq_codebooks_12_f32.bin"
    codebooks = _load_codebooks(codebooks_path)
    soundstream = _build_mlx_spectrostream(checkpoint_path)
    library = _render_core_library(legacy_dual_window=args.legacy_dual_window_dsp)
    render_core = library.render_core_create(2, SAMPLE_RATE * 2)
    if not render_core:
        raise RuntimeError("render_core_create failed")
    output_path = Path(args.output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows = 0
    rendered_frames = 0
    decoder_stride = DECODER_STRIDE_FRAMES - args.decoder_context_frames
    crop_stft_frames = args.decoder_context_frames * 4
    started = time.perf_counter()
    try:
        with sf.SoundFile(
            output_path, mode="w", samplerate=SAMPLE_RATE, channels=2, subtype="FLOAT"
        ) as output:
            for start in range(
                0, tokens.shape[0] - DECODER_INPUT_FRAMES + 1, decoder_stride
            ):
                token_window = tokens[start : start + DECODER_INPUT_FRAMES]
                embeddings = _lookup_embeddings(codebooks, token_window)
                sequence = sl.Sequence(
                    mx.array(embeddings[np.newaxis], dtype=mx.float32),
                    mx.ones((1, DECODER_INPUT_FRAMES), dtype=mx.bool_),
                )
                prediction = soundstream.decoder.layer(sequence)
                mx.eval(prediction.values)
                stft = np.ascontiguousarray(prediction.values, dtype=np.float32)
                expected_shape = (1, DECODER_STFT_FRAMES, STFT_BINS, STFT_CHANNELS)
                if stft.shape != expected_shape:
                    raise ValueError(f"Unexpected decoder_stft shape {stft.shape}")
                if windows > 0 and crop_stft_frames:
                    stft = np.ascontiguousarray(
                        stft[:, crop_stft_frames:], dtype=np.float32
                    )
                rendered_stft_frames = int(stft.shape[1])
                capacity = rendered_stft_frames * STFT_HOP_SAMPLES
                pcm = np.empty((capacity, 2), dtype=np.float32)
                rendered = int(
                    library.render_core_render_decoder_stft(
                        render_core,
                        stft.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        rendered_stft_frames,
                        pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        capacity,
                        2,
                    )
                )
                if rendered != capacity:
                    raise RuntimeError(
                        f"RenderCore produced {rendered}/{capacity} frames"
                    )
                output.write(pcm)
                windows += 1
                rendered_frames += rendered
                if windows % 25 == 0:
                    print(f"mlx-stft/core-dsp window {windows}", flush=True)
    finally:
        library.render_core_destroy(render_core)
    report_path = output_path.with_suffix(".json")
    _write_json(
        report_path,
        {
            "schema": "mrt2-crossover-decode-v1",
            "tokenPath": str(tokens_path),
            "tokenSha256": _sha256(tokens_path),
            "decoder": "mlx-float32-pre-istft-plus-production-rendercore",
            "checkpointSha256": _sha256(checkpoint_path),
            "renderCoreSha256": _sha256(
                REPO_ROOT / "Sources" / "CrossfadeRuntimeCore" / "RenderCore.cpp"
            ),
            "dspWindow": (
                "legacy-dual-normalized-hann"
                if args.legacy_dual_window_dsp
                else "trained-periodic-hann"
            ),
            "codebooksSha256": _sha256(codebooks_path),
            "windows": windows,
            "decoderContextFrames": int(args.decoder_context_frames),
            "decoderStrideFrames": int(decoder_stride),
            "renderedFrames": rendered_frames,
            "durationSeconds": rendered_frames / SAMPLE_RATE,
            "elapsedSeconds": time.perf_counter() - started,
            "wav": {"path": str(output_path), "sha256": _sha256(output_path)},
        },
    )
    return report_path


def _build_mlx_spectrostream(checkpoint_path: Path):
    """Build only the MLX SpectroStream decoder."""
    import magenta_rt  # noqa: F401

    from magenta_rt.mlx import model
    from magenta_rt.mlx import spectrostream
    from magenta_rt.mlx.spectrostream.load_weights import load_spectrostream_weights

    experiment = model.get_model_class("mrt2_small")()
    config = spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
        rvq_truncation_level=experiment.spectrostream.rvq_truncation_level,
        use_unique_codes=False,
    )
    soundstream = config.make()
    load_spectrostream_weights(soundstream, checkpoint_path)
    return soundstream


def decode_mlx(args: argparse.Namespace) -> Path:
    """Decode either token source with the clean MLX streaming decoder."""
    import magenta_rt  # noqa: F401  (installs vendored sequence_layers import path)
    import mlx.core as mx
    import sequence_layers.mlx as sl
    import soundfile as sf

    tokens_path = Path(args.tokens)
    tokens = _load_raw_tokens(tokens_path)
    checkpoint_path = Path(args.checkpoint)
    codebooks_path = Path(args.models_dir) / "spectrostream_rvq_codebooks_12_f32.bin"
    codebooks = _load_codebooks(codebooks_path)
    embeddings = _lookup_embeddings(codebooks, tokens)
    soundstream = _build_mlx_spectrostream(checkpoint_path)
    layer = soundstream.embeddings_to_waveform_layer
    state = layer.get_initial_state(
        1, sl.ChannelSpec(shape=[DECODER_EMBEDDING_DIM], dtype=mx.float32)
    )
    output_path = Path(args.output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_frames = 0
    dropped_warmup_frames = 0
    started = time.perf_counter()
    with sf.SoundFile(
        output_path, mode="w", samplerate=SAMPLE_RATE, channels=2, subtype="FLOAT"
    ) as output:
        for index, embedding in enumerate(embeddings):
            sequence = sl.Sequence(
                mx.array(embedding.reshape(1, 1, -1), dtype=mx.float32),
                mx.ones((1, 1), dtype=mx.bool_),
            )
            decoded, state = layer.step(sequence, state)
            mx.eval(decoded.values, state)
            pcm = np.asarray(decoded.values, dtype=np.float32).reshape(-1, 2)
            if index == 0:
                dropped_warmup_frames = int(pcm.shape[0])
            else:
                output.write(pcm)
                rendered_frames += int(pcm.shape[0])
            if index % 625 == 0:
                print(f"mlx decoder frame {index}/{embeddings.shape[0]}", flush=True)
    report_path = output_path.with_suffix(".json")
    _write_json(
        report_path,
        {
            "schema": "mrt2-crossover-decode-v1",
            "tokenPath": str(tokens_path),
            "tokenSha256": _sha256(tokens_path),
            "decoder": "mlx-streaming-spectrostream",
            "checkpointSha256": _sha256(checkpoint_path),
            "codebooksSha256": _sha256(codebooks_path),
            "droppedWarmupFrames": dropped_warmup_frames,
            "renderedFrames": rendered_frames,
            "durationSeconds": rendered_frames / SAMPLE_RATE,
            "elapsedSeconds": time.perf_counter() - started,
            "wav": {"path": str(output_path), "sha256": _sha256(output_path)},
        },
    )
    return report_path


def _tensor_comparison(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    """Return finite, correlation, and absolute-error metrics for one probe arm."""
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"comparison shape mismatch {reference.shape} != {candidate.shape}"
        )
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not np.all(finite):
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(reference, candidate)[0, 1])
    error = np.abs(reference - candidate)
    return {
        "finiteRatio": float(np.mean(finite)),
        "correlation": correlation,
        "maxAbsoluteError": float(np.nanmax(error)),
        "meanAbsoluteError": float(np.nanmean(error)),
    }


def probe_decoder_context(args: argparse.Namespace) -> Path:
    """Measure how much left context recovers stateful MLX decoder output."""
    import magenta_rt  # noqa: F401
    import mlx.core as mx
    import sequence_layers.mlx as sl

    tokens_path = Path(args.tokens)
    tokens = _load_raw_tokens(tokens_path)
    checkpoint_path = Path(args.checkpoint)
    codebooks_path = Path(args.models_dir) / "spectrostream_rvq_codebooks_12_f32.bin"
    embeddings = _lookup_embeddings(_load_codebooks(codebooks_path), tokens)
    contexts = sorted({int(value) for value in args.probe_contexts.split(",")})
    if not contexts or contexts[0] < 0:
        raise ValueError("probe contexts must be non-negative")
    start = int(args.probe_start_frame)
    target_frames = DECODER_INPUT_FRAMES
    end = start + target_frames
    if start < max(contexts) or end > embeddings.shape[0]:
        raise ValueError(
            "probe segment and contexts must fit inside the token trajectory"
        )
    soundstream = _build_mlx_spectrostream(checkpoint_path)
    layer = soundstream.decoder.layer

    def predict(values: np.ndarray) -> np.ndarray:
        sequence = sl.Sequence(
            mx.array(values[np.newaxis], dtype=mx.float32),
            mx.ones((1, values.shape[0]), dtype=mx.bool_),
        )
        output = layer(sequence)
        mx.eval(output.values)
        return np.asarray(output.values, dtype=np.float32)

    full = predict(embeddings[:end])
    emitted_stft_frames = (target_frames - 1) * 4
    truth = full[:, start * 4 : start * 4 + emitted_stft_frames]
    arms: dict[str, dict[str, float | int]] = {}
    for context in contexts:
        candidate = predict(embeddings[start - context : end])
        candidate = candidate[:, context * 4 : context * 4 + emitted_stft_frames]
        arms[str(context)] = {
            "contextTokenFrames": context,
            **_tensor_comparison(truth, candidate),
        }
    output_path = Path(args.output_json)
    _write_json(
        output_path,
        {
            "schema": "mrt2-decoder-context-probe-v1",
            "tokenPath": str(tokens_path),
            "tokenSha256": _sha256(tokens_path),
            "checkpointSha256": _sha256(checkpoint_path),
            "codebooksSha256": _sha256(codebooks_path),
            "startTokenFrame": start,
            "targetTokenFrames": target_frames,
            "targetSTFTFrames": emitted_stft_frames,
            "arms": arms,
        },
    )
    return output_path


def summarize_command(args: argparse.Namespace) -> Path:
    """Write token summary JSON for an existing token file."""
    tokens_path = Path(args.tokens)
    output_path = Path(args.output_json)
    _write_json(
        output_path,
        summarize_tokens(
            _load_raw_tokens(tokens_path), window_seconds=args.window_seconds
        ),
    )
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "generate-mlx",
            "generate-coreml",
            "decode-mlx",
            "decode-coreml",
            "decode-mlx-stft-core-dsp",
            "probe-decoder-context",
            "summarize",
        ),
    )
    parser.add_argument("--output-dir", default="Scratchpad/system_paper_crossover")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument(
        "--reference-models-dir", default=str(DEFAULT_REFERENCE_MODELS_DIR)
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--source-conditioning", default=str(DEFAULT_SOURCE_CONDITIONING)
    )
    parser.add_argument("--prompt", default="warm ambient texture")
    parser.add_argument("--seconds", type=int, default=DEFAULT_SECONDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--refresh-seconds", type=float, default=DEFAULT_REFRESH_SECONDS
    )
    parser.add_argument(
        "--reset-policy",
        choices=RESET_POLICIES,
        default=RESET_POLICY_AUTO,
        help="State reset at each refresh boundary; auto preserves legacy both/off behavior.",
    )
    parser.add_argument("--tokens")
    parser.add_argument("--output-wav")
    parser.add_argument("--output-json")
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--probe-start-frame", type=int, default=150)
    parser.add_argument("--probe-contexts", default="0,1,2,4,8,12")
    parser.add_argument("--legacy-dual-window-dsp", action="store_true")
    parser.add_argument("--decoder-context-frames", type=int, default=0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k != DEFAULT_TOP_K:
        parser.error("the shipped graph has top-k 40 baked in")
    if args.refresh_seconds < 0:
        parser.error("--refresh-seconds must be nonnegative")
    try:
        _resolve_reset_policy(args.refresh_seconds, args.reset_policy)
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.decoder_context_frames < DECODER_STRIDE_FRAMES:
        parser.error(
            f"--decoder-context-frames must be in 0..{DECODER_STRIDE_FRAMES - 1}"
        )
    if args.command.startswith("decode-") and (not args.tokens or not args.output_wav):
        parser.error("decode commands require --tokens and --output-wav")
    if args.command in {"summarize", "probe-decoder-context"} and (
        not args.tokens or not args.output_json
    ):
        parser.error(f"{args.command} requires --tokens and --output-json")
    return args


def main() -> None:
    """Run the selected crossover operation."""
    args = parse_args()
    operations = {
        "generate-mlx": generate_mlx_tokens,
        "generate-coreml": generate_coreml_tokens,
        "decode-mlx": decode_mlx,
        "decode-coreml": decode_coreml,
        "decode-mlx-stft-core-dsp": decode_mlx_stft_core_dsp,
        "probe-decoder-context": probe_decoder_context,
        "summarize": summarize_command,
    }
    result = operations[args.command](args)
    print(result)


if __name__ == "__main__":
    main()
