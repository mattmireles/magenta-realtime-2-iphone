#!/usr/bin/env python3
"""Analyze one frozen refreshed/unrefreshed MRT2 liveness pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


SAMPLE_RATE = 48_000
TOKEN_RATE = 25
RVQ_LEVELS = 12
CODEBOOK_SIZE = 1_024
WINDOWS = (("early", 60, 90), ("middle", 285, 315), ("late", 570, 600))
EXPECTED_DECODER_CONTEXT_FRAMES = 12
EXPECTED_DECODER = "mlx-float32-pre-istft-plus-production-rendercore"
EXPECTED_DECODER_STRIDE_FRAMES = 12
EXPECTED_DSP_WINDOW = "trained-periodic-hann"
EXPECTED_CODEBOOKS_SHA256 = (
    "4e236269d4194ffe2d7463c483a1a36f4aff7d619c34f8e8bfe451c7af92d496"
)
EXPECTED_RENDER_CORE_SHA256 = (
    "a0525a65d70c68c51505f58de7f61cbfeef01454093336d75528c8e1ff2f192f"
)
FULL_SCAN_BLOCK_FRAMES = 30 * SAMPLE_RATE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _protocol_windows(protocol: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    rows = protocol.get("analysisWindowsSeconds")
    if not isinstance(rows, list):
        raise ValueError("protocol is missing analysisWindowsSeconds")
    try:
        windows = tuple(
            (str(row["id"]), int(row["start"]), int(row["end"])) for row in rows
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("protocol analysis windows are malformed") from error
    if windows != WINDOWS:
        raise ValueError(f"protocol analysis windows drifted: {windows!r}")
    return windows


def _protocol_fixture(protocol: dict[str, Any], prompt_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in protocol.get("fixtures", [])
        if isinstance(row, dict) and row.get("id") == prompt_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one protocol fixture for promptId={prompt_id}"
        )
    return matches[0]


def _protocol_arm(
    protocol: dict[str, Any], *, prompt_id: str, seed: int, mode: str
) -> dict[str, Any]:
    matches = [
        row
        for row in protocol.get("arms", [])
        if isinstance(row, dict)
        and row.get("promptId") == prompt_id
        and row.get("seed") == seed
        and row.get("mode") == mode
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one protocol arm for promptId={prompt_id} seed={seed} mode={mode}"
        )
    return matches[0]


def _source_record(manifest: dict[str, Any]) -> dict[str, Any]:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(
        inputs.get("sourceConditioning"), dict
    ):
        raise ValueError("run manifest is missing inputs.sourceConditioning")
    return inputs["sourceConditioning"]


def _validate_source_hash(
    manifest: dict[str, Any], *, expected_sha256: str, label: str
) -> None:
    source = _source_record(manifest)
    if source.get("sha256") != expected_sha256:
        raise ValueError(f"{label} source-conditioning hash is not the frozen fixture")
    source_path = Path(str(source.get("path", "")))
    if not source_path.is_file():
        raise ValueError(f"{label} source-conditioning file is missing: {source_path}")
    if _sha256(source_path) != expected_sha256:
        raise ValueError(f"{label} source-conditioning bytes do not match their hash")


def _validate_checkpoint_hash(
    protocol: dict[str, Any], manifest: dict[str, Any], *, label: str
) -> None:
    expected = protocol.get("checkpoint", {}).get("sha256")
    checkpoint = manifest.get("inputs", {}).get("checkpoint")
    if not isinstance(expected, str) or not isinstance(checkpoint, dict):
        raise ValueError(f"{label} checkpoint provenance is missing")
    if checkpoint.get("sha256") != expected:
        raise ValueError(f"{label} checkpoint hash is not the frozen checkpoint")


def _validate_decode_receipt(
    wav_path: Path,
    *,
    token_manifest: dict[str, Any],
    expected_frames: int,
    label: str,
) -> dict[str, Any]:
    receipt_path = wav_path.with_suffix(".json")
    if not receipt_path.is_file():
        raise ValueError(f"{label} decode receipt is missing: {receipt_path}")
    receipt = _load_json(receipt_path)
    if receipt.get("schema") != "mrt2-crossover-decode-v1":
        raise ValueError(f"{label} decode receipt schema is unsupported")
    if receipt.get("decoderContextFrames") != EXPECTED_DECODER_CONTEXT_FRAMES:
        raise ValueError(f"{label} WAV is not bound to the context-12 decoder")
    if receipt.get("decoder") != EXPECTED_DECODER:
        raise ValueError(f"{label} decode receipt has the wrong decoder implementation")
    if receipt.get("decoderStrideFrames") != EXPECTED_DECODER_STRIDE_FRAMES:
        raise ValueError(f"{label} decode receipt has the wrong decoder stride")
    if receipt.get("dspWindow") != EXPECTED_DSP_WINDOW:
        raise ValueError(f"{label} decode receipt has the wrong DSP window")
    if receipt.get("codebooksSha256") != EXPECTED_CODEBOOKS_SHA256:
        raise ValueError(f"{label} decode receipt has the wrong codebooks")
    if receipt.get("renderCoreSha256") != EXPECTED_RENDER_CORE_SHA256:
        raise ValueError(f"{label} decode receipt has the wrong RenderCore")
    if receipt.get("renderedFrames") != expected_frames:
        raise ValueError(f"{label} decode receipt has the wrong rendered frame count")
    if receipt.get("durationSeconds") != expected_frames / SAMPLE_RATE:
        raise ValueError(f"{label} decode receipt has the wrong duration")
    if receipt.get("tokenSha256") != token_manifest["tokens"]["sha256"]:
        raise ValueError(f"{label} decode receipt is bound to different tokens")
    token_path = Path(str(token_manifest["tokens"].get("path", "")))
    receipt_token_path = Path(str(receipt.get("tokenPath", "")))
    if receipt_token_path.resolve() != token_path.resolve():
        raise ValueError(f"{label} decode receipt has the wrong token path")
    expected_windows = (
        int(token_manifest["tokenFrames"]) - 25
    ) // EXPECTED_DECODER_STRIDE_FRAMES + 1
    if receipt.get("windows") != expected_windows:
        raise ValueError(f"{label} decode receipt has the wrong window count")
    wav = receipt.get("wav")
    if (
        not isinstance(wav, dict)
        or Path(str(wav.get("path", ""))).resolve() != wav_path.resolve()
        or wav.get("sha256") != _sha256(wav_path)
    ):
        raise ValueError(f"{label} decode receipt WAV hash mismatch")
    return {
        "path": str(receipt_path),
        "sha256": _sha256(receipt_path),
        "decoderContextFrames": receipt["decoderContextFrames"],
        "renderedFrames": receipt["renderedFrames"],
    }


def _entropy(values: np.ndarray) -> float:
    counts = np.bincount(values, minlength=CODEBOOK_SIZE)
    probabilities = counts[counts > 0] / values.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def _token_metrics(tokens: np.ndarray) -> dict[str, Any]:
    if tokens.ndim != 2 or tokens.shape[1] != RVQ_LEVELS:
        raise ValueError(f"expected [frames,{RVQ_LEVELS}] tokens, got {tokens.shape}")
    if np.any(tokens < 0) or np.any(tokens >= CODEBOOK_SIZE):
        raise ValueError("tokens must be codebook-local 0..1023")
    lag_matches = {
        str(lag): float(np.mean(np.all(tokens[lag:] == tokens[:-lag], axis=1)))
        for lag in range(1, 9)
    }
    return {
        "frames": int(tokens.shape[0]),
        "finite": True,
        "entropyBitsByRVQLevel": [
            _entropy(tokens[:, level]) for level in range(RVQ_LEVELS)
        ],
        "exactFrameMatchFractionByLag": lag_matches,
        "exactCycleLags": [lag for lag, value in lag_matches.items() if value == 1.0],
        "distinctFrameRatio": float(
            np.unique(tokens, axis=0).shape[0] / tokens.shape[0]
        ),
    }


def _envelope_pulse_share(audio: np.ndarray, sample_rate: int) -> float:
    mono = audio.mean(axis=1)
    block = max(1, int(round(sample_rate * 0.010)))
    usable = mono[: mono.size - mono.size % block]
    envelope = np.sqrt(np.mean(usable.reshape(-1, block) ** 2, axis=1))
    envelope -= np.mean(envelope)
    power = np.abs(np.fft.rfft(envelope)) ** 2
    frequencies = np.fft.rfftfreq(envelope.size, d=block / sample_rate)
    positive = frequencies > 0
    denominator = float(power[positive].sum())
    if denominator == 0:
        return 0.0
    return float(power[(frequencies >= 4) & (frequencies <= 16)].sum() / denominator)


def _spectral_flatness(audio: np.ndarray) -> float:
    mono = audio.mean(axis=1)
    fft_size = 2048
    hop = 1024
    if mono.size < fft_size:
        return 0.0
    frames = np.lib.stride_tricks.sliding_window_view(mono, fft_size)[::hop]
    magnitudes = (
        np.abs(np.fft.rfft(frames * np.hanning(fft_size), axis=1))[:, 1:] + 1e-12
    )
    values = np.exp(np.mean(np.log(magnitudes), axis=1)) / np.mean(magnitudes, axis=1)
    return float(np.mean(values))


def _audio_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    finite = np.isfinite(audio)
    residual = audio[:, 0] - audio[:, 1]
    overall_rms = float(np.sqrt(np.mean(audio * audio)))
    residual_rms = float(np.sqrt(np.mean(residual * residual)))
    left_std = float(np.std(audio[:, 0]))
    right_std = float(np.std(audio[:, 1]))
    correlation = None
    if left_std > 0 and right_std > 0:
        correlation = float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1])
    return {
        "frames": int(audio.shape[0]),
        "seconds": float(audio.shape[0] / sample_rate),
        "finiteRatio": float(np.mean(finite)),
        "rms": overall_rms,
        "peak": float(np.max(np.abs(audio))),
        "clippedSampleRatio": float(np.mean(np.abs(audio) >= 1.0)),
        "spectralFlatness": _spectral_flatness(audio),
        "leftRightCorrelation": correlation,
        "interchannelResidualRMSRatio": residual_rms / max(overall_rms, 1e-12),
        "envelopePulseShare4To16Hz": _envelope_pulse_share(audio, sample_rate),
        "pcmFloat32Sha256": hashlib.sha256(
            np.ascontiguousarray(audio, dtype="<f4").tobytes()
        ).hexdigest(),
    }


def _scan_full_capture(
    path: Path, *, expected_frames: int
) -> tuple[dict[str, Any], list[str]]:
    """Scan every decoded sample without retaining a 600-second WAV in memory."""
    total_samples = 0
    finite_samples = 0
    square_sum = 0.0
    residual_square_sum = 0.0
    peak = 0.0
    clipped_samples = 0
    exact_channel_equality = True
    short_reads = 0
    window_hashes: list[str] = []
    with sf.SoundFile(path) as handle:
        if handle.samplerate != SAMPLE_RATE or handle.channels != 2:
            raise ValueError(f"expected 48 kHz stereo WAV: {path}")
        declared_frames = int(handle.frames)
        while True:
            audio = handle.read(FULL_SCAN_BLOCK_FRAMES, dtype="float32", always_2d=True)
            if audio.shape[0] == 0:
                break
            if (
                audio.shape[0] < FULL_SCAN_BLOCK_FRAMES
                and total_samples + audio.size < declared_frames * 2
            ):
                short_reads += 1
            finite = np.isfinite(audio)
            finite_samples += int(np.count_nonzero(finite))
            total_samples += int(audio.size)
            safe = np.where(finite, audio, 0.0).astype(np.float64, copy=False)
            square_sum += float(np.sum(safe * safe, dtype=np.float64))
            residual = safe[:, 0] - safe[:, 1]
            residual_square_sum += float(np.sum(residual * residual, dtype=np.float64))
            peak = max(peak, float(np.max(np.abs(safe), initial=0.0)))
            clipped_samples += int(np.count_nonzero(np.abs(safe) >= 1.0))
            exact_channel_equality = exact_channel_equality and bool(
                np.array_equal(audio[:, 0], audio[:, 1])
            )
            if audio.shape[0] == FULL_SCAN_BLOCK_FRAMES:
                window_hashes.append(
                    hashlib.sha256(
                        np.ascontiguousarray(audio, dtype="<f4").tobytes()
                    ).hexdigest()
                )
        read_frames = total_samples // 2
    rms = float(np.sqrt(square_sum / max(total_samples, 1)))
    residual_rms = float(np.sqrt(residual_square_sum / max(read_frames, 1)))
    exact_duration = (
        declared_frames == expected_frames and read_frames == expected_frames
    )
    report = {
        "declaredFrames": declared_frames,
        "readFrames": read_frames,
        "expectedFrames": int(expected_frames),
        "seconds": float(read_frames / SAMPLE_RATE),
        "finiteRatio": float(finite_samples / max(total_samples, 1)),
        "nonFiniteSamples": int(total_samples - finite_samples),
        "rms": rms,
        "peak": peak,
        "clippedSampleRatio": float(clipped_samples / max(total_samples, 1)),
        "interchannelResidualRMSRatio": residual_rms / max(rms, 1e-12),
        "exactChannelEquality": exact_channel_equality,
        "unexpectedShortReadCount": short_reads,
        "exactExpectedDuration": exact_duration,
        "captureDiscontinuityDetected": bool(short_reads or not exact_duration),
        "completeThirtySecondWindowCount": len(window_hashes),
    }
    return report, window_hashes


def _load_audio_windows(
    path: Path, *, windows: tuple[tuple[str, int, int], ...] = WINDOWS
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    with sf.SoundFile(path) as handle:
        if handle.samplerate != SAMPLE_RATE or handle.channels != 2:
            raise ValueError(f"expected 48 kHz stereo WAV: {path}")
        for name, start, end in windows:
            handle.seek(start * SAMPLE_RATE)
            audio = handle.read(
                (end - start) * SAMPLE_RATE, dtype="float32", always_2d=True
            )
            if audio.shape[0] != (end - start) * SAMPLE_RATE:
                raise ValueError(f"{path} is too short for {name} [{start},{end})")
            reports[name] = _audio_metrics(audio, SAMPLE_RATE)
    return reports


def _plot_windows(
    paths: dict[str, Path],
    output_path: Path,
    *,
    windows: tuple[tuple[str, int, int], ...] = WINDOWS,
) -> None:
    figure, axes = plt.subplots(
        len(windows), len(paths), figsize=(14, 10), squeeze=False
    )
    for column, (mode, path) in enumerate(paths.items()):
        with sf.SoundFile(path) as handle:
            for row, (name, start, end) in enumerate(windows):
                handle.seek(start * SAMPLE_RATE)
                audio = handle.read(
                    (end - start) * SAMPLE_RATE, dtype="float32", always_2d=True
                )
                axes[row, column].specgram(
                    audio.mean(axis=1),
                    NFFT=2048,
                    Fs=SAMPLE_RATE,
                    noverlap=1536,
                    cmap="magma",
                )
                axes[row, column].set_title(f"{mode} {name} [{start},{end})")
                axes[row, column].set_ylim(0, 16_000)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _load_run(manifest_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "mrt2-long-horizon-token-run-v1":
        raise ValueError(f"unsupported token-run schema: {manifest.get('schema')!r}")
    token_path = Path(manifest["tokens"]["path"])
    if not token_path.is_file():
        raise ValueError(f"token file is missing: {token_path}")
    if _sha256(token_path) != manifest["tokens"]["sha256"]:
        raise ValueError(f"token hash mismatch: {token_path}")
    raw_tokens = np.load(token_path, allow_pickle=False)
    if not np.issubdtype(raw_tokens.dtype, np.number):
        raise ValueError("tokens must use a numeric dtype")
    if not bool(np.all(np.isfinite(raw_tokens))):
        raise ValueError("token capture contains non-finite values")
    if not bool(np.all(raw_tokens == np.floor(raw_tokens))):
        raise ValueError("token capture contains non-integral values")
    tokens = np.asarray(raw_tokens, dtype=np.int32)
    if manifest.get("tokenFrames") != int(tokens.shape[0]):
        raise ValueError("manifest tokenFrames does not match token file")
    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("token-run manifest is missing summary provenance")
    summary_path = Path(str(summary.get("path", "")))
    if not summary_path.is_file() or _sha256(summary_path) != summary.get("sha256"):
        raise ValueError("token summary file/hash mismatch")
    return manifest, tokens


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_json(args.protocol)
    if protocol.get("schema") != "mrt2-liveness-protocol-v1":
        raise ValueError(
            f"unsupported liveness protocol schema: {protocol.get('schema')!r}"
        )
    windows = _protocol_windows(protocol)
    generation = protocol.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("protocol is missing generation")
    expected_seconds = int(generation.get("audibleSeconds", -1))
    expected_token_frames = int(
        generation.get("expectedTokenFramesIncludingLookahead", -1)
    )
    expected_audio_frames = expected_seconds * SAMPLE_RATE
    if expected_seconds != 600 or expected_token_frames != 15_001:
        raise ValueError(
            "protocol generation horizon drifted from the frozen 600-second contract"
        )
    if generation.get("tokenRateHz") != TOKEN_RATE:
        raise ValueError("protocol token rate drifted")
    if generation.get("decoderContextFrames") != EXPECTED_DECODER_CONTEXT_FRAMES:
        raise ValueError("protocol decoder context is not the frozen context-12 path")

    refreshed_manifest, refreshed_tokens = _load_run(args.refreshed_manifest)
    candidate_manifest, candidate_tokens = _load_run(args.unrefreshed_manifest)
    invariant_fields = (
        "tokenSource",
        "seed",
        "prompt",
        "temperature",
        "topK",
        "requestedAudibleSeconds",
        "tokenFrames",
        "inputs",
    )
    for field in invariant_fields:
        if refreshed_manifest[field] != candidate_manifest[field]:
            raise ValueError(f"pair mismatch at {field}")
    if refreshed_manifest["refreshMode"] != "periodic":
        raise ValueError("refreshed arm is not periodic")
    if (
        candidate_manifest["refreshMode"] != "off"
        or candidate_manifest["temporalResetCount"] != 0
    ):
        raise ValueError("candidate is not explicit zero-reset unrefreshed mode")
    if (
        refreshed_tokens.shape[0] != expected_token_frames
        or candidate_tokens.shape[0] != expected_token_frames
    ):
        raise ValueError(f"token streams must contain {expected_token_frames} frames")
    if refreshed_manifest["requestedAudibleSeconds"] != expected_seconds:
        raise ValueError("run duration is not the frozen protocol horizon")
    if refreshed_manifest["temperature"] != generation.get("temperature"):
        raise ValueError("generation temperature is not frozen protocol value")
    if refreshed_manifest["topK"] != generation.get("topK"):
        raise ValueError("generation top-k is not frozen protocol value")

    source_sha256 = str(_source_record(refreshed_manifest).get("sha256", ""))
    fixture_matches = [
        row
        for row in protocol.get("fixtures", [])
        if isinstance(row, dict)
        and row.get("prompt") == refreshed_manifest["prompt"]
        and row.get("sourceConditioning", {}).get("sha256") == source_sha256
    ]
    if len(fixture_matches) != 1:
        raise ValueError("pair does not match exactly one frozen prompt fixture")
    prompt_id = str(fixture_matches[0]["id"])
    fixture = _protocol_fixture(protocol, prompt_id)
    seed = int(refreshed_manifest["seed"])
    refreshed_arm = _protocol_arm(
        protocol, prompt_id=prompt_id, seed=seed, mode="refresh10"
    )
    candidate_arm = _protocol_arm(
        protocol, prompt_id=prompt_id, seed=seed, mode="unrefreshed"
    )
    fixture_sha256 = str(fixture["sourceConditioning"]["sha256"])
    for label, manifest, arm in (
        ("refreshed", refreshed_manifest, refreshed_arm),
        ("unrefreshed", candidate_manifest, candidate_arm),
    ):
        if arm.get("sourceConditioningSha256") != fixture_sha256:
            raise ValueError(f"{label} protocol arm has a fixture-hash mismatch")
        _validate_source_hash(manifest, expected_sha256=fixture_sha256, label=label)
        _validate_checkpoint_hash(protocol, manifest, label=label)
    if refreshed_arm.get("refreshSeconds") != refreshed_manifest.get(
        "trajectoryRefreshSeconds"
    ):
        raise ValueError("refreshed arm interval does not match protocol")
    if candidate_arm.get("refreshSeconds") != candidate_manifest.get(
        "trajectoryRefreshSeconds"
    ):
        raise ValueError("unrefreshed arm interval does not match protocol")
    refresh_frames = next(
        (
            row.get("refreshFrames")
            for row in protocol.get("refreshModes", [])
            if isinstance(row, dict) and row.get("id") == "refresh10"
        ),
        None,
    )
    if not isinstance(refresh_frames, int) or refresh_frames <= 0:
        raise ValueError("protocol refresh10 interval is missing")
    expected_reset_count = (expected_token_frames - 1) // refresh_frames
    if refreshed_manifest.get("temporalResetCount") != expected_reset_count:
        raise ValueError("refreshed reset count does not match protocol horizon")

    token_reports = {}
    for mode, tokens in (
        ("refreshed", refreshed_tokens),
        ("unrefreshed", candidate_tokens),
    ):
        token_reports[mode] = {
            name: _token_metrics(tokens[start * TOKEN_RATE : end * TOKEN_RATE])
            for name, start, end in windows
        }
    audio_reports = {}
    full_capture_reports = {}
    decode_receipts = {}
    repeated = {}
    for mode, path, manifest in (
        ("refreshed", args.refreshed_wav, refreshed_manifest),
        ("unrefreshed", args.unrefreshed_wav, candidate_manifest),
    ):
        decode_receipts[mode] = _validate_decode_receipt(
            path,
            token_manifest=manifest,
            expected_frames=expected_audio_frames,
            label=mode,
        )
        full_capture_reports[mode], hashes = _scan_full_capture(
            path, expected_frames=expected_audio_frames
        )
        audio_reports[mode] = _load_audio_windows(path, windows=windows)
        repeated[mode] = len(hashes) != len(set(hashes))

    paired = {}
    for name, _, _ in windows:
        paired[name] = {
            metric: audio_reports["unrefreshed"][name][metric]
            - audio_reports["refreshed"][name][metric]
            for metric in (
                "rms",
                "peak",
                "clippedSampleRatio",
                "spectralFlatness",
                "interchannelResidualRMSRatio",
                "envelopePulseShare4To16Hz",
            )
        }
    refreshed_full = full_capture_reports["refreshed"]
    candidate_full = full_capture_reports["unrefreshed"]
    channel_collapse = bool(
        not refreshed_full["exactChannelEquality"]
        and candidate_full["exactChannelEquality"]
    )
    channel_collapse_comparison = {
        "criterion": (
            "candidate channels are sample-identical while matched refreshed channels are not"
        ),
        "refreshedExactChannelEquality": refreshed_full["exactChannelEquality"],
        "unrefreshedExactChannelEquality": candidate_full["exactChannelEquality"],
        "refreshedInterchannelResidualRMSRatio": refreshed_full[
            "interchannelResidualRMSRatio"
        ],
        "unrefreshedInterchannelResidualRMSRatio": candidate_full[
            "interchannelResidualRMSRatio"
        ],
        "collapsed": channel_collapse,
    }
    catastrophic = {
        "unrefreshedNonFiniteToken": False,
        "unrefreshedNonFinite": any(
            row["finiteRatio"] != 1.0 for row in audio_reports["unrefreshed"].values()
        )
        or candidate_full["finiteRatio"] != 1.0,
        "unrefreshedDurationMismatch": not candidate_full["exactExpectedDuration"],
        "unrefreshedShorterThan600Seconds": candidate_full["readFrames"]
        < expected_audio_frames,
        "unrefreshedCaptureDiscontinuity": candidate_full[
            "captureDiscontinuityDetected"
        ],
        "unrefreshedExactRepeated30SecondWindow": repeated["unrefreshed"],
        "unrefreshedIntroducesClipping": candidate_full["clippedSampleRatio"]
        > refreshed_full["clippedSampleRatio"],
        "unrefreshedChannelCollapseRelativeToRefreshed": channel_collapse,
        "unrefreshedExactTokenCycle": any(
            token_reports["unrefreshed"][name]["exactCycleLags"]
            for name, _, _ in windows
        ),
    }
    catastrophic_failures = sorted(key for key, value in catastrophic.items() if value)
    reference_failures = {
        "refreshedNonFinite": refreshed_full["finiteRatio"] != 1.0,
        "refreshedDurationMismatch": not refreshed_full["exactExpectedDuration"],
        "refreshedCaptureDiscontinuity": refreshed_full["captureDiscontinuityDetected"],
    }
    reference_failure_ids = sorted(
        key for key, value in reference_failures.items() if value
    )
    all_failure_ids = reference_failure_ids + catastrophic_failures
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectrogram = args.output_dir / "pilot-windows-spectrogram.png"
    _plot_windows(
        {"refreshed": args.refreshed_wav, "unrefreshed": args.unrefreshed_wav},
        spectrogram,
        windows=windows,
    )
    return {
        "schema": "mrt2-liveness-pair-analysis-v1",
        "protocolSha256": _sha256(args.protocol),
        "promptId": prompt_id,
        "seed": seed,
        "armIds": {
            "refreshed": refreshed_arm["armId"],
            "unrefreshed": candidate_arm["armId"],
        },
        "pairInvariants": {
            field: refreshed_manifest[field] for field in invariant_fields
        },
        "manifests": {
            "refreshed": {
                "path": str(args.refreshed_manifest),
                "sha256": _sha256(args.refreshed_manifest),
            },
            "unrefreshed": {
                "path": str(args.unrefreshed_manifest),
                "sha256": _sha256(args.unrefreshed_manifest),
            },
        },
        "wavs": {
            "refreshed": {
                "path": str(args.refreshed_wav),
                "sha256": _sha256(args.refreshed_wav),
            },
            "unrefreshed": {
                "path": str(args.unrefreshed_wav),
                "sha256": _sha256(args.unrefreshed_wav),
            },
        },
        "decodeReceipts": decode_receipts,
        "tokenWindows": token_reports,
        "fullCapture": full_capture_reports,
        "channelCollapseComparison": channel_collapse_comparison,
        "audioWindows": audio_reports,
        "pairedUnrefreshedMinusRefreshed": paired,
        "catastrophicIntegrityFailures": catastrophic,
        "refreshedReferenceIntegrityFailures": reference_failures,
        "catastrophicIntegrityStatus": {
            "verdict": "fail" if all_failure_ids else "pass",
            "passed": not all_failure_ids,
            "failureIds": all_failure_ids,
            "candidateFailureIds": catastrophic_failures,
            "referenceFailureIds": reference_failure_ids,
            "allChecksAssessed": True,
        },
        "spectrogram": {"path": str(spectrogram), "sha256": _sha256(spectrogram)},
        "promptAdherence": {"status": "unavailable-in-pilot", "universalGate": False},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--refreshed-manifest", type=Path, required=True)
    parser.add_argument("--unrefreshed-manifest", type=Path, required=True)
    parser.add_argument("--refreshed-wav", type=Path, required=True)
    parser.add_argument("--unrefreshed-wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(args)
    output = args.output_dir / "pair-analysis.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
