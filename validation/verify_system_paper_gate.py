#!/usr/bin/env python3
"""Verify the frozen MRT2 system-paper evidence gates.

The private Crossfade runtime produces raw logs and device traces. This public
verifier consumes small JSON manifests that name and hash those artifacts, then
applies the paper's predeclared pass/fail rules. It deliberately contains no
device-control code and makes no inference from requested Core ML compute units.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable


G1_SCHEMA = "mrt2-system-paper-g1-v2"
G2_SCHEMA = "mrt2-system-paper-g2-v1"
G3_SCHEMA = "mrt2-system-paper-g3-v1"
G4_SCHEMA = "mrt2-system-paper-g4-v1"
REPORT_SCHEMA = "mrt2-system-paper-gate-report-v1"

A17_PRO_DEVICE = "iPhone16,2"
A14_DEVICE = "iPhone13,3"
REQUIRED_MODEL_FAMILIES = ("temporal", "depth", "decoder")
REQUIRED_ANE_MODEL_FAMILIES = ("temporal", "decoder")
REQUIRED_KNOWN_BAD_CONTROLS = (
    "stride-corruption",
    "missing-temporal-feedback",
    "write-only-state",
    "click-comb",
    "channel-collapse",
    "dropout-injection",
)
EXPECTED_EFFECTIVE_FRAME_DEFINITION = "temporal+depth+sampling+decoder/decoded_frames"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MIN_COLD_LAUNCHES = 10
MIN_SOAK_SECONDS = 600.0
MIN_STREAMING_FRAMES = 42
MIN_EFFECTIVE_FRAME_COUNT = 15_000
MAX_EFFECTIVE_FRAME_P99_MS = 40.0
MIN_STATE_CORRELATION = 0.999
MAX_STATE_ABSOLUTE_ERROR = 2.5
MIN_STATE_DIVERGENCE = 1e-4

G3_FIXED_PROMPT = "warm ambient texture"
G3_MAX_TEMPERATURE = 1.1
G3_TOP_K = 40
G3_MIN_DURATION_SECONDS = 600.0
G3_MAX_CLIPPED_SAMPLE_RATIO = 1e-5
G3_MAX_BOUNDARY_ABS_JUMP = 0.07
G3_MIN_LEFT_RIGHT_CORRELATION = 0.97
G3_MIN_PROMPT_ADHERENCE = 0.30
G3_MIN_EMBEDDING_SIMILARITY = 0.80
G3_MAX_ENVELOPE_PULSE_SHARE = 0.07
G3_REQUIRED_VALID_VOTES = 5
G3_REQUIRED_CANDIDATE_PASSES = 4


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _is_finite_number(value: object) -> bool:
    """Return whether ``value`` is a finite, non-boolean number."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is a lowercase hexadecimal SHA-256 digest."""
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    value: object = None,
    expected: object = None,
) -> None:
    """Append one serializable check row."""
    row: dict[str, Any] = {"name": name, "passed": bool(passed), "value": value}
    if expected is not None:
        row["expected"] = expected
    checks.append(row)


def _artifact_hashes_present(value: object) -> bool:
    """Return whether a path-to-SHA256 mapping is nonempty and valid."""
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(path, str) and path and _is_sha256(digest)
            for path, digest in value.items()
        )
    )


def _thermal_timeline_covers(value: object, seconds: float) -> bool:
    """Return whether a thermal timeline spans the requested interval."""
    if not isinstance(value, list) or len(value) < 2:
        return False
    elapsed = [
        item.get("elapsedSeconds")
        for item in value
        if isinstance(item, dict) and _is_finite_number(item.get("elapsedSeconds"))
    ]
    states_valid = all(
        isinstance(item, dict)
        and item.get("state") in {"nominal", "fair", "serious", "critical"}
        for item in value
    )
    return (
        states_valid
        and bool(elapsed)
        and min(elapsed) <= 1.0
        and max(elapsed) >= seconds
    )


def verify_g1(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify 10-launch placement reliability and temporal-state correctness."""
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "schema",
        manifest.get("schema") == G1_SCHEMA,
        value=manifest.get("schema"),
        expected=G1_SCHEMA,
    )
    device_identifier = manifest.get("device", {}).get("modelIdentifier")
    device_role = manifest.get("deviceRole")
    expected_device_by_role = {
        "primary": A17_PRO_DEVICE,
        "cross-device": A14_DEVICE,
    }
    _check(
        checks,
        "device_role",
        device_role in expected_device_by_role
        and device_identifier == expected_device_by_role[device_role],
        value={"role": device_role, "modelIdentifier": device_identifier},
        expected=expected_device_by_role,
    )

    launches = manifest.get("launches")
    launches = launches if isinstance(launches, list) else []
    _check(
        checks,
        "cold_launch_count",
        len(launches) >= MIN_COLD_LAUNCHES,
        value=len(launches),
        expected=f">={MIN_COLD_LAUNCHES}",
    )
    _check(
        checks,
        "unique_run_ids",
        len({item.get("runId") for item in launches if isinstance(item, dict)})
        == len(launches),
        value=[item.get("runId") for item in launches if isinstance(item, dict)],
    )
    _check(
        checks,
        "all_launches_cold",
        bool(launches)
        and all(
            item.get("coldLaunch") is True
            for item in launches
            if isinstance(item, dict)
        ),
        value=[item.get("coldLaunch") for item in launches if isinstance(item, dict)],
    )

    launch_hashes: list[dict[str, Any]] = []
    launch_policies: list[dict[str, Any]] = []
    all_admission_pass = bool(launches)
    all_artifacts_hashed = bool(launches)
    for launch in launches:
        if not isinstance(launch, dict):
            all_admission_pass = False
            all_artifacts_hashed = False
            continue
        hashes = launch.get("modelSha256")
        policy = launch.get("computePolicy")
        launch_hashes.append(hashes if isinstance(hashes, dict) else {})
        launch_policies.append(policy if isinstance(policy, dict) else {})
        all_artifacts_hashed &= (
            isinstance(hashes, dict)
            and set(hashes) == set(REQUIRED_MODEL_FAMILIES)
            and all(_is_sha256(hashes.get(name)) for name in REQUIRED_MODEL_FAMILIES)
            and _artifact_hashes_present(launch.get("artifactSha256"))
        )
        plan = launch.get("temporalComputePlan")
        plan_pass = (
            isinstance(plan, dict)
            and plan.get("passed") is True
            and _is_finite_number(plan.get("aneEstimatedCostWeight"))
            and plan["aneEstimatedCostWeight"] >= 0.95
            and plan.get("gpuOperationCount") == 0
            and plan.get("gpuEstimatedCostWeight") == 0
        )
        all_admission_pass &= (
            plan_pass
            and launch.get("stateProofPassed") is True
            and launch.get("fixtureProofPassed") is True
        )

    _check(checks, "launch_artifacts_hashed", all_artifacts_hashed, value=launch_hashes)
    _check(
        checks,
        "all_launches_pass_temporal_plan_and_state",
        all_admission_pass,
    )
    _check(
        checks,
        "model_hashes_invariant",
        bool(launch_hashes)
        and all(value == launch_hashes[0] for value in launch_hashes),
        value=launch_hashes,
    )
    expected_policy = {name: "cpuAndNeuralEngine" for name in REQUIRED_MODEL_FAMILIES}
    _check(
        checks,
        "compute_policy_invariant_and_ane_only",
        bool(launch_policies)
        and all(value == launch_policies[0] for value in launch_policies)
        and launch_policies[0] == expected_policy,
        value=launch_policies,
        expected=expected_policy,
    )

    trace = manifest.get("traceEvidence")
    trace = trace if isinstance(trace, dict) else {}
    ane_evidence = trace.get("aneModelFamilies")
    ane_evidence = ane_evidence if isinstance(ane_evidence, dict) else {}
    ane_pass = set(ane_evidence) == set(REQUIRED_ANE_MODEL_FAMILIES) and all(
        _is_finite_number(ane_evidence.get(name, {}).get("anePredictionCount"))
        and ane_evidence[name]["anePredictionCount"] > 0
        and _is_finite_number(ane_evidence[name].get("anePredictionTotalNs"))
        and ane_evidence[name]["anePredictionTotalNs"] > 0
        for name in REQUIRED_ANE_MODEL_FAMILIES
    )
    runtime_evidence = trace.get("runtimeModelFamilies")
    runtime_evidence = runtime_evidence if isinstance(runtime_evidence, dict) else {}
    runtime_pass = set(runtime_evidence) == set(REQUIRED_MODEL_FAMILIES) and all(
        _is_finite_number(runtime_evidence.get(name, {}).get("coremlPredictionCount"))
        and runtime_evidence[name]["coremlPredictionCount"] > 0
        and _is_finite_number(runtime_evidence[name].get("coremlPredictionTotalNs"))
        and runtime_evidence[name]["coremlPredictionTotalNs"] > 0
        for name in REQUIRED_MODEL_FAMILIES
    )
    trace_gpu_pass = (
        trace.get("appGpuIntervalCount") == 0 and trace.get("appGpuTotalNs") == 0
    )
    _check(
        checks,
        "trace_required_models_have_ane_predictions",
        ane_pass,
        value=ane_evidence,
        expected=list(REQUIRED_ANE_MODEL_FAMILIES),
    )
    _check(
        checks,
        "trace_pipeline_models_have_coreml_predictions",
        runtime_pass,
        value=runtime_evidence,
        expected=list(REQUIRED_MODEL_FAMILIES),
    )
    _check(checks, "trace_has_zero_app_gpu", trace_gpu_pass)
    _check(
        checks,
        "trace_artifact_matches_launches",
        bool(launch_hashes) and trace.get("modelSha256") == launch_hashes[0],
        value=trace.get("modelSha256"),
        expected=launch_hashes[0] if launch_hashes else None,
    )
    _check(
        checks,
        "trace_artifacts_hashed",
        _artifact_hashes_present(trace.get("artifactSha256ByPath")),
        value=trace.get("artifactSha256ByPath"),
    )

    state = manifest.get("stateEvidence")
    state = state if isinstance(state, dict) else {}
    temporal_hash = launch_hashes[0].get("temporal") if launch_hashes else None
    _check(
        checks,
        "state_artifact_matches_launch",
        state.get("artifactSha256") == temporal_hash and _is_sha256(temporal_hash),
        value=state.get("artifactSha256"),
        expected=temporal_hash,
    )
    divergence = state.get("freshWarmedMaxAbsDelta")
    _check(
        checks,
        "state_is_read",
        _is_finite_number(divergence) and divergence > MIN_STATE_DIVERGENCE,
        value=divergence,
        expected=f">{MIN_STATE_DIVERGENCE}",
    )
    streaming_frames = state.get("streamingFrames")
    window_frames = state.get("windowFrames")
    _check(
        checks,
        "streaming_test_past_window",
        isinstance(streaming_frames, int)
        and isinstance(window_frames, int)
        and window_frames == 41
        and streaming_frames >= MIN_STREAMING_FRAMES
        and streaming_frames > window_frames,
        value={"streamingFrames": streaming_frames, "windowFrames": window_frames},
        expected={"windowFrames": 41, "streamingFrames": f">={MIN_STREAMING_FRAMES}"},
    )
    correlation = state.get("correlation")
    _check(
        checks,
        "streaming_reference_match",
        _is_finite_number(correlation) and correlation >= MIN_STATE_CORRELATION,
        value=correlation,
        expected=f">={MIN_STATE_CORRELATION}",
    )
    max_absolute_error = state.get("maxAbsoluteError")
    _check(
        checks,
        "streaming_absolute_error_bounded",
        _is_finite_number(max_absolute_error)
        and max_absolute_error <= MAX_STATE_ABSOLUTE_ERROR,
        value=max_absolute_error,
        expected=f"<={MAX_STATE_ABSOLUTE_ERROR}",
    )
    _check(
        checks,
        "state_output_finite",
        state.get("finiteRatio") == 1.0,
        value=state.get("finiteRatio"),
        expected=1.0,
    )
    _check(
        checks,
        "state_artifacts_hashed",
        _artifact_hashes_present(state.get("artifactSha256ByPath")),
        value=state.get("artifactSha256ByPath"),
    )
    return _report("G1", checks, manifest)


def _soak_checks(
    manifest: dict[str, Any],
    *,
    expected_device: str,
    require_nondraining_reservoir: bool,
) -> list[dict[str, Any]]:
    """Build checks shared by the G2 and native-real-time G4 soaks."""
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "device",
        manifest.get("device", {}).get("modelIdentifier") == expected_device,
        value=manifest.get("device", {}).get("modelIdentifier"),
        expected=expected_device,
    )
    _check(
        checks,
        "foreground_screen_on",
        manifest.get("foreground") is True and manifest.get("screenOn") is True,
        value={
            "foreground": manifest.get("foreground"),
            "screenOn": manifest.get("screenOn"),
        },
    )
    measured = manifest.get("measuredWindowSeconds")
    _check(
        checks,
        "measured_window",
        _is_finite_number(measured) and measured >= MIN_SOAK_SECONDS,
        value=measured,
        expected=f">={MIN_SOAK_SECONDS}",
    )
    pulled = manifest.get("pulledAudioSeconds")
    _check(
        checks,
        "pulled_audio",
        _is_finite_number(pulled) and pulled >= MIN_SOAK_SECONDS,
        value=pulled,
        expected=f">={MIN_SOAK_SECONDS}",
    )
    generated = manifest.get("generatedAudioSeconds")
    rate = (
        generated / measured
        if _is_finite_number(generated) and _is_finite_number(measured) and measured > 0
        else None
    )
    _check(
        checks,
        "generation_rate",
        rate is not None and rate >= 1.0,
        value=rate,
        expected=">=1.0",
    )
    _check(
        checks,
        "zero_underruns",
        manifest.get("maxUnderruns") == 0,
        value=manifest.get("maxUnderruns"),
        expected=0,
    )
    _check(
        checks,
        "zero_dropped",
        manifest.get("maxDropped") == 0,
        value=manifest.get("maxDropped"),
        expected=0,
    )
    _check(
        checks,
        "effective_frame_definition",
        manifest.get("effectiveFrameDefinition") == EXPECTED_EFFECTIVE_FRAME_DEFINITION,
        value=manifest.get("effectiveFrameDefinition"),
        expected=EXPECTED_EFFECTIVE_FRAME_DEFINITION,
    )
    frame_count = manifest.get("effectiveFrameCount")
    _check(
        checks,
        "effective_frame_count",
        isinstance(frame_count, int) and frame_count >= MIN_EFFECTIVE_FRAME_COUNT,
        value=frame_count,
        expected=f">={MIN_EFFECTIVE_FRAME_COUNT}",
    )
    p99 = manifest.get("p99EffectiveFrameMs")
    _check(
        checks,
        "p99_frame_budget",
        _is_finite_number(p99) and p99 < MAX_EFFECTIVE_FRAME_P99_MS,
        value=p99,
        expected=f"<{MAX_EFFECTIVE_FRAME_P99_MS}",
    )
    reservoir_start = manifest.get("reservoirStartFrames")
    reservoir_end = manifest.get("reservoirEndFrames")
    reservoir_slope = manifest.get("reservoirSlopeFramesPerSecond")
    if require_nondraining_reservoir:
        reservoir_pass = (
            _is_finite_number(reservoir_start)
            and _is_finite_number(reservoir_end)
            and _is_finite_number(reservoir_slope)
            and reservoir_end >= reservoir_start
            and reservoir_slope >= 0.0
        )
        _check(
            checks,
            "reservoir_not_draining",
            reservoir_pass,
            value={
                "startFrames": reservoir_start,
                "endFrames": reservoir_end,
                "slopeFramesPerSecond": reservoir_slope,
            },
            expected="end>=start and slope>=0",
        )
    _check(
        checks,
        "thermal_timeline",
        _thermal_timeline_covers(manifest.get("thermalTimeline"), MIN_SOAK_SECONDS),
        value=manifest.get("thermalTimeline"),
        expected=f"coverage>={MIN_SOAK_SECONDS}s",
    )
    _check(
        checks,
        "g1_receipt_hashed",
        _is_sha256(manifest.get("g1ReportSha256")),
        value=manifest.get("g1ReportSha256"),
    )
    _check(
        checks,
        "run_artifacts_hashed",
        _artifact_hashes_present(manifest.get("artifactSha256")),
        value=manifest.get("artifactSha256"),
    )
    return checks


def verify_g2(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify the sustained A17 Pro real-time gate."""
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "schema",
        manifest.get("schema") == G2_SCHEMA,
        value=manifest.get("schema"),
        expected=G2_SCHEMA,
    )
    checks.extend(
        _soak_checks(
            manifest, expected_device=A17_PRO_DEVICE, require_nondraining_reservoir=True
        )
    )
    return _report("G2", checks, manifest)


def verify_g3(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify the frozen objective and blinded automated audio-integrity gate."""
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "schema",
        manifest.get("schema") == G3_SCHEMA,
        value=manifest.get("schema"),
        expected=G3_SCHEMA,
    )
    protocol = manifest.get("protocol")
    protocol = protocol if isinstance(protocol, dict) else {}
    _check(
        checks,
        "fixed_prompt",
        protocol.get("prompt") == G3_FIXED_PROMPT,
        value=protocol.get("prompt"),
        expected=G3_FIXED_PROMPT,
    )
    _check(
        checks,
        "fixed_sampling",
        protocol.get("topK") == G3_TOP_K
        and _is_finite_number(protocol.get("temperature"))
        and protocol["temperature"] <= G3_MAX_TEMPERATURE,
        value={
            "topK": protocol.get("topK"),
            "temperature": protocol.get("temperature"),
        },
        expected={"topK": G3_TOP_K, "temperature": f"<={G3_MAX_TEMPERATURE}"},
    )
    _check(
        checks,
        "blind_order_seed_frozen",
        isinstance(protocol.get("blindOrderSeeds"), list)
        and len(protocol["blindOrderSeeds"]) == G3_REQUIRED_VALID_VOTES
        and len(set(protocol["blindOrderSeeds"])) == G3_REQUIRED_VALID_VOTES,
        value=protocol.get("blindOrderSeeds"),
    )
    _check(
        checks,
        "g2_receipt_hashed",
        _is_sha256(manifest.get("g2ReportSha256")),
        value=manifest.get("g2ReportSha256"),
    )
    _check(
        checks,
        "capture_has_zero_runtime_failures",
        manifest.get("maxUnderruns") == 0 and manifest.get("maxDropped") == 0,
        value={
            "maxUnderruns": manifest.get("maxUnderruns"),
            "maxDropped": manifest.get("maxDropped"),
        },
        expected={"maxUnderruns": 0, "maxDropped": 0},
    )

    metrics = manifest.get("objectiveMetrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    thresholds: tuple[tuple[str, Callable[[object], bool], object], ...] = (
        ("sampleRateHz", lambda value: value == 48_000, 48_000),
        ("channels", lambda value: value == 2, 2),
        (
            "channelOrder",
            lambda value: value == ["left", "right"],
            ["left", "right"],
        ),
        (
            "durationSeconds",
            lambda value: _is_finite_number(value) and value >= G3_MIN_DURATION_SECONDS,
            f">={G3_MIN_DURATION_SECONDS}",
        ),
        ("finiteRatio", lambda value: value == 1.0, 1.0),
        (
            "clippedSampleRatio",
            lambda value: (
                _is_finite_number(value) and value <= G3_MAX_CLIPPED_SAMPLE_RATIO
            ),
            f"<={G3_MAX_CLIPPED_SAMPLE_RATIO}",
        ),
        (
            "maxChunkBoundaryAbsJump",
            lambda value: (
                _is_finite_number(value) and value <= G3_MAX_BOUNDARY_ABS_JUMP
            ),
            f"<={G3_MAX_BOUNDARY_ABS_JUMP}",
        ),
        (
            "leftRightCorrelation",
            lambda value: (
                _is_finite_number(value) and value >= G3_MIN_LEFT_RIGHT_CORRELATION
            ),
            f">={G3_MIN_LEFT_RIGHT_CORRELATION}",
        ),
        (
            "promptAdherence",
            lambda value: _is_finite_number(value) and value >= G3_MIN_PROMPT_ADHERENCE,
            f">={G3_MIN_PROMPT_ADHERENCE}",
        ),
        (
            "embeddingSimilarityToReference",
            lambda value: (
                _is_finite_number(value) and value >= G3_MIN_EMBEDDING_SIMILARITY
            ),
            f">={G3_MIN_EMBEDDING_SIMILARITY}",
        ),
        (
            "envelopePulseShare4To16Hz",
            lambda value: (
                _is_finite_number(value) and value <= G3_MAX_ENVELOPE_PULSE_SHARE
            ),
            f"<={G3_MAX_ENVELOPE_PULSE_SHARE}",
        ),
    )
    for name, predicate, expected in thresholds:
        _check(
            checks,
            f"objective_{name}",
            predicate(metrics.get(name)),
            value=metrics.get(name),
            expected=expected,
        )

    controls = manifest.get("knownBadControls")
    controls = controls if isinstance(controls, list) else []
    control_map = {item.get("id"): item for item in controls if isinstance(item, dict)}
    _check(
        checks,
        "all_known_bad_controls_rejected",
        set(control_map) == set(REQUIRED_KNOWN_BAD_CONTROLS)
        and all(
            control_map[name].get("rejected") is True
            for name in REQUIRED_KNOWN_BAD_CONTROLS
        ),
        value=control_map,
        expected=list(REQUIRED_KNOWN_BAD_CONTROLS),
    )

    votes = manifest.get("blindAutomatedVotes")
    votes = votes if isinstance(votes, list) else []
    valid_votes = [
        vote
        for vote in votes
        if isinstance(vote, dict)
        and vote.get("knownGoodPass") is True
        and vote.get("knownBadPass") is False
        and vote.get("controlsRankedCorrectly") is True
    ]
    candidate_passes = sum(vote.get("candidatePass") is True for vote in valid_votes)
    _check(
        checks,
        "valid_blind_vote_count",
        len(valid_votes) == G3_REQUIRED_VALID_VOTES,
        value=len(valid_votes),
        expected=G3_REQUIRED_VALID_VOTES,
    )
    _check(
        checks,
        "blind_candidate_supermajority",
        candidate_passes >= G3_REQUIRED_CANDIDATE_PASSES,
        value=candidate_passes,
        expected=f">={G3_REQUIRED_CANDIDATE_PASSES}/{G3_REQUIRED_VALID_VOTES}",
    )
    _check(
        checks,
        "audio_artifacts_hashed",
        _artifact_hashes_present(manifest.get("artifactSha256")),
        value=manifest.get("artifactSha256"),
    )
    return _report("G3", checks, manifest)


def verify_g4(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify that the A14 is honestly classified as native or reservoir tier."""
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "schema",
        manifest.get("schema") == G4_SCHEMA,
        value=manifest.get("schema"),
        expected=G4_SCHEMA,
    )
    _check(
        checks,
        "a14_device",
        manifest.get("device", {}).get("modelIdentifier") == A14_DEVICE,
        value=manifest.get("device", {}).get("modelIdentifier"),
        expected=A14_DEVICE,
    )
    outcome = manifest.get("outcome")
    _check(
        checks,
        "outcome_selected",
        outcome in {"native-real-time", "reservoir-tier", "bounded-reservoir"},
        value=outcome,
        expected=["native-real-time", "reservoir-tier", "bounded-reservoir"],
    )
    evidence = manifest.get("evidence")
    evidence = copy.deepcopy(evidence) if isinstance(evidence, dict) else {}
    evidence["device"] = manifest.get("device", {})
    if outcome == "native-real-time":
        checks.extend(
            _soak_checks(
                evidence, expected_device=A14_DEVICE, require_nondraining_reservoir=True
            )
        )
    elif outcome in {"reservoir-tier", "bounded-reservoir"}:
        _check(
            checks,
            "reservoir_foreground_screen_on",
            evidence.get("foreground") is True and evidence.get("screenOn") is True,
            value={
                "foreground": evidence.get("foreground"),
                "screenOn": evidence.get("screenOn"),
            },
        )
        measured = evidence.get("measuredWindowSeconds")
        pulled = evidence.get("pulledAudioSeconds")
        _check(
            checks,
            "reservoir_measured_window",
            _is_finite_number(measured) and measured >= MIN_SOAK_SECONDS,
            value=measured,
            expected=f">={MIN_SOAK_SECONDS}",
        )
        if outcome == "reservoir-tier":
            _check(
                checks,
                "reservoir_pulled_audio",
                _is_finite_number(pulled) and pulled >= MIN_SOAK_SECONDS,
                value=pulled,
                expected=f">={MIN_SOAK_SECONDS}",
            )
            _check(
                checks,
                "reservoir_zero_underruns",
                evidence.get("maxUnderruns") == 0,
                value=evidence.get("maxUnderruns"),
                expected=0,
            )
        else:
            continuous = evidence.get("continuousPlaySecondsBeforeFirstUnderrun")
            capture = evidence.get("pcmCaptureSeconds")
            maximum_reservoir = evidence.get("maximumStartReservoirSeconds")
            _check(
                checks,
                "bounded_reservoir_failure_measured",
                evidence.get("maxUnderruns", 0) > 0
                and _is_finite_number(continuous)
                and continuous > 0
                and continuous < MIN_SOAK_SECONDS
                and _is_finite_number(capture)
                and capture > 0
                and capture < MIN_SOAK_SECONDS
                and _is_finite_number(maximum_reservoir)
                and maximum_reservoir > 0,
                value={
                    "maxUnderruns": evidence.get("maxUnderruns"),
                    "continuousPlaySecondsBeforeFirstUnderrun": continuous,
                    "pcmCaptureSeconds": capture,
                    "maximumStartReservoirSeconds": maximum_reservoir,
                },
                expected="measured underflow boundary below 600 s",
            )
        _check(
            checks,
            "reservoir_zero_dropped",
            evidence.get("maxDropped") == 0,
            value=evidence.get("maxDropped"),
            expected=0,
        )
        prime = evidence.get("primeSeconds")
        startup = evidence.get("startupToFirstAudioSeconds")
        _check(
            checks,
            "reservoir_startup_reported",
            _is_finite_number(prime)
            and prime > 0
            and _is_finite_number(startup)
            and startup >= prime,
            value={"primeSeconds": prime, "startupToFirstAudioSeconds": startup},
            expected="startup>=prime>0",
        )
        percentiles = [
            evidence.get(name)
            for name in (
                "p50EffectiveFrameMs",
                "p90EffectiveFrameMs",
                "p99EffectiveFrameMs",
            )
        ]
        _check(
            checks,
            "reservoir_latency_distribution",
            all(_is_finite_number(value) and value > 0 for value in percentiles)
            and percentiles == sorted(percentiles),
            value=percentiles,
            expected="0<p50<=p90<=p99",
        )
        _check(
            checks,
            "reservoir_effective_frame_definition",
            evidence.get("effectiveFrameDefinition")
            == EXPECTED_EFFECTIVE_FRAME_DEFINITION,
            value=evidence.get("effectiveFrameDefinition"),
            expected=EXPECTED_EFFECTIVE_FRAME_DEFINITION,
        )
        frame_count = evidence.get("effectiveFrameCount")
        minimum_frame_count = (
            MIN_EFFECTIVE_FRAME_COUNT
            if outcome == "reservoir-tier"
            else 10_000
        )
        _check(
            checks,
            "reservoir_effective_frame_count",
            isinstance(frame_count, int) and frame_count >= minimum_frame_count,
            value=frame_count,
            expected=f">={minimum_frame_count}",
        )
        _check(
            checks,
            "reservoir_rate_reported",
            _is_finite_number(evidence.get("generationRate"))
            and evidence["generationRate"] > 0,
            value=evidence.get("generationRate"),
            expected=">0 (reported, not gated)",
        )
        _check(
            checks,
            "reservoir_slope_reported",
            _is_finite_number(evidence.get("reservoirSlopeFramesPerSecond")),
            value=evidence.get("reservoirSlopeFramesPerSecond"),
        )
        if outcome == "reservoir-tier":
            _check(
                checks,
                "reservoir_remains_nonempty",
                _is_finite_number(evidence.get("reservoirEndFrames"))
                and evidence["reservoirEndFrames"] > 0,
                value=evidence.get("reservoirEndFrames"),
                expected=">0",
            )
        _check(
            checks,
            "reservoir_thermal_timeline",
            _thermal_timeline_covers(evidence.get("thermalTimeline"), MIN_SOAK_SECONDS),
            value=evidence.get("thermalTimeline"),
            expected=f"coverage>={MIN_SOAK_SECONDS}s",
        )
        _check(
            checks,
            "reservoir_unqualified_claim_forbidden",
            evidence.get("unqualifiedRealTimeClaimAllowed") is False,
            value=evidence.get("unqualifiedRealTimeClaimAllowed"),
            expected=False,
        )
        _check(
            checks,
            "reservoir_artifacts_hashed",
            _artifact_hashes_present(evidence.get("artifactSha256")),
            value=evidence.get("artifactSha256"),
        )
    return _report("G4", checks, manifest)


def _report(
    gate: str, checks: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Build a stable verification report with a manifest digest."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": REPORT_SCHEMA,
        "gate": gate,
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "manifestSha256": hashlib.sha256(canonical).hexdigest(),
        "checks": checks,
    }


VERIFIERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "g1": verify_g1,
    "g2": verify_g2,
    "g3": verify_g3,
    "g4": verify_g4,
}


def main() -> None:
    """Run one frozen gate verifier from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=sorted(VERIFIERS))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    report = VERIFIERS[args.gate](_load_json(args.manifest))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
