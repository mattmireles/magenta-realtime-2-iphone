#!/usr/bin/env python3
"""Verify the publication contracts for MRT2 liveness (G5) and steering (G6)."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


G5_SCHEMA = "mrt2-system-paper-g5-v1"
G6_SCHEMA = "mrt2-system-paper-g6-v1"
G5_SEEDS = {20260718, 271828, 1618033}
G5_POLICIES = {"off", "kv-only", "feedback-only", "both"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
A17_PRO_MODEL_IDENTIFIERS = {"iPhone16,1", "iPhone16,2"}


def _sha(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _no_absolute_paths(value: object) -> bool:
    if isinstance(value, dict):
        return all(_no_absolute_paths(key) and _no_absolute_paths(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_no_absolute_paths(item) for item in value)
    return not (isinstance(value, str) and value.startswith("/"))


def _result(schema: str, checks: list[tuple[str, bool, object]]) -> dict[str, Any]:
    rows = [{"name": name, "passed": passed, "value": value} for name, passed, value in checks]
    return {"schema": schema, "passed": all(row["passed"] for row in rows), "checks": rows}


def verify_g5(manifest: dict[str, Any]) -> dict[str, Any]:
    checks: list[tuple[str, bool, object]] = []
    checks.append(("schema", manifest.get("schema") == G5_SCHEMA, manifest.get("schema")))
    checks.append(("status_is_failed", manifest.get("status") == "fail", manifest.get("status")))
    checks.append(("no_machine_local_paths", _no_absolute_paths(manifest), None))
    source = manifest.get("sourceEvidence", {})
    checks.append((
        "source_evidence_hashes",
        isinstance(source, dict) and bool(source) and all(_sha(value) for value in source.values()),
        source,
    ))
    metric = manifest.get("primaryFailureMetric", {})
    checks.append((
        "frozen_threshold",
        metric == {
            "id": "floatPcmFullScaleExceedance",
            "comparison": "abs(sample) >= 1.0",
            "requiredCandidateCount": 0,
        },
        metric,
    ))

    rows = manifest.get("seeds")
    rows = rows if isinstance(rows, list) else []
    checks.append(("exact_seed_set", {row.get("seed") for row in rows if isinstance(row, dict)} == G5_SEEDS, len(rows)))
    all_valid = len(rows) == len(G5_SEEDS)
    all_off_fail = all_valid
    all_both_pass = all_valid
    for row in rows:
        if not isinstance(row, dict):
            all_valid = all_off_fail = all_both_pass = False
            continue
        arms = row.get("arms")
        if not isinstance(arms, dict) or set(arms) != G5_POLICIES:
            all_valid = all_off_fail = all_both_pass = False
            continue
        for policy, arm in arms.items():
            if not isinstance(arm, dict):
                all_valid = False
                continue
            resets = arm.get("resetCounts", {})
            expected = {
                "off": (0, 0),
                "kv-only": (60, 0),
                "feedback-only": (0, 60),
                "both": (60, 60),
            }[policy]
            all_valid &= (
                resets.get("kv") == expected[0]
                and resets.get("feedback") == expected[1]
                and resets.get("rng") == 0
                and resets.get("absoluteStep") == 0
                and _finite(arm.get("peakAbsoluteSample"))
                and isinstance(arm.get("fullScaleSampleCount"), int)
                and arm["fullScaleSampleCount"] >= 0
                and _sha(arm.get("tokenSha256"))
                and _sha(arm.get("wavSha256"))
            )
        all_off_fail &= arms["off"].get("fullScaleSampleCount", 0) > 0
        all_both_pass &= arms["both"].get("fullScaleSampleCount") == 0
    checks.append(("all_arms_valid", all_valid, len(rows) * 4))
    checks.append(("off_failure_replicates_3_of_3", all_off_fail, None))
    checks.append(("matched_both_controls_pass_3_of_3", all_both_pass, None))

    aggregate = manifest.get("aggregate", {})
    checks.append((
        "factorial_attribution_is_ambiguous",
        aggregate.get("classification") == "ambiguous"
        and aggregate.get("classificationCounts") == {"ambiguous": 2, "cache-only-rescue": 1}
        and aggregate.get("selectedMitigation") is None,
        aggregate,
    ))
    votes = manifest.get("eventCenteredModelVotes")
    votes = votes if isinstance(votes, list) else []
    controls_valid = len(votes) == 9 and all(
        vote.get("baselineVerdict") == "pass"
        and vote.get("corruptionVerdict") == "fail"
        and vote.get("candidateVerdict") in {"pass", "fail"}
        and _sha(vote.get("lineupSha256"))
        and _sha(vote.get("resultSha256"))
        for vote in votes if isinstance(vote, dict)
    )
    checks.append(("supplemental_votes_control_valid", controls_valid, len(votes)))
    wording = manifest.get("permittedClaim")
    checks.append((
        "claim_is_narrow",
        isinstance(wording, str)
        and "tested" in wording.lower()
        and "all prompts" not in wording.lower()
        and "never" not in wording.lower(),
        wording,
    ))
    return _result("mrt2-system-paper-g5-report-v1", checks)


def verify_g6(manifest: dict[str, Any]) -> dict[str, Any]:
    checks: list[tuple[str, bool, object]] = []
    checks.append(("schema", manifest.get("schema") == G6_SCHEMA, manifest.get("schema")))
    checks.append(("no_machine_local_paths", _no_absolute_paths(manifest), None))
    status = manifest.get("status")
    checks.append(("valid_status", status in {"buffered", "responsive", "live"}, status))
    device = manifest.get("device", {})
    checks.append((
        "a17_pro_device_identified",
        device.get("modelIdentifier") in A17_PRO_MODEL_IDENTIFIERS
        and isinstance(device.get("udid"), str)
        and bool(device.get("udid"))
        and isinstance(device.get("osVersion"), str)
        and bool(device.get("osVersion"))
        and isinstance(device.get("osBuild"), str)
        and bool(device.get("osBuild")),
        device,
    ))
    runtime = manifest.get("runtime", {})
    decoder_input = runtime.get("decoderInputFrames")
    context = runtime.get("decoderContextFrames")
    checks.append((
        "runtime_protocol_frozen",
        isinstance(decoder_input, int)
        and decoder_input in {5, 25}
        and isinstance(context, int)
        and 0 < context < decoder_input
        and runtime.get("decoderStrideFrames") == decoder_input - context
        and runtime.get("sampleRate") == 48_000
        and runtime.get("channels") == 2
        and runtime.get("trajectoryRefreshSeconds") in {0, 10}
        and isinstance(runtime.get("queueTargetFrames"), int)
        and runtime["queueTargetFrames"] > 0
        and isinstance(runtime.get("fadeInFrames"), int)
        and runtime["fadeInFrames"] >= 0,
        runtime,
    ))
    proof = manifest.get("postRingProof", {})
    class_counts = proof.get("classTransitionCounts", {})
    nonnegative_counters = all(
        isinstance(proof.get(key), int) and proof[key] >= 0
        for key in (
            "underrunFrames",
            "producerDroppedFrames",
            "captureDroppedFrames",
            "captureOverflowEvents",
            "fullScaleSampleCount",
        )
    )
    measurement_complete = (
        proof.get("measurementKind") == "full"
        and proof.get("transitionCount", 0) >= 30
        and proof.get("calibrationCount", 0) >= 4
        and proof.get("runDurationSeconds", 0) >= 600
        and proof.get("durationSeconds", 0) >= 590
        and proof.get("framesWritten", 0) >= 28_320_000
        and proof.get("capturedFrames", 0) >= proof.get("framesWritten", 0)
        and proof.get("sampleRate") == 48_000
        and proof.get("channels") == 2
        and proof.get("finiteRatio") == 1.0
        and nonnegative_counters
        and proof.get("detectorThresholdSource") == "paired-no-op-calibration"
        and proof.get("detectorId")
        == "event-aligned-paired-waveform-reference-v1"
        and isinstance(class_counts, dict)
        and len(class_counts) == 2
        and all(isinstance(value, int) and value >= 15 for value in class_counts.values())
        and sum(class_counts.values()) >= 30
        and proof.get("controlEpochTransitionCount", 0) >= 30
        and proof.get("prunedPendingDecoderFrames", 0) > 0
        and proof.get("temporalResetCount", 0) >= 30
        and proof.get("controlTriggeredResetCount", 0) >= 30
        and proof.get("rngResetCount") == proof.get("controlTriggeredResetCount")
        and isinstance(proof.get("detectedTransitionCount"), int)
        and 0 <= proof["detectedTransitionCount"] <= proof["transitionCount"]
        and _finite(proof.get("detectorThreshold"))
        and _sha(proof.get("latencyReportSha256"))
        and _sha(proof.get("postRingWavSha256"))
    )
    diagnostic_failure = (
        proof.get("measurementKind") == "fail-fast-diagnostic"
        and proof.get("diagnosticOutcome") == "no-calibrated-transition-detected"
        and proof.get("transitionCount", 0) >= 5
        and proof.get("detectedTransitionCount") == 0
        and proof.get("calibrationCount", 0) >= 6
        and proof.get("durationSeconds", 0) >= 200
        and proof.get("framesWritten", 0) >= 9_600_000
        and proof.get("capturedFrames", 0) >= proof.get("framesWritten", 0)
        and proof.get("sampleRate") == 48_000
        and proof.get("channels") == 2
        and proof.get("finiteRatio") == 1.0
        and nonnegative_counters
        and proof.get("detectorThresholdSource") == "no-op-calibration"
        and proof.get("detectorId")
        == "within-run-leave-one-transition-out-spectral-prototype-v1"
        and isinstance(class_counts, dict)
        and class_counts == {"warm": 3, "techno": 2}
        and proof.get("controlEpochTransitionCount", 0) >= 12
        and proof.get("controlTriggeredResetCount", 0) >= 12
        and proof.get("rngResetCount") == proof.get("controlTriggeredResetCount")
        and _finite(proof.get("detectorThreshold"))
        and _sha(proof.get("latencyReportSha256"))
        and _sha(proof.get("postRingWavSha256"))
    )
    checks.append((
        "post_ring_evidence_sufficient",
        measurement_complete or diagnostic_failure,
        {"measurementComplete": measurement_complete, "diagnosticFailure": diagnostic_failure},
    ))
    integrity_pass = (
        measurement_complete
        and proof.get("detectedTransitionCount") == proof.get("transitionCount")
        and proof.get("underrunFrames") == 0
        and proof.get("producerDroppedFrames") == 0
        and proof.get("captureDroppedFrames") == 0
        and proof.get("captureOverflowEvents") == 0
        and proof.get("fullScaleSampleCount") == 0
    )
    p95 = proof.get("p95EndToEndSeconds")
    maximum = proof.get("maxEndToEndSeconds")
    expected_status = "buffered"
    if (
        integrity_pass
        and _finite(p95)
        and _finite(maximum)
        and 0 <= p95 <= 0.5
        and 0 <= maximum <= 0.75
    ):
        expected_status = "responsive"
    if (
        integrity_pass
        and _finite(p95)
        and _finite(maximum)
        and 0 <= p95 <= 0.2
        and 0 <= maximum <= 0.25
    ):
        expected_status = "live"
    listening = manifest.get("listening", {})
    listening_pass = (
        listening.get("validModelVotes") == 3
        and listening.get("unanimousNoClickCorruption") is True
        and listening.get("physicalSpeakerCheck") == "pass"
        and isinstance(listening.get("voteReportSha256"), list)
        and len(listening["voteReportSha256"]) == 3
        and all(_sha(value) for value in listening["voteReportSha256"])
        and _sha(listening.get("physicalSpeakerRecordSha256"))
    )
    if expected_status in {"responsive", "live"}:
        expected_status = expected_status if listening_pass else "buffered"
    checks.append(("status_matches_attained_rung", status == expected_status, expected_status))
    wording = manifest.get("permittedClaim")
    checks.append((
        "claim_matches_attained_rung",
        isinstance(wording, str)
        and expected_status in wording.lower()
        and "all prompts" not in wording.lower()
        and "instant" not in wording.lower(),
        wording,
    ))
    checks.append((
        "artifacts_hashed",
        isinstance(manifest.get("artifactSha256"), dict)
        and bool(manifest["artifactSha256"])
        and all(_sha(value) for value in manifest["artifactSha256"].values()),
        manifest.get("artifactSha256"),
    ))
    return _result("mrt2-system-paper-g6-report-v1", checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g5-manifest", type=Path, required=True)
    parser.add_argument("--g6-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    g5 = verify_g5(json.loads(args.g5_manifest.read_text()))
    g6 = verify_g6(json.loads(args.g6_manifest.read_text()))
    report = {
        "schema": "mrt2-system-paper-liveness-verdict-v1",
        "passed": g5["passed"] and g6["passed"],
        "g5": g5,
        "g6": g6,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output_json)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
