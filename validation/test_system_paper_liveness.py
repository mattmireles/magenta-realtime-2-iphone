"""Tests for the G5/G6 publication-contract verifier."""

from __future__ import annotations

import copy
import unittest

from validation.verify_system_paper_liveness import G5_SCHEMA, G6_SCHEMA, verify_g5, verify_g6


H = "a" * 64


def _g5() -> dict:
    seeds = []
    counts = {
        20260718: {"off": 2, "kv-only": 0, "feedback-only": 0, "both": 0},
        271828: {"off": 57, "kv-only": 0, "feedback-only": 13, "both": 0},
        1618033: {"off": 46, "kv-only": 0, "feedback-only": 0, "both": 0},
    }
    resets = {"off": (0, 0), "kv-only": (60, 0), "feedback-only": (0, 60), "both": (60, 60)}
    for seed, per_policy in counts.items():
        arms = {}
        for policy, count in per_policy.items():
            kv, feedback = resets[policy]
            arms[policy] = {
                "fullScaleSampleCount": count,
                "peakAbsoluteSample": 1.1 if count else 0.9,
                "resetCounts": {"kv": kv, "feedback": feedback, "rng": 0, "absoluteStep": 0},
                "tokenSha256": H,
                "wavSha256": H,
            }
        seeds.append({"seed": seed, "arms": arms})
    return {
        "schema": G5_SCHEMA,
        "status": "fail",
        "sourceEvidence": {"protocol": H, "analysis": H},
        "primaryFailureMetric": {"id": "floatPcmFullScaleExceedance", "comparison": "abs(sample) >= 1.0", "requiredCandidateCount": 0},
        "seeds": seeds,
        "aggregate": {"classification": "ambiguous", "classificationCounts": {"ambiguous": 2, "cache-only-rescue": 1}, "selectedMitigation": None},
        "eventCenteredModelVotes": [{"baselineVerdict": "pass", "corruptionVerdict": "fail", "candidateVerdict": "fail", "lineupSha256": H, "resultSha256": H} for _ in range(9)],
        "permittedClaim": "The tested no-reset 600-second warm-prompt configuration failed the frozen liveness gate.",
    }


def _g6() -> dict:
    return {
        "schema": G6_SCHEMA,
        "status": "live",
        "device": {"modelIdentifier": "iPhone16,1", "udid": "device", "osVersion": "26.5", "osBuild": "23F1"},
        "runtime": {"decoderInputFrames": 5, "decoderContextFrames": 2, "decoderStrideFrames": 3, "sampleRate": 48000, "channels": 2, "trajectoryRefreshSeconds": 0, "queueTargetFrames": 7680, "fadeInFrames": 1920},
        "postRingProof": {"measurementKind": "full", "transitionCount": 30, "detectedTransitionCount": 30, "calibrationCount": 4, "runDurationSeconds": 600, "durationSeconds": 590, "framesWritten": 28_320_000, "capturedFrames": 28_800_384, "sampleRate": 48000, "channels": 2, "finiteRatio": 1.0, "fullScaleSampleCount": 0, "underrunFrames": 0, "producerDroppedFrames": 0, "captureDroppedFrames": 0, "captureOverflowEvents": 0, "detectorThresholdSource": "paired-no-op-calibration", "detectorThreshold": 0.4, "detectorId": "event-aligned-paired-waveform-reference-v1", "classTransitionCounts": {"cmaj": 15, "cmin": 15}, "controlEpochTransitionCount": 37, "prunedPendingDecoderFrames": 10, "temporalResetCount": 37, "controlTriggeredResetCount": 37, "rngResetCount": 37, "p95EndToEndSeconds": 0.19, "maxEndToEndSeconds": 0.24, "latencyReportSha256": H, "postRingWavSha256": H},
        "listening": {"validModelVotes": 3, "unanimousNoClickCorruption": True, "physicalSpeakerCheck": "pass", "voteReportSha256": [H, H, H], "physicalSpeakerRecordSha256": H},
        "permittedClaim": "The tested iPhone 15 Pro configuration attained the live rung.",
        "artifactSha256": {"events": H},
    }


class SystemPaperLivenessTests(unittest.TestCase):
    def test_failed_g5_is_a_valid_publishable_result(self) -> None:
        self.assertTrue(verify_g5(_g5())["passed"])

    def test_g5_rejects_threshold_relaxation(self) -> None:
        manifest = _g5()
        manifest["primaryFailureMetric"]["comparison"] = "abs(sample) >= 1.25"
        self.assertFalse(verify_g5(manifest)["passed"])

    def test_g5_rejects_hidden_no_reset_intervention(self) -> None:
        manifest = _g5()
        manifest["seeds"][0]["arms"]["off"]["resetCounts"]["kv"] = 1
        self.assertFalse(verify_g5(manifest)["passed"])

    def test_g6_live_accepts_a17_pro_family_and_requires_speaker_check(self) -> None:
        self.assertTrue(verify_g6(_g6())["passed"])
        pro_max = copy.deepcopy(_g6())
        pro_max["device"]["modelIdentifier"] = "iPhone16,2"
        self.assertTrue(verify_g6(pro_max)["passed"])
        no_speaker = copy.deepcopy(_g6())
        no_speaker["listening"]["physicalSpeakerCheck"] = "not-run"
        self.assertFalse(verify_g6(no_speaker)["passed"])

    def test_g6_accepts_complete_negative_measurement_as_buffered(self) -> None:
        manifest = _g6()
        manifest["status"] = "buffered"
        manifest["postRingProof"]["underrunFrames"] = 1
        manifest["listening"] = {
            "validModelVotes": 0,
            "unanimousNoClickCorruption": False,
            "physicalSpeakerCheck": "not-run",
        }
        manifest["permittedClaim"] = "The tested iPhone 15 Pro configuration remains buffered."
        self.assertTrue(verify_g6(manifest)["passed"])

    def test_g6_missed_transition_blocks_live_promotion(self) -> None:
        manifest = _g6()
        manifest["status"] = "buffered"
        manifest["postRingProof"]["detectedTransitionCount"] = 29
        manifest["listening"] = {
            "validModelVotes": 0,
            "unanimousNoClickCorruption": False,
            "physicalSpeakerCheck": "not-run",
        }
        manifest["permittedClaim"] = "The tested iPhone 15 Pro configuration remains buffered."
        self.assertTrue(verify_g6(manifest)["passed"])

    def test_g6_accepts_predeclared_diagnostic_failure_as_buffered(self) -> None:
        manifest = _g6()
        manifest["status"] = "buffered"
        manifest["device"]["modelIdentifier"] = "iPhone16,2"
        manifest["postRingProof"] = {
            "measurementKind": "fail-fast-diagnostic",
            "diagnosticOutcome": "no-calibrated-transition-detected",
            "transitionCount": 5,
            "detectedTransitionCount": 0,
            "calibrationCount": 6,
            "durationSeconds": 200,
            "framesWritten": 9_600_000,
            "capturedFrames": 9_600_384,
            "sampleRate": 48_000,
            "channels": 2,
            "finiteRatio": 1.0,
            "fullScaleSampleCount": 0,
            "underrunFrames": 0,
            "producerDroppedFrames": 0,
            "captureDroppedFrames": 0,
            "captureOverflowEvents": 0,
            "detectorThresholdSource": "no-op-calibration",
            "detectorThreshold": 0.0665,
            "detectorId": "within-run-leave-one-transition-out-spectral-prototype-v1",
            "classTransitionCounts": {"warm": 3, "techno": 2},
            "controlEpochTransitionCount": 12,
            "controlTriggeredResetCount": 12,
            "rngResetCount": 12,
            "latencyReportSha256": H,
            "postRingWavSha256": H,
        }
        manifest["listening"] = {"physicalSpeakerCheck": "not-run"}
        manifest["permittedClaim"] = "The tested A17 Pro configuration remains buffered."
        self.assertTrue(verify_g6(manifest)["passed"])


if __name__ == "__main__":
    unittest.main()
