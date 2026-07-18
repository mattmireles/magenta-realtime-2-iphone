"""Regression tests for the frozen system-paper gate verifier."""

from __future__ import annotations

import unittest

from validation.verify_system_paper_gate import (
    G1_SCHEMA,
    G2_SCHEMA,
    G3_SCHEMA,
    G4_SCHEMA,
    REQUIRED_KNOWN_BAD_CONTROLS,
    verify_g1,
    verify_g2,
    verify_g3,
    verify_g4,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _thermal_timeline() -> list[dict[str, object]]:
    """Return a timeline spanning the 600-second gate."""
    return [
        {"elapsedSeconds": 0.0, "state": "nominal"},
        {"elapsedSeconds": 300.0, "state": "fair"},
        {"elapsedSeconds": 600.0, "state": "serious"},
    ]


def _artifact_hashes() -> dict[str, str]:
    """Return a minimal valid artifact manifest."""
    return {"receipt.json": HASH_D}


def _passing_g1(
    *,
    device_role: str = "primary",
    model_identifier: str = "iPhone16,2",
) -> dict[str, object]:
    """Return a passing G1 manifest."""
    launches = []
    for index in range(10):
        launches.append(
            {
                "runId": f"cold-{index:02d}",
                "coldLaunch": True,
                "modelSha256": {"temporal": HASH_A, "depth": HASH_B, "decoder": HASH_C},
                "computePolicy": {
                    "temporal": "cpuAndNeuralEngine",
                    "depth": "cpuAndNeuralEngine",
                    "decoder": "cpuAndNeuralEngine",
                },
                "placementEvidence": {
                    "modelFamilies": {
                        name: {"anePredictionCount": 10, "anePredictionTotalNs": 1_000}
                        for name in ("temporal", "depth", "decoder")
                    },
                    "appGpuIntervalCount": 0,
                    "appGpuTotalNs": 0,
                },
                "artifactSha256": _artifact_hashes(),
            }
        )
    return {
        "schema": G1_SCHEMA,
        "device": {"modelIdentifier": model_identifier},
        "deviceRole": device_role,
        "launches": launches,
        "stateEvidence": {
            "artifactSha256": HASH_A,
            "freshWarmedMaxAbsDelta": 0.5,
            "streamingFrames": 64,
            "windowFrames": 41,
            "relativeMaxError": 0.008,
            "finiteRatio": 1.0,
            "artifactSha256ByPath": _artifact_hashes(),
        },
    }


def _passing_soak(schema: str, device: str) -> dict[str, object]:
    """Return a passing native-real-time soak manifest."""
    return {
        "schema": schema,
        "device": {"modelIdentifier": device},
        "foreground": True,
        "screenOn": True,
        "measuredWindowSeconds": 600.0,
        "pulledAudioSeconds": 600.0,
        "generatedAudioSeconds": 610.0,
        "maxUnderruns": 0,
        "maxDropped": 0,
        "effectiveFrameDefinition": "temporal+depth+sampling+decoder/decoded_frames",
        "effectiveFrameCount": 15_250,
        "p99EffectiveFrameMs": 35.0,
        "reservoirStartFrames": 96_000,
        "reservoirEndFrames": 100_000,
        "reservoirSlopeFramesPerSecond": 6.67,
        "thermalTimeline": _thermal_timeline(),
        "g1ReportSha256": HASH_A,
        "artifactSha256": _artifact_hashes(),
    }


def _passing_g3() -> dict[str, object]:
    """Return a passing G3 manifest."""
    return {
        "schema": G3_SCHEMA,
        "protocol": {
            "prompt": "warm ambient texture",
            "topK": 40,
            "temperature": 1.0,
            "blindOrderSeeds": [11, 22, 33, 44, 55],
        },
        "g2ReportSha256": HASH_A,
        "maxUnderruns": 0,
        "maxDropped": 0,
        "objectiveMetrics": {
            "sampleRateHz": 48_000,
            "channels": 2,
            "channelOrder": ["left", "right"],
            "durationSeconds": 600.0,
            "finiteRatio": 1.0,
            "clippedSampleRatio": 0.0,
            "maxChunkBoundaryAbsJump": 0.06,
            "leftRightCorrelation": 0.99,
            "promptAdherence": 0.33,
            "embeddingSimilarityToReference": 0.87,
            "envelopePulseShare4To16Hz": 0.056,
        },
        "knownBadControls": [
            {"id": name, "rejected": True} for name in REQUIRED_KNOWN_BAD_CONTROLS
        ],
        "blindAutomatedVotes": [
            {
                "knownGoodPass": True,
                "knownBadPass": False,
                "controlsRankedCorrectly": True,
                "candidatePass": index < 4,
            }
            for index in range(5)
        ],
        "artifactSha256": _artifact_hashes(),
    }


class SystemPaperGateTests(unittest.TestCase):
    """Prove each gate accepts its contract and rejects its binding failure."""

    def test_g1_passes_ten_invariant_cold_launches(self) -> None:
        """G1 accepts ten ANE-proven, GPU-empty launches plus valid state evidence."""
        self.assertTrue(verify_g1(_passing_g1())["passed"])

    def test_g1_rejects_one_app_gpu_interval(self) -> None:
        """Any app-attributed GPU interval invalidates the GPU-free claim."""
        manifest = _passing_g1()
        manifest["launches"][4]["placementEvidence"]["appGpuIntervalCount"] = 1
        self.assertFalse(verify_g1(manifest)["passed"])

    def test_g1_rejects_hash_drift(self) -> None:
        """A different model instance in one launch invalidates reliability evidence."""
        manifest = _passing_g1()
        manifest["launches"][9]["modelSha256"]["temporal"] = HASH_D
        self.assertFalse(verify_g1(manifest)["passed"])

    def test_g1_accepts_declared_a14_cross_device_run(self) -> None:
        """The same launch contract can report supporting evidence on A14."""
        manifest = _passing_g1(
            device_role="cross-device", model_identifier="iPhone13,3"
        )
        self.assertTrue(verify_g1(manifest)["passed"])

    def test_g1_rejects_device_role_mismatch(self) -> None:
        """An A14 manifest cannot be mislabeled as the primary A17 evidence."""
        manifest = _passing_g1(device_role="primary", model_identifier="iPhone13,3")
        self.assertFalse(verify_g1(manifest)["passed"])

    def test_g2_passes_nondraining_600_second_soak(self) -> None:
        """G2 accepts sustained production with p99 headroom and no reservoir drain."""
        self.assertTrue(verify_g2(_passing_soak(G2_SCHEMA, "iPhone16,2"))["passed"])

    def test_g2_rejects_reservoir_survival(self) -> None:
        """A zero-underrun run that drains its prime is not native real time."""
        manifest = _passing_soak(G2_SCHEMA, "iPhone16,2")
        manifest["reservoirEndFrames"] = 10_000
        manifest["reservoirSlopeFramesPerSecond"] = -143.3
        self.assertFalse(verify_g2(manifest)["passed"])

    def test_g3_passes_frozen_metrics_controls_and_votes(self) -> None:
        """G3 accepts the predeclared objective bands and blinded vote supermajority."""
        self.assertTrue(verify_g3(_passing_g3())["passed"])

    def test_g3_rejects_unreliable_listening_votes(self) -> None:
        """A lineup that misranks controls is discarded and makes the gate fail."""
        manifest = _passing_g3()
        manifest["blindAutomatedVotes"][0]["controlsRankedCorrectly"] = False
        self.assertFalse(verify_g3(manifest)["passed"])

    def test_g3_rejects_click_metric_even_when_votes_pass(self) -> None:
        """Subjective votes cannot override the frozen click boundary."""
        manifest = _passing_g3()
        manifest["objectiveMetrics"]["maxChunkBoundaryAbsJump"] = 0.08
        self.assertFalse(verify_g3(manifest)["passed"])

    def test_g4_accepts_native_a14_outcome(self) -> None:
        """G4 accepts a true A14 native real-time pass."""
        evidence = _passing_soak(G4_SCHEMA, "iPhone13,3")
        evidence.pop("schema")
        manifest = {
            "schema": G4_SCHEMA,
            "device": {"modelIdentifier": "iPhone13,3"},
            "outcome": "native-real-time",
            "evidence": evidence,
        }
        self.assertTrue(verify_g4(manifest)["passed"])

    def test_g4_accepts_honest_reservoir_tier(self) -> None:
        """G4 accepts a fully measured A14 reservoir tier without a realtime claim."""
        manifest = {
            "schema": G4_SCHEMA,
            "device": {"modelIdentifier": "iPhone13,3"},
            "outcome": "reservoir-tier",
            "evidence": {
                "foreground": True,
                "screenOn": True,
                "measuredWindowSeconds": 600.0,
                "pulledAudioSeconds": 600.0,
                "maxUnderruns": 0,
                "maxDropped": 0,
                "primeSeconds": 15.0,
                "startupToFirstAudioSeconds": 17.0,
                "p50EffectiveFrameMs": 47.0,
                "p90EffectiveFrameMs": 49.0,
                "p99EffectiveFrameMs": 52.0,
                "effectiveFrameDefinition": "temporal+depth+sampling+decoder/decoded_frames",
                "effectiveFrameCount": 15_000,
                "generationRate": 0.96,
                "reservoirSlopeFramesPerSecond": -1_920.0,
                "reservoirEndFrames": 288_000,
                "thermalTimeline": _thermal_timeline(),
                "unqualifiedRealTimeClaimAllowed": False,
                "artifactSha256": _artifact_hashes(),
            },
        }
        self.assertTrue(verify_g4(manifest)["passed"])

    def test_g4_rejects_reservoir_tier_labeled_real_time(self) -> None:
        """The A14 buffering result cannot opt back into an unqualified claim."""
        manifest = {
            "schema": G4_SCHEMA,
            "device": {"modelIdentifier": "iPhone13,3"},
            "outcome": "reservoir-tier",
            "evidence": {
                "foreground": True,
                "screenOn": True,
                "measuredWindowSeconds": 600.0,
                "pulledAudioSeconds": 600.0,
                "maxUnderruns": 0,
                "maxDropped": 0,
                "primeSeconds": 15.0,
                "startupToFirstAudioSeconds": 17.0,
                "p50EffectiveFrameMs": 47.0,
                "p90EffectiveFrameMs": 49.0,
                "p99EffectiveFrameMs": 52.0,
                "effectiveFrameDefinition": "temporal+depth+sampling+decoder/decoded_frames",
                "effectiveFrameCount": 15_000,
                "generationRate": 0.96,
                "reservoirSlopeFramesPerSecond": -1_920.0,
                "reservoirEndFrames": 288_000,
                "thermalTimeline": _thermal_timeline(),
                "unqualifiedRealTimeClaimAllowed": True,
                "artifactSha256": _artifact_hashes(),
            },
        }
        self.assertFalse(verify_g4(manifest)["passed"])


if __name__ == "__main__":
    unittest.main()
