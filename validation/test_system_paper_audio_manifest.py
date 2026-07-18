"""Tests for the G3 audio-manifest normalizer."""

from __future__ import annotations

import argparse
import json

from build_system_paper_audio_manifest import build
from verify_system_paper_gate import REQUIRED_KNOWN_BAD_CONTROLS, verify_g3


def _write_json(path, value):
  path.write_text(json.dumps(value) + "\n")


def test_blind_votes_are_resolved_through_neutral_labels(tmp_path):
  analysis = tmp_path / "analysis.json"
  lineups = tmp_path / "lineups.json"
  g2_manifest = tmp_path / "g2-manifest.json"
  g2_report = tmp_path / "g2-report.json"
  vote_dir = tmp_path / "votes"
  vote_dir.mkdir()
  metrics = {
      "sampleRateHz": 48_000,
      "channels": 2,
      "channelOrder": ["left", "right"],
      "durationSeconds": 600.0,
      "finiteRatio": 1.0,
      "clippedSampleRatio": 0.0,
      "maxChunkBoundaryAbsJump": 0.01,
      "leftRightCorrelation": 0.99,
      "promptAdherence": 0.35,
      "embeddingSimilarityToReference": 0.9,
      "envelopePulseShare4To16Hz": 0.05,
  }
  _write_json(analysis, {
      "objectiveMetrics": metrics,
      "knownBadControls": [
          {"id": name, "rejected": True} for name in REQUIRED_KNOWN_BAD_CONTROLS
      ],
  })
  seeds = [11, 22, 33, 44, 55]
  rows = []
  for seed in seeds:
    role_to_label = {
        "reference": "sample-b",
        "candidate": "sample-c",
        "known-bad": "sample-a",
    }
    rows.append({"seed": seed, "roleToLabel": role_to_label})
    _write_json(vote_dir / f"seed-{seed}.json", {
        "clips": {
            "sample-a": {"verdict": "fail"},
            "sample-b": {"verdict": "pass"},
            "sample-c": {"verdict": "pass"},
        },
        "comparisons": {
            "sample-c_vs_sample-b": {"same_quality_class": seed != 55}
        },
    })
  _write_json(lineups, {"seeds": seeds, "lineups": rows})
  _write_json(g2_manifest, {
      "protocol": {
          "prompt": "warm ambient texture",
          "topK": 40,
          "temperature": 1.0,
          "trajectoryRefreshSeconds": 10.0,
      },
      "maxUnderruns": 0,
      "maxDropped": 0,
  })
  _write_json(g2_report, {"passed": True})
  args = argparse.Namespace(
      analysis=analysis,
      lineups_manifest=lineups,
      vote_dir=vote_dir,
      g2_manifest=g2_manifest,
      g2_report=g2_report,
  )
  manifest = build(args)
  assert verify_g3(manifest)["passed"] is True
  assert [vote["candidatePass"] for vote in manifest["blindAutomatedVotes"]] == [
      True,
      True,
      True,
      True,
      False,
  ]
  assert manifest["protocol"]["trajectoryRefreshSeconds"] == 10.0
