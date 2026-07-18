#!/usr/bin/env python3
"""Normalize objective audio analysis and blind votes into the public G3 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


G3_SCHEMA = "mrt2-system-paper-g3-v1"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text())
  if not isinstance(value, dict):
    raise ValueError(f"{path} must contain a JSON object")
  return value


def _clip_pass(vote: dict[str, Any], label: str) -> bool:
  clips = vote.get("clips")
  if not isinstance(clips, dict) or not isinstance(clips.get(label), dict):
    raise ValueError(f"vote has no clip verdict for {label}")
  verdict = clips[label].get("verdict")
  if verdict not in {"pass", "fail"}:
    raise ValueError(f"vote verdict for {label} must be pass or fail")
  return verdict == "pass"


def _same_quality_class(
    vote: dict[str, Any], candidate_label: str, reference_label: str
) -> bool:
  comparisons = vote.get("comparisons")
  key = f"{candidate_label}_vs_{reference_label}"
  if not isinstance(comparisons, dict) or not isinstance(comparisons.get(key), dict):
    raise ValueError(f"vote has no comparison {key}")
  value = comparisons[key].get("same_quality_class")
  if not isinstance(value, bool):
    raise ValueError(f"vote comparison {key} must have a boolean same_quality_class")
  return value


def build(args: argparse.Namespace) -> dict[str, Any]:
  analysis = _read_object(args.analysis)
  lineups = _read_object(args.lineups_manifest)
  g2_manifest = _read_object(args.g2_manifest)
  seeds = lineups.get("seeds")
  lineup_rows = lineups.get("lineups")
  if not isinstance(seeds, list) or not isinstance(lineup_rows, list):
    raise ValueError("lineups manifest must contain seeds and lineups arrays")
  if len(seeds) != len(lineup_rows):
    raise ValueError("lineups seed and row counts differ")

  vote_paths: list[Path] = []
  votes: list[dict[str, Any]] = []
  for lineup in lineup_rows:
    if not isinstance(lineup, dict):
      raise ValueError("lineup rows must be objects")
    seed = int(lineup["seed"])
    vote_path = args.vote_dir / f"seed-{seed}.json"
    vote = _read_object(vote_path)
    vote_paths.append(vote_path)
    role_to_label = lineup.get("roleToLabel")
    if not isinstance(role_to_label, dict):
      raise ValueError(f"lineup {seed} has no roleToLabel")
    reference_label = str(role_to_label["reference"])
    candidate_label = str(role_to_label["candidate"])
    known_bad_label = str(role_to_label["known-bad"])
    known_good_pass = _clip_pass(vote, reference_label)
    known_bad_pass = _clip_pass(vote, known_bad_label)
    candidate_clip_pass = _clip_pass(vote, candidate_label)
    candidate_same_class = _same_quality_class(
        vote, candidate_label, reference_label
    )
    votes.append({
        "seed": seed,
        "knownGoodPass": known_good_pass,
        "knownBadPass": known_bad_pass,
        "controlsRankedCorrectly": known_good_pass and not known_bad_pass,
        "candidatePass": candidate_clip_pass and candidate_same_class,
        "candidateClipPass": candidate_clip_pass,
        "candidateSameQualityClass": candidate_same_class,
        "artifactSha256": _sha256(vote_path),
    })

  g2_protocol = g2_manifest.get("protocol")
  if not isinstance(g2_protocol, dict):
    raise ValueError("G2 manifest has no protocol object")
  artifacts = {
      "analysis.json": _sha256(args.analysis),
      "lineups-manifest.json": _sha256(args.lineups_manifest),
      "g2-manifest.json": _sha256(args.g2_manifest),
      "g2-report.json": _sha256(args.g2_report),
  }
  artifacts.update({
      f"vote-seed-{seed}.json": _sha256(path)
      for seed, path in zip(seeds, vote_paths)
  })
  return {
      "schema": G3_SCHEMA,
      "protocol": {
          "prompt": g2_protocol["prompt"],
          "topK": g2_protocol["topK"],
          "temperature": g2_protocol["temperature"],
          "trajectoryRefreshSeconds": g2_protocol.get(
              "trajectoryRefreshSeconds"
          ),
          "blindOrderSeeds": seeds,
      },
      "g2ReportSha256": _sha256(args.g2_report),
      "maxUnderruns": g2_manifest["maxUnderruns"],
      "maxDropped": g2_manifest["maxDropped"],
      "objectiveMetrics": analysis["objectiveMetrics"],
      "knownBadControls": analysis["knownBadControls"],
      "blindAutomatedVotes": votes,
      "artifactSha256": artifacts,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--analysis", type=Path, required=True)
  parser.add_argument("--lineups-manifest", type=Path, required=True)
  parser.add_argument("--vote-dir", type=Path, required=True)
  parser.add_argument("--g2-manifest", type=Path, required=True)
  parser.add_argument("--g2-report", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  manifest = build(args)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
