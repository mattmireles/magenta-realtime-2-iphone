#!/usr/bin/env python3
"""Verify the reviewer-motivated crossover and corrected A17 device receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGGREGATE = ROOT / "validation/results/system-paper/crossover/aggregate.json"
DEFAULT_DEVICE = (
    ROOT
    / "validation/results/system-paper/a17pro/context12/context12-soak-manifest.json"
)
DEFAULT_SEED = ROOT / "validation/results/system-paper/crossover/seed-20260718.json"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_for_report(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _check(actual: Any, passed: bool, requirement: str) -> dict[str, Any]:
  return {
      "actual": actual,
      "passed": bool(passed),
      "requirement": requirement,
  }


def build(aggregate_path: Path, device_path: Path, seed_path: Path) -> dict[str, Any]:
  aggregate = json.loads(aggregate_path.read_text())
  device = json.loads(device_path.read_text())
  seed = json.loads(seed_path.read_text())
  if aggregate.get("schema") != "mrt2-long-horizon-crossover-aggregate-v1":
    raise ValueError("unexpected crossover aggregate schema")
  if device.get("schema") != "mrt2-system-paper-context12-device-run-v1":
    raise ValueError("unexpected corrected-device schema")
  if seed.get("schema") != "mrt2-long-horizon-crossover-analysis-v1":
    raise ValueError("unexpected seed report schema")

  effects = aggregate["effects"]
  stateless = effects["statelessWindowingAndLegacyDspVsStreamingMlx"]
  graph = effects["coremlGraphVsMlxGraphAtLegacyDsp"]
  context = effects["context12InterventionAtCoremlGraph"]
  context_graph = effects["coremlGraphVsMlxGraphAtContext12"]
  probe = aggregate["decoderContextProbe"]["arms"]
  runtime = device["runtime"]
  protocol = device["protocol"]
  audio = device["audio"]
  reference_count = seed["cells"]["mlxTokens_mlxDecoder"]["windowsOverFrozenLimit"]

  checks = {
      "threeIndependentSeeds": _check(
          aggregate["replications"],
          aggregate["replications"] == 3 and len(set(aggregate["seeds"])) == 3,
          "exactly three unique seed reports",
      ),
      "fullHorizonPerArm": _check(
          aggregate["durationSecondsPerArm"],
          aggregate["durationSecondsPerArm"] == 600,
          "600 seconds for every crossover arm",
      ),
      "statelessEffectPositiveEveryWindow": _check(
          {
              "medianSeedMean": stateless["medianSeedMean"],
              "minimumSeedMean": stateless["minSeedMean"],
              "positiveWindows": stateless["positiveWindows"],
              "windows": stateless["windows"],
          },
          stateless["medianSeedMean"] > 0.015
          and stateless["minSeedMean"] > 0.0
          and stateless["positiveWindows"] == stateless["windows"] == 60,
          "effect > 0.015 median seed mean and positive in all 60 windows",
      ),
      "coremlGraphTooSmallToExplainDefect": _check(
          {
              "medianSeedMean": graph["medianSeedMean"],
              "maximumSeedMean": graph["maxSeedMean"],
          },
          abs(graph["medianSeedMean"]) < 0.001 and abs(graph["maxSeedMean"]) < 0.001,
          "absolute graph-only effect < 0.001 for median and worst seed mean",
      ),
      "contextInterventionNegativeEveryWindow": _check(
          {
              "medianSeedMean": context["medianSeedMean"],
              "maximumSeedMean": context["maxSeedMean"],
              "positiveWindows": context["positiveWindows"],
              "windows": context["windows"],
          },
          context["medianSeedMean"] < -0.015
          and context["maxSeedMean"] < 0.0
          and context["positiveWindows"] == 0
          and context["windows"] == 60,
          "effect < -0.015 median seed mean and negative in all 60 windows",
      ),
      "coremlGraphRemainsMatchedWithContext": _check(
          context_graph["medianSeedMean"],
          abs(context_graph["medianSeedMean"]) < 0.001,
          "absolute graph-only effect with context < 0.001",
      ),
      "contextProbeRecoversTensor": _check(
          {
              "correlationAt0": probe["0"]["correlation"],
              "correlationAt12": probe["12"]["correlation"],
              "maxAbsoluteErrorAt12": probe["12"]["maxAbsoluteError"],
          },
          probe["0"]["correlation"] < 0.2
          and probe["12"]["correlation"] > 0.999999999
          and probe["12"]["maxAbsoluteError"] < 0.001,
          "12-frame context correlation > 0.999999999 and max error < 0.001",
      ),
      "correctedDeviceSustainsPlayback": _check(
          {
              "pcmCaptureSeconds": protocol["pcmCaptureSeconds"],
              "nominalProducerRate": runtime["nominalProducerRate"],
              "maxUnderruns": runtime["maxUnderruns"],
              "maxDroppedFrames": runtime["maxDroppedFrames"],
          },
          protocol["pcmCaptureSeconds"] >= 600.0
          and runtime["nominalProducerRate"] >= 1.0
          and runtime["maxUnderruns"] == 0
          and runtime["maxDroppedFrames"] == 0,
          "at least 600 seconds, producer rate >= 1, zero underruns and drops",
      ),
      "correctedDeviceAudioIsContinuous": _check(
          {
              "finiteRatio": audio["finiteRatio"],
              "clippedSampleRatio": audio["clippedSampleRatio"],
              "longestNearZeroRunSeconds": audio["longestNearZeroRunSeconds"],
          },
          audio["finiteRatio"] == 1.0
          and audio["clippedSampleRatio"] <= 1e-5
          and audio["longestNearZeroRunSeconds"] <= 1.0 / 48_000.0,
          "finite, clipping <= 1e-5, and no near-zero run longer than one sample",
      ),
      "deviceMatchesStatefulSeedDiagnostic": _check(
          {
              "device": audio["windowsOverOriginalDiagnosticLimit"],
              "statefulMlx": reference_count,
              "windows": audio["windowCount"],
          },
          audio["windowsOverOriginalDiagnosticLimit"] == reference_count
          and audio["windowCount"] == 20,
          "device and stateful MLX have the same prompt-specific count over 20 windows",
      ),
  }
  passed = all(check["passed"] for check in checks.values())
  return {
      "schema": "mrt2-system-paper-revision-verdict-v1",
      "outcome": "pass" if passed else "fail",
      "inputs": {
          "aggregate": {
              "path": _path_for_report(aggregate_path),
              "sha256": _sha256(aggregate_path),
          },
          "device": {"path": _path_for_report(device_path), "sha256": _sha256(device_path)},
          "principalSeed": {
              "path": _path_for_report(seed_path),
              "sha256": _sha256(seed_path),
          },
      },
      "checks": checks,
      "interpretation": (
          "The measured excess follows stateless decoder windowing, and a "
          "12-frame causal-context intervention removes it on the fixed-token "
          "crossover and the physical A17 Pro run."
      ),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
  parser.add_argument("--device", type=Path, default=DEFAULT_DEVICE)
  parser.add_argument("--principal-seed", type=Path, default=DEFAULT_SEED)
  parser.add_argument("--output-json", type=Path)
  args = parser.parse_args()
  report = build(args.aggregate, args.device, args.principal_seed)
  rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
  if args.output_json:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(rendered)
  print(rendered, end="")
  if report["outcome"] != "pass":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
