#!/usr/bin/env python3
"""Aggregate seed-level MRT2 crossover reports without pooling away seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  digest.update(path.read_bytes())
  return digest.hexdigest()


def _aggregate_effect(
    reports: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
  selected = [(int(report["seed"]), selector(report)) for report in reports]
  seed_means = np.asarray([effect["mean"] for _, effect in selected], dtype=np.float64)
  total_windows = int(sum(effect["windows"] for _, effect in selected))
  return {
      "perSeed": [
          {
              "seed": seed,
              "mean": float(effect["mean"]),
              "median": float(effect["median"]),
              "positiveWindows": int(effect["positiveWindows"]),
              "windows": int(effect["windows"]),
          }
          for seed, effect in selected
      ],
      "medianSeedMean": float(np.median(seed_means)),
      "minSeedMean": float(np.min(seed_means)),
      "maxSeedMean": float(np.max(seed_means)),
      "pooledMeanOfSeedMeans": float(np.mean(seed_means)),
      "positiveWindows": int(sum(effect["positiveWindows"] for _, effect in selected)),
      "windows": total_windows,
  }


def build(paths: list[Path], context_probe: Path) -> dict[str, Any]:
  reports = [json.loads(path.read_text()) for path in paths]
  if len(reports) < 2:
    raise ValueError("at least two seed reports are required")
  if any(report.get("schema") != "mrt2-long-horizon-crossover-analysis-v1" for report in reports):
    raise ValueError("unexpected crossover report schema")
  reports.sort(key=lambda report: int(report["seed"]))
  seeds = [int(report["seed"]) for report in reports]
  if len(seeds) != len(set(seeds)):
    raise ValueError("crossover seeds must be unique")
  probe = json.loads(context_probe.read_text())
  if probe.get("schema") != "mrt2-decoder-context-probe-v1":
    raise ValueError("unexpected decoder context probe schema")

  effects = {
      "tokenSourceAtStreamingMlxDecoder": _aggregate_effect(
          reports,
          lambda report: report["windowPulseFactorialEffects"]
          ["coremlTokenEffectAtMlxDecoder"],
      ),
      "statelessWindowingAndLegacyDspVsStreamingMlx": _aggregate_effect(
          reports,
          lambda report: report["h3SplitWindowPulseEffects"]
          ["mlxWindowingAndProductionDspVsStreamingMlx"],
      ),
      "coremlGraphVsMlxGraphAtLegacyDsp": _aggregate_effect(
          reports,
          lambda report: report["h3SplitWindowPulseEffects"]
          ["coremlGraphVsMlxGraphAtProductionDsp"],
      ),
      "trainedHannVsLegacyDspAtMlxGraph": _aggregate_effect(
          reports,
          lambda report: report["fixedDspWindowPulseEffects"]
          ["fixOnMlxTokensMlxGraph"],
      ),
      "context12MlxVsStreamingMlx": _aggregate_effect(
          reports,
          lambda report: report["context12WindowPulseEffects"]
          ["mlxContext12VsStreamingMlx"],
      ),
      "context12InterventionAtMlxGraph": _aggregate_effect(
          reports,
          lambda report: report["context12WindowPulseEffects"]
          ["mlxContext12VsNoContextAtFixedDsp"],
      ),
      "context12InterventionAtCoremlGraph": _aggregate_effect(
          reports,
          lambda report: report["context12WindowPulseEffects"]
          ["coremlContext12VsNoContextAtFixedDsp"],
      ),
      "coremlGraphVsMlxGraphAtContext12": _aggregate_effect(
          reports,
          lambda report: report["context12WindowPulseEffects"]
          ["coremlGraphVsMlxGraphAtContext12"],
      ),
  }

  threshold_counts = {
      "streamingMlx": {
          "overLimit": int(sum(
              report["cells"]["mlxTokens_mlxDecoder"]["windowsOverFrozenLimit"]
              for report in reports
          )),
          "windows": int(sum(
              report["cells"]["mlxTokens_mlxDecoder"]["windowCount"]
              for report in reports
          )),
      },
      "statelessCoremlLegacyDsp": {
          "overLimit": int(sum(
              report["cells"]["mlxTokens_coremlDecoder"]["windowsOverFrozenLimit"]
              for report in reports
          )),
          "windows": int(sum(
              report["cells"]["mlxTokens_coremlDecoder"]["windowCount"]
              for report in reports
          )),
      },
      "context12CoremlTrainedHann": {
          "overLimit": int(sum(
              report["context12Cells"]["mlxTokens_coremlContext12FixedDsp"]
              ["windowsOverFrozenLimit"]
              for report in reports
          )),
          "windows": int(sum(
              report["context12Cells"]["mlxTokens_coremlContext12FixedDsp"]["windowCount"]
              for report in reports
          )),
      },
  }
  # The 0.070 band was frozen for the original ambient-prompt gate. The
  # matched crossover treats it as a diagnostic count, not a universal music
  # acceptance boundary.
  return {
      "schema": "mrt2-long-horizon-crossover-aggregate-v1",
      "seeds": seeds,
      "replications": len(seeds),
      "durationSecondsPerArm": 600,
      "prompt": reports[0]["protocol"]["prompt"],
      "reportArtifacts": [
          {"path": str(path), "sha256": _sha256(path)} for path in paths
      ],
      "effects": effects,
      "diagnosticThresholdCounts": threshold_counts,
      "decoderContextProbe": {
          "path": str(context_probe),
          "sha256": _sha256(context_probe),
          "startTokenFrame": int(probe["startTokenFrame"]),
          "targetTokenFrames": int(probe["targetTokenFrames"]),
          "arms": probe["arms"],
      },
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("inputs", nargs="+", type=Path)
  parser.add_argument("--context-probe", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  report = build(args.inputs, args.context_probe)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
