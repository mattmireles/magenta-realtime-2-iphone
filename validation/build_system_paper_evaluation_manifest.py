#!/usr/bin/env python3
"""Summarize repeated device campaigns into the public evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


FRAME_RATE_HZ = 25
GENERATION_PREFIX = "CFHOST event=generationIterationCompleted "
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9]*)=(?P<value>\S+)")
STAGES = ("temporalMs", "depthMs", "samplingMs", "decoderMs")


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


def _percentile(values: list[float], percentile: float) -> float:
  ordered = sorted(values)
  position = (len(ordered) - 1) * percentile / 100.0
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  fraction = position - lower
  return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _event_time(path: Path, prefix: str) -> float:
  events = _read_object(path).get("events")
  if not isinstance(events, list):
    raise ValueError(f"{path} has no events array")
  matches = [
      float(event["elapsedSeconds"])
      for event in events
      if isinstance(event, dict)
      and str(event.get("message", "")).startswith(prefix)
  ]
  if not matches:
    raise ValueError(f"{path} has no event beginning {prefix}")
  return matches[0]


def _summarize_run(log_path: Path, event_path: Path) -> dict[str, Any]:
  rows = []
  for line in log_path.read_text(errors="replace").splitlines():
    if GENERATION_PREFIX not in line:
      continue
    rows.append({
        match.group("key"): float(match.group("value"))
        for match in KEY_VALUE_RE.finditer(line)
        if match.group("key") in {*STAGES, "underruns", "dropped"}
    })
  if not rows:
    raise ValueError(f"{log_path} has no generation rows")
  values = {
      "effectiveFrameMs": [
          sum(row[name] for name in STAGES) / FRAME_RATE_HZ for row in rows
      ]
  }
  values.update({name: [row[name] / FRAME_RATE_HZ for row in rows] for name in STAGES})
  metrics = {
      name: {
          "p50": _percentile(samples, 50),
          "p90": _percentile(samples, 90),
          "p99": _percentile(samples, 99),
      }
      for name, samples in values.items()
  }
  auto_start = _event_time(event_path, "autoStart ")
  audio_start = _event_time(event_path, "event=audioStarted")
  return {
      "iterationCount": len(rows),
      "effectiveFrameCount": len(rows) * FRAME_RATE_HZ,
      "startupToFirstAudioSeconds": audio_start - auto_start,
      "maxUnderruns": int(max(row["underruns"] for row in rows)),
      "maxDropped": int(max(row["dropped"] for row in rows)),
      "latencyMsPerEffectiveFrame": metrics,
  }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
  paths = [
      ("startupToFirstAudioSeconds",),
      *[
          ("latencyMsPerEffectiveFrame", stage, percentile)
          for stage in ("effectiveFrameMs", *STAGES)
          for percentile in ("p50", "p90", "p99")
      ],
  ]
  result: dict[str, Any] = {}
  for path in paths:
    samples = []
    for run in runs:
      value: Any = run
      for key in path:
        value = value[key]
      samples.append(float(value))
    key = ".".join(path)
    q1 = _percentile(samples, 25)
    q3 = _percentile(samples, 75)
    result[key] = {
        "runValues": samples,
        "median": statistics.median(samples),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }
  return result


def _parse_cell(value: str) -> tuple[str, Path]:
  label, separator, raw_path = value.partition("=")
  if not separator or not label:
    raise argparse.ArgumentTypeError("expected LABEL=CAMPAIGN_DIRECTORY")
  return label, Path(raw_path)


def build(args: argparse.Namespace) -> dict[str, Any]:
  cells = []
  all_artifacts: dict[str, str] = {}
  for label, campaign_dir in args.cell:
    campaign_path = campaign_dir / "campaign-manifest.json"
    campaign = _read_object(campaign_path)
    run_rows = campaign.get("runs")
    if not isinstance(run_rows, list) or len(run_rows) < 5:
      raise ValueError(f"{label} requires at least five runs")
    runs = []
    for run_row in run_rows:
      if not isinstance(run_row, dict):
        raise ValueError(f"{label} run rows must be objects")
      run_id = str(run_row["runId"])
      log_path = campaign_dir / run_id / "console.log"
      event_path = campaign_dir / run_id / "events.json"
      summary = _summarize_run(log_path, event_path)
      summary["runId"] = run_id
      summary["artifactSha256"] = {
          "console": _sha256(log_path),
          "events": _sha256(event_path),
      }
      runs.append(summary)
      all_artifacts[f"{label}/{run_id}/console.log"] = _sha256(log_path)
      all_artifacts[f"{label}/{run_id}/events.json"] = _sha256(event_path)
    all_artifacts[f"{label}/campaign-manifest.json"] = _sha256(campaign_path)
    cells.append({
        "label": label,
        "device": campaign["device"],
        "protocol": campaign["protocol"],
        "runCount": len(runs),
        "runs": runs,
        "dispersion": _aggregate(runs),
    })
  return {
      "schema": "mrt2-system-paper-evaluation-v1",
      "effectiveFrameDefinition": "temporal+depth+sampling+decoder/decoded_frames",
      "dispersionContract": "median and IQR across independent process runs",
      "cells": cells,
      "artifactSha256": all_artifacts,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cell", action="append", type=_parse_cell, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  manifest = build(args)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
