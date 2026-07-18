#!/usr/bin/env python3
"""Normalize a matched 600 s compute-policy control without requiring a WAV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_system_paper_soak_manifest import (
    AUDIO_RATE_HZ,
    DEVICE_METADATA,
    EFFECTIVE_FRAME_DEFINITION,
    FRAME_RATE_HZ,
    _continuous_play_seconds,
    _event_time,
    _events,
    _minute_bins,
    _parse_log,
    _percentile,
    _sha256,
    _thermal_timeline,
)


def build(args: argparse.Namespace) -> dict:
  rows, reservoir_start, protocol = _parse_log(args.run_log)
  events = _events(args.event_trace)
  generation_start = _event_time(events, "event=generationStarted")
  generation_stop = _event_time(events, "event=generationStopped", last=True)
  audio_start = _event_time(events, "event=audioStarted")
  reservoir_started = _event_time(events, "event=reservoirStarted")
  measured_seconds = generation_stop - generation_start
  if measured_seconds <= 0:
    raise ValueError("generation stop must follow generation start")
  effective_costs = [
      sum(
          float(row[name])
          for name in ("temporalMs", "depthMs", "samplingMs", "decoderMs")
      ) / FRAME_RATE_HZ
      for row in rows
  ]
  final = rows[-1]
  reservoir_end = int(final["fill"])
  return {
      "schema": "mrt2-system-paper-control-soak-v1",
      "device": DEVICE_METADATA[args.device],
      "controlRole": "matched-temporal-cpuAndGPU-policy",
      "requestedPolicyIsPlacementProof": False,
      "foreground": True,
      "screenOn": True,
      "measuredWindowSeconds": measured_seconds,
      "generatedAudioSeconds": len(rows),
      "pulledAudioSeconds": int(final["pulled"]) / AUDIO_RATE_HZ,
      "generationRate": len(rows) / measured_seconds,
      "maxUnderruns": max(int(row["underruns"]) for row in rows),
      "maxDropped": max(int(row["dropped"]) for row in rows),
      "effectiveFrameDefinition": EFFECTIVE_FRAME_DEFINITION,
      "effectiveFrameCount": len(rows) * FRAME_RATE_HZ,
      "p50EffectiveFrameMs": _percentile(effective_costs, 50),
      "p90EffectiveFrameMs": _percentile(effective_costs, 90),
      "p99EffectiveFrameMs": _percentile(effective_costs, 99),
      "reservoirStartFrames": reservoir_start,
      "reservoirEndFrames": reservoir_end,
      "reservoirSlopeFramesPerSecond": (
          reservoir_end - reservoir_start
      ) / measured_seconds,
      "primeSeconds": audio_start - reservoir_started,
      "continuousPlaySecondsBeforeFirstUnderrun": _continuous_play_seconds(
          rows, events, audio_start, measured_seconds
      ),
      "thermalTimeline": _thermal_timeline(
          rows, events, generation_start, measured_seconds
      ),
      "minuteBins": _minute_bins(
          rows, events, generation_start, measured_seconds
      ),
      "protocol": {
          "prompt": "warm ambient texture",
          "temperature": protocol["temperature"],
          "topK": protocol["topK"],
          "temporalMode": "streamingCarry",
          "trajectoryRefreshSeconds": protocol["trajectoryRefreshSeconds"],
          "computeUnits": protocol["computeUnits"],
      },
      "artifactSha256": {
          "runtime-log": _sha256(args.run_log),
          "event-trace": _sha256(args.event_trace),
      },
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", choices=sorted(DEVICE_METADATA), required=True)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--event-trace", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  manifest = build(args)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
