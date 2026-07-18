#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalize private Crossfade soak evidence into public G2/G4 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any


GENERATION_PREFIX = "CFHOST event=generationIterationCompleted "
RESERVOIR_RE = re.compile(
    r"CFHOST event=reservoirCompleted availableFrames=(?P<frames>\d+) "
    r"iterations=(?P<iterations>\d+)"
)
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9]*)=(?P<value>\S+)")
COMPUTE_UNITS_RE = re.compile(
    r"computeUnits=temporal=(?P<temporal>\S+) "
    r"depth=(?P<depth>\S+) decoder=(?P<decoder>\S+)"
)
TRAJECTORY_REFRESH_RE = re.compile(
    r"trajectoryRefreshSeconds=(?P<seconds>[0-9]+(?:\.[0-9]+)?)"
)
FRAME_RATE_HZ = 25
AUDIO_RATE_HZ = 48_000
EFFECTIVE_FRAME_DEFINITION = "temporal+depth+sampling+decoder/decoded_frames"
DEVICE_METADATA = {
    "a17pro": {
        "name": "Commas",
        "marketingName": "iPhone 15 Pro Max",
        "modelIdentifier": "iPhone16,2",
        "chip": "A17 Pro",
        "osVersion": "26.5.2",
    },
    "a14": {
        "name": "Webcam",
        "marketingName": "iPhone 12 Pro",
        "modelIdentifier": "iPhone13,3",
        "chip": "A14 Bionic",
        "osVersion": "26.5",
    },
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _wav_duration_seconds(path: Path) -> float:
  with path.open("rb") as handle:
    if handle.read(4) != b"RIFF":
      raise ValueError(f"{path} is not a RIFF WAV")
    handle.seek(8)
    if handle.read(4) != b"WAVE":
      raise ValueError(f"{path} is not a WAVE file")
    byte_rate: int | None = None
    data_bytes: int | None = None
    while chunk_header := handle.read(8):
      if len(chunk_header) != 8:
        break
      chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
      if chunk_id == b"fmt ":
        payload = handle.read(chunk_size)
        if len(payload) < 12:
          raise ValueError(f"{path} has a truncated fmt chunk")
        byte_rate = int(struct.unpack_from("<I", payload, 8)[0])
      elif chunk_id == b"data":
        data_bytes = chunk_size
        handle.seek(chunk_size, 1)
      else:
        handle.seek(chunk_size, 1)
      if chunk_size % 2:
        handle.seek(1, 1)
      if byte_rate is not None and data_bytes is not None:
        break
  if not byte_rate or data_bytes is None:
    raise ValueError(f"{path} has no usable fmt/data chunks")
  return data_bytes / byte_rate


def _parse_value(value: str) -> int | float | str:
  try:
    return float(value) if "." in value else int(value)
  except ValueError:
    return value


def _parse_log(path: Path) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  reservoir_frames: int | None = None
  compute_units: dict[str, str] | None = None
  trajectory_refresh_seconds: float | None = None
  sampling: dict[str, Any] | None = None
  for line in path.read_text(errors="replace").splitlines():
    if reservoir_frames is None and (match := RESERVOIR_RE.search(line)):
      reservoir_frames = int(match.group("frames"))
    if compute_units is None and (match := COMPUTE_UNITS_RE.search(line)):
      compute_units = {
          "temporal": match.group("temporal"),
          "depth": match.group("depth"),
          "decoder": match.group("decoder"),
      }
    if trajectory_refresh_seconds is None and (
        match := TRAJECTORY_REFRESH_RE.search(line)
    ):
      trajectory_refresh_seconds = float(match.group("seconds"))
    if sampling is None and "CFHOST event=promptControlChanged " in line:
      values = {
          match.group("key"): _parse_value(match.group("value"))
          for match in KEY_VALUE_RE.finditer(line)
      }
      sampling = {
          "temperature": float(values["temperature"]),
          "topK": int(values["topK"]),
      }
    if GENERATION_PREFIX not in line:
      continue
    row = {
        match.group("key"): _parse_value(match.group("value"))
        for match in KEY_VALUE_RE.finditer(line)
    }
    rows.append(row)
  if reservoir_frames is None:
    raise ValueError(f"no reservoir completion in {path}")
  if not rows:
    raise ValueError(f"no generation iterations in {path}")
  if compute_units is None:
    raise ValueError(f"no compute-unit declaration in {path}")
  if trajectory_refresh_seconds is None:
    raise ValueError(f"no trajectory-refresh declaration in {path}")
  if sampling is None:
    raise ValueError(f"no prompt-control declaration in {path}")
  return rows, reservoir_frames, {
      **sampling,
      "computeUnits": compute_units,
      "trajectoryRefreshSeconds": trajectory_refresh_seconds,
  }


def _events(path: Path) -> list[dict[str, Any]]:
  value = json.loads(path.read_text())
  events = value.get("events")
  if not isinstance(events, list):
    raise ValueError(f"{path} has no events array")
  return [event for event in events if isinstance(event, dict)]


def _event_time(events: list[dict[str, Any]], prefix: str, *, last: bool = False) -> float:
  matches = [
      float(event["elapsedSeconds"])
      for event in events
      if str(event.get("message", "")).startswith(prefix)
  ]
  if not matches:
    raise ValueError(f"missing event prefix: {prefix}")
  return matches[-1] if last else matches[0]


def _percentile(values: list[float], percentile: float) -> float:
  ordered = sorted(values)
  position = (len(ordered) - 1) * percentile / 100.0
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  fraction = position - lower
  return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _thermal_timeline(
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    generation_start: float,
    measured_seconds: float,
) -> list[dict[str, Any]]:
  iteration_times = [
      float(event["elapsedSeconds"]) - generation_start
      for event in events
      if str(event.get("message", "")).startswith(
          "event=generationIterationCompleted"
      )
  ]
  if len(iteration_times) != len(rows):
    raise ValueError(
        f"raw/event generation mismatch: {len(rows)} != {len(iteration_times)}"
    )
  timeline: list[dict[str, Any]] = []
  previous: str | None = None
  for elapsed, row in zip(iteration_times, rows):
    state = str(row.get("thermal", "unknown"))
    if state not in {"nominal", "fair", "serious", "critical"}:
      raise ValueError(f"unsupported thermal state: {state}")
    if state != previous:
      timeline.append({"elapsedSeconds": max(0.0, elapsed), "state": state})
      previous = state
  timeline[0]["elapsedSeconds"] = 0.0
  timeline.append({"elapsedSeconds": measured_seconds, "state": previous})
  return timeline


def _minute_bins(
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    generation_start: float,
    measured_seconds: float,
) -> list[dict[str, Any]]:
  iteration_times = [
      float(event["elapsedSeconds"]) - generation_start
      for event in events
      if str(event.get("message", "")).startswith(
          "event=generationIterationCompleted"
      )
  ]
  if len(iteration_times) != len(rows):
    raise ValueError(
        f"raw/event generation mismatch: {len(rows)} != {len(iteration_times)}"
    )
  buckets: dict[int, list[dict[str, Any]]] = {}
  for elapsed, row in zip(iteration_times, rows):
    index = min(int(max(0.0, elapsed) // 60), int(measured_seconds // 60))
    buckets.setdefault(index, []).append(row)
  result = []
  for index, bucket_rows in sorted(buckets.items()):
    effective_costs = [
        sum(
            float(row[name])
            for name in ("temporalMs", "depthMs", "samplingMs", "decoderMs")
        )
        / FRAME_RATE_HZ
        for row in bucket_rows
    ]
    result.append({
        "startSeconds": index * 60.0,
        "endSeconds": min((index + 1) * 60.0, measured_seconds),
        "generationIterations": len(bucket_rows),
        "p50EffectiveFrameMs": _percentile(effective_costs, 50),
        "p99EffectiveFrameMs": _percentile(effective_costs, 99),
        "minReservoirFrames": min(int(row["fill"]) for row in bucket_rows),
        "endReservoirFrames": int(bucket_rows[-1]["fill"]),
        "thermalStates": sorted({str(row["thermal"]) for row in bucket_rows}),
    })
  return result


def _continuous_play_seconds(
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    audio_start: float,
    measured_seconds: float,
) -> float:
  iteration_times = [
      float(event["elapsedSeconds"])
      for event in events
      if str(event.get("message", "")).startswith(
          "event=generationIterationCompleted"
      )
  ]
  if len(iteration_times) != len(rows):
    raise ValueError(
        f"raw/event generation mismatch: {len(rows)} != {len(iteration_times)}"
    )
  for elapsed, row in zip(iteration_times, rows):
    if int(row["underruns"]) > 0:
      return max(0.0, elapsed - audio_start)
  return measured_seconds


def build(args: argparse.Namespace) -> dict[str, Any]:
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
      sum(float(row[name]) for name in ("temporalMs", "depthMs", "samplingMs", "decoderMs"))
      / FRAME_RATE_HZ
      for row in rows
  ]
  final = rows[-1]
  reservoir_end = int(final["fill"])
  generated_seconds = len(rows)
  evidence = {
      "foreground": True,
      "screenOn": True,
      "measuredWindowSeconds": measured_seconds,
      "pulledAudioSeconds": int(final["pulled"]) / AUDIO_RATE_HZ,
      "pcmCaptureSeconds": _wav_duration_seconds(args.wav),
      "generatedAudioSeconds": generated_seconds,
      "generationRate": generated_seconds / measured_seconds,
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
      "startupToFirstAudioSeconds": audio_start,
      "continuousPlaySecondsBeforeFirstUnderrun": _continuous_play_seconds(
          rows, events, audio_start, measured_seconds
      ),
      "thermalTimeline": _thermal_timeline(
          rows, events, generation_start, measured_seconds
      ),
      "minuteBins": _minute_bins(
          rows, events, generation_start, measured_seconds
      ),
      "g1ReportSha256": _sha256(args.g1_report),
      "artifactSha256": {
          "runtime-log": _sha256(args.run_log),
          "event-trace": _sha256(args.event_trace),
          "pcm-capture": _sha256(args.wav),
      },
      "protocol": {
          "prompt": "warm ambient texture",
          "temperature": protocol["temperature"],
          "topK": protocol["topK"],
          "temporalMode": "streamingCarry",
          "trajectoryRefreshSeconds": protocol["trajectoryRefreshSeconds"],
          "computeUnits": protocol["computeUnits"],
          "modelSha256": {
              "decoder": _sha256(args.decoder_weight)
              if args.decoder_weight is not None
              else None,
          },
      },
      "runSummary": {
          "generationIterations": len(rows),
          "finalFillFrames": reservoir_end,
          "minObservedFillFrames": min(int(row["minFill"]) for row in rows),
          "finalPushedFrames": int(final["pushed"]),
          "finalPulledFrames": int(final["pulled"]),
          "thermalStates": sorted({str(row["thermal"]) for row in rows}),
      },
  }
  if args.device == "a17pro":
    return {
        "schema": "mrt2-system-paper-g2-v1",
        "device": DEVICE_METADATA[args.device],
        **evidence,
    }

  native = (
      evidence["p99EffectiveFrameMs"] < 40.0
      and evidence["generationRate"] >= 1.0
      and evidence["reservoirSlopeFramesPerSecond"] >= 0.0
      and evidence["maxUnderruns"] == 0
      and evidence["maxDropped"] == 0
  )
  bounded = not native and evidence["maxUnderruns"] > 0
  evidence["unqualifiedRealTimeClaimAllowed"] = native
  evidence["maximumStartReservoirSeconds"] = reservoir_start / AUDIO_RATE_HZ
  return {
      "schema": "mrt2-system-paper-g4-v1",
      "device": DEVICE_METADATA[args.device],
      "outcome": (
          "native-real-time"
          if native
          else "bounded-reservoir" if bounded else "reservoir-tier"
      ),
      "evidence": evidence,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", choices=sorted(DEVICE_METADATA), required=True)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--event-trace", type=Path, required=True)
  parser.add_argument("--wav", type=Path, required=True)
  parser.add_argument("--g1-report", type=Path, required=True)
  parser.add_argument("--decoder-weight", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  manifest = build(args)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
