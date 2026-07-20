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

"""Summarize CrossfadeRuntimeHost event-trace JSON proof artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


COMPILER_RE = re.compile(
    r'compilerApplied label="(?P<label>[^"]+)"(?: command="(?P<command>[^"]+)")? source="(?P<source>[^"]*)" '
    r"frames=(?P<frames>\d+) elapsedSeconds=(?P<elapsed>[0-9.]+)"
)
CONDUCTOR_RE = re.compile(
    r'conductorEvent label="(?P<label>[^"]+)" afterSeconds=(?P<after>[0-9.]+)(?: command="(?P<command>[^"]+)")?.*'
    r'sourceResource="(?P<source>[^"]*)"'
)
GENERATION_RE = re.compile(
    r"event=generationIterationCompleted fill=(?P<fill>\d+) minFill=(?P<min>\d+) "
    r"underruns=(?P<underruns>\d+) dropped=(?P<dropped>\d+)(?: discarded=(?P<discarded>\d+))?"
)
DISCARD_RE = re.compile(
    r"event=audioQueuedFramesDiscarded requestedTargetFrames=(?P<target>\d+) "
    r"(?:fadeInFrames=(?P<fade>\d+) )?"
    r"discardedFrames=(?P<discarded>\d+) before=(?P<before>\d+) after=(?P<after>\d+)"
)
CAPTURE_RE = re.compile(
    r'event=pcmCaptureCompleted path="(?P<path>[^"]+)" frames=(?P<frames>\d+) '
    r"maxFrames=(?P<max>\d+) peak=(?P<peak>[0-9.eE+-]+) rms=(?P<rms>[0-9.eE+-]+) "
    r"checksum=(?P<checksum>-?\d+)"
)
POST_RING_START_RE = re.compile(
    r'event=postRingCaptureStarted path="(?P<path>[^"]+)" maxFrames=(?P<max>\d+)'
)
POST_RING_COMPLETE_RE = re.compile(
    r'event=postRingCaptureCompleted path="(?P<path>[^"]+)" '
    r'frames=(?P<frames>\d+) maxFrames=(?P<max>\d+) '
    r'capturedFrames=(?P<captured>\d+) proofDrops=(?P<drops>\d+) '
    r'overflowEvents=(?P<overflows>\d+) peak=(?P<peak>[0-9.eE+-]+) '
    r'rms=(?P<rms>[0-9.eE+-]+) checksum=(?P<checksum>-?\d+)'
)
THERMAL_RE = re.compile(r"event=thermalStateChanged state=(?P<state>\w+)")
PROMPT_RE = re.compile(
    r'event=promptControlChanged .*notes=\[(?P<notes>[^\]]*)\] '
    r"drumsEnabled=(?P<drums>\w+) sourceFrames=(?P<source_frames>\d+)"
)
PLACEMENT_RE = re.compile(
    r"placementPolicy temporal=(?P<temporal>\w+) depth=(?P<depth>\w+) "
    r"decoder=(?P<decoder>\w+) aneResidencyProven=(?P<ane>\w+)"
)


def _event_time(event: dict[str, Any]) -> float:
  return float(event["elapsedSeconds"])


def summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
  """Return a compact proof summary from a host-written trace."""
  events = trace.get("events", [])
  compiler_applications: list[dict[str, Any]] = []
  conductor_events: list[dict[str, Any]] = []
  prompt_changes: list[dict[str, Any]] = []
  thermal_events: list[dict[str, Any]] = []
  audio_discards: list[dict[str, Any]] = []
  capture: dict[str, Any] | None = None
  post_ring_capture: dict[str, Any] | None = None
  post_ring_capture_started_trace_elapsed_seconds: float | None = None
  placement_policy: dict[str, Any] | None = None
  generation_iterations = 0
  final_underruns = 0
  final_dropped = 0
  min_fill_frames: int | None = None
  min_fill_target_frames: int | None = None

  for event in events:
    message = str(event.get("message", ""))
    elapsed = _event_time(event)
    if match := COMPILER_RE.search(message):
      compiler_applications.append({
          "label": match.group("label"),
          "command": match.group("command") or "applyControl",
          "source": match.group("source"),
          "frame_count": int(match.group("frames")),
          "elapsed_seconds": float(match.group("elapsed")),
          "trace_elapsed_seconds": elapsed,
      })
      continue
    if match := CONDUCTOR_RE.search(message):
      conductor_events.append({
          "label": match.group("label"),
          "command": match.group("command") or "applyControl",
          "scheduled_after_seconds": float(match.group("after")),
          "source": match.group("source"),
          "trace_elapsed_seconds": elapsed,
      })
      continue
    if match := PROMPT_RE.search(message):
      notes = [
          int(value.strip())
          for value in match.group("notes").split(",")
          if value.strip()
      ]
      prompt_changes.append({
          "notes": notes,
          "drums_enabled": match.group("drums") == "true",
          "source_frames": int(match.group("source_frames")),
          "trace_elapsed_seconds": elapsed,
      })
      continue
    if match := GENERATION_RE.search(message):
      generation_iterations += 1
      fill = int(match.group("fill"))
      min_fill_target_frames = int(match.group("min"))
      final_underruns = int(match.group("underruns"))
      final_dropped = int(match.group("dropped"))
      min_fill_frames = fill if min_fill_frames is None else min(min_fill_frames, fill)
      continue
    if match := DISCARD_RE.search(message):
      audio_discards.append({
          "requested_target_frames": int(match.group("target")),
          "requested_fade_in_frames": int(match.group("fade") or 0),
          "discarded_frames": int(match.group("discarded")),
          "available_frames_before": int(match.group("before")),
          "available_frames_after": int(match.group("after")),
          "trace_elapsed_seconds": elapsed,
      })
      continue
    if match := CAPTURE_RE.search(message):
      capture = {
          "path": match.group("path"),
          "frames": int(match.group("frames")),
          "max_frames": int(match.group("max")),
          "peak": float(match.group("peak")),
          "rms": float(match.group("rms")),
          "checksum": int(match.group("checksum")),
          "trace_elapsed_seconds": elapsed,
      }
      continue
    if match := POST_RING_START_RE.search(message):
      post_ring_capture_started_trace_elapsed_seconds = elapsed
      post_ring_capture = {
          "path": match.group("path"),
          "max_frames": int(match.group("max")),
          "started_trace_elapsed_seconds": elapsed,
      }
      continue
    if match := POST_RING_COMPLETE_RE.search(message):
      post_ring_capture = {
          "path": match.group("path"),
          "frames": int(match.group("frames")),
          "max_frames": int(match.group("max")),
          "captured_frames": int(match.group("captured")),
          "dropped_proof_frames": int(match.group("drops")),
          "overflow_events": int(match.group("overflows")),
          "peak": float(match.group("peak")),
          "rms": float(match.group("rms")),
          "checksum": int(match.group("checksum")),
          "started_trace_elapsed_seconds": post_ring_capture_started_trace_elapsed_seconds,
          "completed_trace_elapsed_seconds": elapsed,
      }
      continue
    if match := THERMAL_RE.search(message):
      thermal_events.append({
          "state": match.group("state"),
          "trace_elapsed_seconds": elapsed,
      })
      continue
    if match := PLACEMENT_RE.search(message):
      placement_policy = {
          "temporal_compute_units": match.group("temporal"),
          "depth_compute_units": match.group("depth"),
          "decoder_compute_units": match.group("decoder"),
          "ane_residency_proven": match.group("ane") == "true",
          "trace_elapsed_seconds": elapsed,
      }

  compiler_latencies = [item["elapsed_seconds"] for item in compiler_applications]
  max_compiler_latency = max(compiler_latencies) if compiler_latencies else None
  first_generation_time = next(
      (
          _event_time(event)
          for event in events
          if str(event.get("message", "")).startswith("event=generationStarted")
      ),
      None,
  )
  audio_started_time = next(
      (
          _event_time(event)
          for event in events
          if str(event.get("message", "")).startswith("event=audioStarted")
      ),
      None,
  )
  post_ring_audio_origin = None
  if post_ring_capture_started_trace_elapsed_seconds is not None:
    post_ring_audio_origin = max(
        post_ring_capture_started_trace_elapsed_seconds,
        audio_started_time or post_ring_capture_started_trace_elapsed_seconds,
    )
  stopped_time = next(
      (
          _event_time(event)
          for event in reversed(events)
          if str(event.get("message", "")).startswith("event=generationStopped")
      ),
      None,
  )
  control_response_latencies = _control_response_latencies(conductor_events, prompt_changes)
  _bind_control_stage_timestamps(
      conductor_events, compiler_applications, control_response_latencies)

  return {
      "schema": "crossfade-runtime-host-event-trace-summary-v1",
      "source_schema": trace.get("schema"),
      "created_at": trace.get("createdAt"),
      "event_count": len(events),
      "compiler_applications": compiler_applications,
      "max_compiler_latency_seconds": max_compiler_latency,
      "conductor_events": conductor_events,
      "prompt_changes": prompt_changes,
      "control_response_latencies": control_response_latencies,
      "generation_iterations": generation_iterations,
      "first_generation_trace_elapsed_seconds": first_generation_time,
      "audio_started_trace_elapsed_seconds": audio_started_time,
      "generation_stopped_trace_elapsed_seconds": stopped_time,
      "final_underruns": final_underruns,
      "final_dropped_frames": final_dropped,
      "min_fill_frames": min_fill_frames,
      "min_fill_target_frames": min_fill_target_frames,
      "capture": capture,
      "post_ring_capture": post_ring_capture,
      "post_ring_capture_started_trace_elapsed_seconds": (
          post_ring_capture_started_trace_elapsed_seconds),
      "post_ring_audio_origin_trace_elapsed_seconds": post_ring_audio_origin,
      "thermal_events": thermal_events,
      "audio_discards": audio_discards,
      "total_discarded_frames": sum(item["discarded_frames"] for item in audio_discards),
      "placement_policy": placement_policy,
      "passed_no_audio_starvation": final_underruns == 0 and final_dropped == 0,
}


def _bind_control_stage_timestamps(
  conductor_events: list[dict[str, Any]],
  compiler_applications: list[dict[str, Any]],
  control_response_latencies: list[dict[str, Any]],
) -> None:
  """Bind receipt, compile/apply, and event labels on one monotonic clock.

  Source compilation completes immediately before the synchronous
  `setPromptControl` call that emits `promptControlChanged`. The trace has no
  separate instruction-level timestamp between those operations, so both are
  conservatively reported at that application event. The compiler's own total
  duration remains available as a diagnostic rather than manufacturing a
  finer timestamp.
  """
  compiler_by_label = {row["label"]: row for row in compiler_applications}
  latency_by_label = {row["label"]: row for row in control_response_latencies}
  for event in conductor_events:
    received = float(event["trace_elapsed_seconds"])
    latency = latency_by_label.get(event["label"])
    applied = (
        float(latency["prompt_trace_elapsed_seconds"])
        if latency is not None else received)
    compiler = compiler_by_label.get(event["label"])
    event["received_trace_elapsed_seconds"] = received
    event["compiled_trace_elapsed_seconds"] = applied
    event["applied_trace_elapsed_seconds"] = applied
    event["compiler_reported_elapsed_seconds"] = (
        compiler.get("elapsed_seconds") if compiler is not None else None)


def _control_response_latencies(
  conductor_events: list[dict[str, Any]],
  prompt_changes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Match each conductor event to the next prompt-control change."""
  latencies: list[dict[str, Any]] = []
  for conductor_event in conductor_events:
    start = conductor_event["trace_elapsed_seconds"]
    next_prompt = next(
        (
            prompt
            for prompt in prompt_changes
            if prompt["trace_elapsed_seconds"] >= start
        ),
        None,
    )
    if next_prompt is None:
      continue
    latencies.append({
        "label": conductor_event["label"],
        "seconds": next_prompt["trace_elapsed_seconds"] - start,
        "conductor_trace_elapsed_seconds": start,
        "prompt_trace_elapsed_seconds": next_prompt["trace_elapsed_seconds"],
    })
  return latencies


def _write_markdown(summary: dict[str, Any], output_path: Path) -> None:
  lines = [
      "# Crossfade Event Trace Summary",
      "",
      f"Created at: `{summary.get('created_at')}`",
      f"Events: `{summary['event_count']}`",
      "",
      "## Compiler Applications",
      "",
      "| Label | Source | Frames | Compiler s | Trace s |",
      "| --- | --- | ---: | ---: | ---: |",
  ]
  for item in summary["compiler_applications"]:
    lines.append(
        f"| `{item['label']}` | `{item['source']}` | {item['frame_count']} | "
        f"{item['elapsed_seconds']:.6f} | {item['trace_elapsed_seconds']:.3f} |"
    )
  if summary["control_response_latencies"]:
    lines += [
        "",
        "## Control Response",
        "",
        "| Label | Event s | Prompt s | Response s |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in summary["control_response_latencies"]:
      lines.append(
          f"| `{item['label']}` | {item['conductor_trace_elapsed_seconds']:.3f} | "
          f"{item['prompt_trace_elapsed_seconds']:.3f} | {item['seconds']:.3f} |"
      )
  lines += [
      "",
      "## Audio Runtime",
      "",
      f"- Generation iterations: `{summary['generation_iterations']}`",
      f"- Final underruns: `{summary['final_underruns']}`",
      f"- Final dropped frames: `{summary['final_dropped_frames']}`",
      f"- Intentional discarded frames: `{summary.get('total_discarded_frames', 0)}`",
      f"- Min fill frames: `{summary['min_fill_frames']}`",
      f"- Min fill target frames: `{summary['min_fill_target_frames']}`",
      f"- No-starvation pass: `{summary['passed_no_audio_starvation']}`",
  ]
  if summary.get("audio_discards"):
    lines += [
        "",
        "## Audio Queue Discards",
        "",
        "| Trace s | Target frames | Before | After | Discarded |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["audio_discards"]:
      lines.append(
          f"| {item['trace_elapsed_seconds']:.3f} | {item['requested_target_frames']} "
          f"(fade {item.get('requested_fade_in_frames', 0)}) | "
          f"{item['available_frames_before']} | {item['available_frames_after']} | "
          f"{item['discarded_frames']} |"
      )
  if summary.get("capture"):
    capture = summary["capture"]
    lines += [
        "",
        "## Capture",
        "",
        f"- Path: `{capture['path']}`",
        f"- Frames: `{capture['frames']}` / `{capture['max_frames']}`",
        f"- Peak: `{capture['peak']:.6f}`",
        f"- RMS: `{capture['rms']:.6f}`",
        f"- Checksum: `{capture['checksum']}`",
    ]
  if summary["thermal_events"]:
    lines += ["", "## Thermal", ""]
    for item in summary["thermal_events"]:
      lines.append(f"- `{item['state']}` at `{item['trace_elapsed_seconds']:.3f}s`")
  if summary.get("placement_policy"):
    policy = summary["placement_policy"]
    lines += [
        "",
        "## Compute Placement Policy",
        "",
        f"- Temporal request: `{policy['temporal_compute_units']}`",
        f"- Depth request: `{policy['depth_compute_units']}`",
        f"- Decoder request: `{policy['decoder_compute_units']}`",
        f"- ANE residency proven: `{policy['ane_residency_proven']}`",
        "",
        "This policy is not runtime placement proof; attach Instruments, "
        "`MLComputePlan`, or powermetrics evidence before claiming ANE execution.",
    ]
  output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--trace", required=True, help="Host event trace JSON path.")
  parser.add_argument("--output-json", required=True, help="Path to write summary JSON.")
  parser.add_argument("--output-md", required=True, help="Path to write summary Markdown.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  trace = json.loads(Path(args.trace).read_text())
  summary = summarize_trace(trace)
  output_json = Path(args.output_json)
  output_md = Path(args.output_md)
  output_json.parent.mkdir(parents=True, exist_ok=True)
  output_md.parent.mkdir(parents=True, exist_ok=True)
  output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  _write_markdown(summary, output_md)
  print(f"Wrote {output_json}")
  print(f"Wrote {output_md}")


if __name__ == "__main__":
  main()
