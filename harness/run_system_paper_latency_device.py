#!/usr/bin/env python3
"""Capture independent fixed-protocol device runs for the system-paper matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


BUNDLE_ID = "com.transcendence.crossfade.CrossfadeRuntimeHost"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _refresh_host_arguments(seconds: float) -> list[str]:
  """Return the host flag for periodic refresh; zero uses the default off mode."""
  if seconds < 0:
    raise ValueError("trajectory refresh seconds must be nonnegative")
  if seconds == 0:
    return []
  return ["--trajectory-refresh-seconds", str(seconds)]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", required=True)
  parser.add_argument("--label", required=True)
  parser.add_argument("--out-root", type=Path, default=Path("Scratchpad/system_paper_latency"))
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--run-seconds", type=float, default=20.0)
  parser.add_argument("--temporal-compute-units", required=True)
  parser.add_argument("--depth-compute-units", default="cpuOnly")
  parser.add_argument("--decoder-compute-units", required=True)
  parser.add_argument("--trajectory-refresh-seconds", type=float, default=10.0)
  parser.add_argument("--start-reservoir-seconds", type=float, default=2.0)
  args = parser.parse_args()
  if args.repeats < 5:
    raise SystemExit("the frozen dispersion contract requires at least five repeats")
  if args.trajectory_refresh_seconds < 0:
    raise SystemExit("--trajectory-refresh-seconds must be nonnegative")

  timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  root = args.out_root / f"{timestamp}-{args.label}"
  root.mkdir(parents=True, exist_ok=False)
  runs = []
  for index in range(args.repeats):
    run_id = f"run-{index + 1:02d}"
    run_dir = root / run_id
    run_dir.mkdir()
    remote_events = f"{args.label}-{run_id}-events.json"
    command = [
        "/usr/bin/xcrun",
        "devicectl",
        "device",
        "process",
        "launch",
        "--device",
        args.device,
        "--terminate-existing",
        "--activate",
        "--console",
        "--timeout",
        str(max(180, int(args.run_seconds) + 120)),
        "--json-output",
        str(run_dir / "launch.json"),
        "--log-output",
        str(run_dir / "launch.log"),
        BUNDLE_ID,
        "--auto-start",
        "--warmups",
        "3",
        "--run-seconds",
        str(args.run_seconds),
        "--exit-after-auto-run",
        "--temporal-mode",
        "streamingCarry",
        "--start-reservoir-seconds",
        str(args.start_reservoir_seconds),
        "--audio-prime-chunks",
        "50",
        "--audio-capacity-chunks",
        "500",
        "--prompt",
        "warm ambient texture",
        "--top-k",
        "40",
        "--style-guidance",
        "3.1",
        "--note-guidance",
        "0.0",
        "--drum-guidance",
        "0.0",
        "--drums-enabled",
        "false",
        "--source-conditioning-resource",
        "warm.bin",
        "--capture-events-json",
        remote_events,
        "--temporal-compute-units",
        args.temporal_compute_units,
        "--depth-compute-units",
        args.depth_compute_units,
        "--decoder-compute-units",
        args.decoder_compute_units,
        "--temperature",
        "1.0",
        "--seed",
        "20260718",
    ]
    command.extend(_refresh_host_arguments(args.trajectory_refresh_seconds))
    console = run_dir / "console.log"
    with console.open("w") as stdout:
      completed = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
      raise SystemExit(f"{run_id} failed with exit {completed.returncode}; see {console}")
    event_trace = run_dir / "events.json"
    copy_command = [
        "/usr/bin/xcrun",
        "devicectl",
        "device",
        "copy",
        "from",
        "--device",
        args.device,
        "--domain-type",
        "appDataContainer",
        "--domain-identifier",
        BUNDLE_ID,
        "--source",
        f"Documents/{remote_events}",
        "--destination",
        str(event_trace),
        "--timeout",
        "120",
    ]
    completed = subprocess.run(copy_command, check=False)
    if completed.returncode:
      raise SystemExit(f"failed to pull events for {run_id}")
    runs.append({
        "runId": run_id,
        "consoleSha256": _sha256(console),
        "launchSha256": _sha256(run_dir / "launch.json"),
        "eventTraceSha256": _sha256(event_trace),
    })

  manifest = {
      "schema": "crossfade-system-paper-latency-campaign-v1",
      "device": args.device,
      "label": args.label,
      "repeats": args.repeats,
      "runSeconds": args.run_seconds,
      "protocol": {
          "temporalMode": "streamingCarry",
          "trajectoryRefreshSeconds": args.trajectory_refresh_seconds,
          "prompt": "warm ambient texture",
          "topK": 40,
          "temperature": 1.0,
          "seed": 20260718,
          "computeUnits": {
              "temporal": args.temporal_compute_units,
              "depth": args.depth_compute_units,
              "decoder": args.decoder_compute_units,
          },
      },
      "runs": runs,
  }
  (root / "campaign-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(root)


if __name__ == "__main__":
  main()
