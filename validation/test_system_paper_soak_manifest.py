"""Tests for the G2/G4 soak-manifest normalizer."""

from __future__ import annotations

import argparse
import json

from build_system_paper_soak_manifest import build
from verify_system_paper_gate import verify_g2, verify_g4


def _write_fixture(tmp_path, *, device: str):
  log = tmp_path / "run.log"
  event_trace = tmp_path / "events.json"
  wav = tmp_path / "capture.wav"
  g1 = tmp_path / "g1-report.json"
  lines = [
      "CFHOST autoStart warmups=3 runSeconds=610.0 computeUnits="
      "temporal=cpuAndNeuralEngine depth=cpuOnly decoder=cpuOnly "
      "trajectoryRefreshSeconds=10.0",
      "CFHOST event=promptControlChanged temperature=1.0 topK=40",
      "CFHOST event=reservoirCompleted availableFrames=96000 iterations=2",
  ]
  events = [
      {"elapsedSeconds": 8.0, "message": "event=reservoirStarted targetFrames=48000"},
      {"elapsedSeconds": 9.5, "message": "event=audioStarted"},
      {"elapsedSeconds": 10.0, "message": "event=generationStarted"},
  ]
  for index in range(601):
    state = "nominal" if index < 300 else "serious"
    pulled = min(28_800_000, (index + 1) * 48_000)
    lines.append(
        "CFHOST event=generationIterationCompleted "
        f"fill={96000 + index} minFill=96000 underruns=0 dropped=0 "
        f"discarded=0 pushed={96000 + (index + 1) * 48000} pulled={pulled} "
        f"thermal={state} totalMs=900 temporalMs=240 depthMs=210 "
        "samplingMs=2 decoderMs=28 audioBackpressureMs=420 decoderCalls=1"
    )
    events.append({
        "elapsedSeconds": 10.5 + index,
        "message": "event=generationIterationCompleted fill=96000 minFill=96000",
    })
  events.append({"elapsedSeconds": 610.1, "message": "event=generationStopped"})
  log.write_text("\n".join(lines) + "\n")
  event_trace.write_text(json.dumps({"events": events}))
  wav.write_bytes(
      b"RIFF" + (36 + 16).to_bytes(4, "little") + b"WAVE"
      + b"fmt " + (16).to_bytes(4, "little")
      + (3).to_bytes(2, "little") + (2).to_bytes(2, "little")
      + (48_000).to_bytes(4, "little") + (384_000).to_bytes(4, "little")
      + (8).to_bytes(2, "little") + (32).to_bytes(2, "little")
      + b"data" + (16).to_bytes(4, "little") + (b"\0" * 16)
  )
  g1.write_text("{}\n")
  return argparse.Namespace(
      device=device,
      run_log=log,
      event_trace=event_trace,
      wav=wav,
      g1_report=g1,
      decoder_weight=None,
  )


def test_a17_manifest_passes_g2(tmp_path):
  manifest = build(_write_fixture(tmp_path, device="a17pro"))
  assert verify_g2(manifest)["passed"] is True
  assert manifest["effectiveFrameCount"] == 15_025
  assert manifest["thermalTimeline"][0]["elapsedSeconds"] == 0.0
  assert manifest["thermalTimeline"][-1]["elapsedSeconds"] >= 600.0
  assert manifest["minuteBins"][0]["p99EffectiveFrameMs"] < 40.0
  assert manifest["minuteBins"][-1]["endReservoirFrames"] > 0


def test_a14_manifest_selects_native_real_time(tmp_path):
  manifest = build(_write_fixture(tmp_path, device="a14"))
  assert manifest["outcome"] == "native-real-time"
  assert manifest["evidence"]["protocol"]["temperature"] == 1.0
  assert manifest["evidence"]["protocol"]["trajectoryRefreshSeconds"] == 10.0
  assert manifest["evidence"]["protocol"]["computeUnits"]["decoder"] == "cpuOnly"
  assert verify_g4(manifest)["passed"] is True
