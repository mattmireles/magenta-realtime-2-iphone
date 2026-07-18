"""Tests for repeated-run evaluation normalization."""

from __future__ import annotations

import argparse
import json

from build_system_paper_evaluation_manifest import build


def test_evaluation_manifest_reports_five_run_dispersion(tmp_path):
  campaign = tmp_path / "campaign"
  campaign.mkdir()
  runs = []
  for index in range(5):
    run_id = f"run-{index + 1:02d}"
    run_dir = campaign / run_id
    run_dir.mkdir()
    lines = []
    for _ in range(3):
      lines.append(
          "CFHOST event=generationIterationCompleted "
          f"temporalMs={250 + index} depthMs=200 samplingMs=2 decoderMs=25 "
          "underruns=0 dropped=0"
      )
    (run_dir / "console.log").write_text("\n".join(lines) + "\n")
    (run_dir / "events.json").write_text(json.dumps({
        "events": [
            {"elapsedSeconds": 0.1, "message": "autoStart warmups=3"},
            {"elapsedSeconds": 4.1 + index, "message": "event=audioStarted"},
        ]
    }))
    runs.append({"runId": run_id})
  (campaign / "campaign-manifest.json").write_text(json.dumps({
      "device": "test-device",
      "protocol": {"computeUnits": {"temporal": "cpuAndNeuralEngine"}},
      "runs": runs,
  }))
  manifest = build(argparse.Namespace(cell=[("test-cell", campaign)]))
  cell = manifest["cells"][0]
  assert cell["runCount"] == 5
  assert cell["dispersion"]["startupToFirstAudioSeconds"]["median"] == 6.0
  assert cell["dispersion"]["latencyMsPerEffectiveFrame.temporalMs.p99"][
      "runValues"
  ] == [10.0, 10.04, 10.08, 10.12, 10.16]
  assert len(manifest["artifactSha256"]) == 11
