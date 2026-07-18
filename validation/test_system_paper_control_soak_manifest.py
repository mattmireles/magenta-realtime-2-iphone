"""Tests for matched compute-policy control normalization."""

from build_system_paper_control_soak_manifest import build
from test_system_paper_soak_manifest import _write_fixture


def test_control_manifest_retains_failure_surfaces_and_policy(tmp_path):
  args = _write_fixture(tmp_path, device="a14")
  manifest = build(args)

  assert manifest["schema"] == "mrt2-system-paper-control-soak-v1"
  assert manifest["requestedPolicyIsPlacementProof"] is False
  assert manifest["effectiveFrameCount"] == 15_025
  assert manifest["maxUnderruns"] == 0
  assert manifest["protocol"]["computeUnits"]["temporal"] == "cpuAndNeuralEngine"
  assert "pcm-capture" not in manifest["artifactSha256"]
