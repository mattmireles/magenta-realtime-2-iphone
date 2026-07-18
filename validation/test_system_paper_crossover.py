"""Tests for the long-horizon token-by-decoder crossover analyzer."""

from __future__ import annotations

import argparse
import json

import numpy as np
import soundfile as sf

from analyze_system_paper_crossover import SAMPLE_RATE_HZ, _longest_true_run, analyze_wav, build
from aggregate_system_paper_crossover import _aggregate_effect
from verify_system_paper_revision import (
    DEFAULT_AGGREGATE,
    DEFAULT_DEVICE,
    DEFAULT_SEED,
    build as verify_revision,
)


def test_longest_true_run_handles_edges():
  assert _longest_true_run(np.array([], dtype=bool)) == 0
  assert _longest_true_run(np.array([True, True, False, True])) == 2
  assert _longest_true_run(np.array([False, True, True, True])) == 3


def test_audio_analysis_detects_envelope_pulse_and_dropout(tmp_path):
  seconds = 30
  time = np.arange(seconds * SAMPLE_RATE_HZ, dtype=np.float64) / SAMPLE_RATE_HZ
  carrier = np.sin(2 * np.pi * 440 * time)
  envelope = 0.2 + 0.15 * (1 + np.sin(2 * np.pi * 8 * time))
  mono = (carrier * envelope).astype(np.float32)
  stereo = np.stack([mono, mono * 0.99], axis=1)
  stereo[1000:5800] = 0
  wav = tmp_path / "pulse.wav"
  sf.write(wav, stereo, SAMPLE_RATE_HZ, subtype="FLOAT")
  report = analyze_wav(wav)
  assert report["windowCount"] == 1
  assert report["tailPulseShare4To16Hz"] > 0.9
  assert report["windowsOverFrozenLimit"] == 1
  assert report["longestNearZeroRunSeconds"] >= 0.1


def test_token_summary_contract_is_json_serializable(tmp_path):
  summary = {
      "frames": 15001,
      "windows": [
          {"frames": 750, "startSeconds": 0.0},
          {"frames": 1, "startSeconds": 600.0},
      ],
  }
  path = tmp_path / "summary.json"
  path.write_text(json.dumps(summary))
  assert json.loads(path.read_text())["frames"] == 15001


def test_context_arms_must_be_supplied_as_a_pair(tmp_path):
  token_summary = tmp_path / "tokens.json"
  token_summary.write_text(json.dumps({
      "frames": 15001,
      "windows": [{"frames": 750, "startSeconds": 0.0}],
  }))
  time = np.arange(30 * SAMPLE_RATE_HZ, dtype=np.float64) / SAMPLE_RATE_HZ
  mono = (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
  wav = tmp_path / "audio.wav"
  sf.write(wav, np.stack([mono, mono], axis=1), SAMPLE_RATE_HZ, subtype="FLOAT")
  args = argparse.Namespace(
      seed=1,
      prompt="test",
      mlx_token_summary=token_summary,
      coreml_token_summary=token_summary,
      mlx_mlx_wav=wav,
      mlx_coreml_wav=wav,
      coreml_mlx_wav=wav,
      coreml_coreml_wav=wav,
      mlx_stft_core_dsp_wav=None,
      mlx_stft_fixed_dsp_wav=None,
      mlx_coreml_fixed_dsp_wav=None,
      coreml_mlx_stft_fixed_dsp_wav=None,
      coreml_coreml_fixed_dsp_wav=None,
      mlx_context12_fixed_dsp_wav=wav,
      coreml_context12_fixed_dsp_wav=None,
  )
  try:
    build(args)
  except ValueError as error:
    assert "both 12-frame-context WAVs" in str(error)
  else:
    raise AssertionError("unpaired context arm must fail")


def test_aggregate_effect_preserves_seed_level_dispersion():
  reports = [
      {"seed": 2, "effect": {"mean": 0.2, "median": 0.1, "positiveWindows": 3, "windows": 4}},
      {"seed": 1, "effect": {"mean": 0.4, "median": 0.3, "positiveWindows": 4, "windows": 4}},
  ]
  aggregate = _aggregate_effect(reports, lambda report: report["effect"])
  assert np.isclose(aggregate["medianSeedMean"], 0.3)
  assert aggregate["minSeedMean"] == 0.2
  assert aggregate["maxSeedMean"] == 0.4
  assert aggregate["positiveWindows"] == 7
  assert aggregate["windows"] == 8


def test_checked_in_revision_evidence_passes():
  report = verify_revision(DEFAULT_AGGREGATE, DEFAULT_DEVICE, DEFAULT_SEED)
  assert report["outcome"] == "pass"
  assert all(check["passed"] for check in report["checks"].values())
