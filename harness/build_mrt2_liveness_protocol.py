#!/usr/bin/env python3
"""Freeze MRT2 unrefreshed-liveness fixtures, arms, analysis, and judge gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_REPO = REPO_ROOT.parent / "magenta-realtime-2-iphone"
EXPORTER = REPO_ROOT / "scripts" / "export_crossfade_source_conditioning.py"
DEFAULT_OUTPUT = REPO_ROOT / "Scratchpad" / "system_paper_liveness" / "protocol.json"
DEFAULT_FIXTURES = DEFAULT_OUTPUT.parent / "fixtures"
DEFAULT_CHECKPOINT = (
    Path.home() / "Documents" / "Magenta" / "magenta-rt-v2" / "checkpoints"
    / "mrt2_small.safetensors"
)
EXPECTED_WARM_SHA256 = "d4d510330a9e45089174d89b9c591360a5e1b111d56aa921c19a41020abbea11"
PROMPTS = (
    ("warm", "warm ambient texture"),
    ("solo-piano", "solo piano, slow and sparse, late night"),
    ("smooth-electronic", "smooth electronic"),
    ("detroit-techno", "Detroit TECHNO with heavy 909 drums, 128 BPM — dark & hypnotic!"),
)
SEEDS = (20_260_718, 271_828, 1_618_033)
LINEUP_SEEDS = {
    "warm": (50_101, 50_102, 50_103),
    "solo-piano": (50_201, 50_202, 50_203),
    "smooth-electronic": (50_301, 50_302, 50_303),
    "detroit-techno": (50_401, 50_402, 50_403),
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git_commit(root: Path) -> str:
  return subprocess.check_output(
      ["/usr/bin/git", "rev-parse", "HEAD"], cwd=root, text=True
  ).strip()


def _relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))
  except ValueError:
    return str(path.resolve())


def _compile_fixture(slug: str, prompt: str, fixtures_dir: Path) -> None:
  command = [
      sys.executable,
      str(EXPORTER),
      "--prompt", prompt,
      "--output-dir", str(fixtures_dir),
      "--output-stem", slug,
      "--temperature", "1.3",
      "--top-k", "40",
      "--cfg-musiccoca", "3.0",
      "--cfg-notes", "1.0",
      "--cfg-drums", "1.0",
      "--drums-enabled", "masked",
      "--intensity", "0.5",
      "--seed", "20260718",
  ]
  subprocess.run(command, cwd=REPO_ROOT, check=True)


def _fixture_record(slug: str, prompt: str, fixtures_dir: Path) -> dict[str, Any]:
  bin_path = fixtures_dir / f"{slug}.bin"
  metadata_path = fixtures_dir / f"{slug}.json"
  if not bin_path.is_file() or not metadata_path.is_file():
    raise FileNotFoundError(f"missing fixture outputs for {slug}")
  metadata = json.loads(metadata_path.read_text())
  expected = {
      "prompt": prompt,
      "temperature": 1.3,
      "top_k": 40,
      "cfg_musiccoca": 3.0,
      "cfg_notes": 1.0,
      "cfg_drums": 1.0,
      "drums_enabled": None,
      "intensity": 0.5,
  }
  for key, value in expected.items():
    if metadata.get(key) != value:
      raise ValueError(f"{slug} metadata {key}={metadata.get(key)!r}, expected {value!r}")
  digest = _sha256(bin_path)
  if metadata.get("sha256", {}).get("bin") != digest:
    raise ValueError(f"{slug} metadata hash does not match fixture bytes")
  if slug == "warm" and digest != EXPECTED_WARM_SHA256:
    raise ValueError(
        f"warm fixture drift: {digest}; expected {EXPECTED_WARM_SHA256}"
    )
  return {
      "id": slug,
      "prompt": prompt,
      "sourceConditioning": {
          "path": _relative(bin_path),
          "sha256": digest,
          "shape": metadata["shape"],
          "dtype": metadata["dtype"],
      },
      "compilerProvenance": {
          "sourceCommit": metadata["source_commit"],
          "exporterSha256": _sha256(EXPORTER),
          "temperature": metadata["temperature"],
          "topK": metadata["top_k"],
          "cfgMusicCoCa": metadata["cfg_musiccoca"],
          "cfgNotes": metadata["cfg_notes"],
          "cfgDrums": metadata["cfg_drums"],
          "notes": "masked",
          "drums": "masked",
          "activeMIDI": False,
          "intensity": metadata["intensity"],
      },
  }


def _protocol(
    fixtures: list[dict[str, Any]],
    *,
    private_commit: str,
    public_commit: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
  modes = (
      {"id": "refresh10", "refreshSeconds": 10.0, "refreshFrames": 250},
      {"id": "unrefreshed", "refreshSeconds": 0.0, "refreshFrames": None},
  )
  control_labels = ("refreshed", "unrefreshed", "context0", "corrupted")
  lineup_orders = {
      prompt_id: [
          random.Random(seed).sample(control_labels, k=len(control_labels))
          for seed in seeds
      ]
      for prompt_id, seeds in LINEUP_SEEDS.items()
  }
  arms = [
      {
          "armId": f"{fixture['id']}__seed-{seed}__{mode['id']}",
          "promptId": fixture["id"],
          "seed": seed,
          "mode": mode["id"],
          "refreshSeconds": mode["refreshSeconds"],
          "sourceConditioningSha256": fixture["sourceConditioning"]["sha256"],
      }
      for fixture in fixtures
      for seed in SEEDS
      for mode in modes
  ]
  return {
      "schema": "mrt2-liveness-protocol-v1",
      "status": "frozen-before-candidate-generation",
      "repositories": {
          "privateCommit": private_commit,
          "publicCommit": public_commit,
      },
      "checkpoint": {
          "path": str(checkpoint_path.resolve()),
          "sha256": _sha256(checkpoint_path),
      },
      "generation": {
          "audibleSeconds": 600,
          "tokenRateHz": 25,
          "expectedTokenFramesIncludingLookahead": 15_001,
          "temperature": 1.0,
          "topK": 40,
          "decoderContextFrames": 12,
          "continuousDecoderOverlapAdd": True,
      },
      "analysisWindowsSeconds": [
          {"id": "early", "start": 60, "end": 90},
          {"id": "middle", "start": 285, "end": 315},
          {"id": "late", "start": 570, "end": 600},
      ],
      "fixtures": fixtures,
      "seeds": list(SEEDS),
      "refreshModes": list(modes),
      "arms": arms,
      "diagnostics": {
          "token": {
              "entropyByRVQLevel": "Shannon bits over local 0..1023 codes per RVQ level and frozen window",
              "lagRecurrence": "exact whole-frame match fraction at predeclared integer lags",
              "exactCycle": "all compared token frames repeat at a tested lag",
          },
          "audio": {
              "rms": "root mean square over finite stereo samples",
              "peak": "maximum absolute finite stereo sample",
              "clipping": "fraction of samples with absolute value at least 1.0",
              "spectralFlatness": "geometric-to-arithmetic power ratio per frozen window",
              "stereoCorrelation": "Pearson correlation of left and right samples",
              "exactRepeatedWindow": "byte-identical decoded PCM across distinct 30-second windows",
          },
          "promptStratifiedOnly": {
              "envelopePulseShare4To16Hz": "band-power share of the amplitude envelope; diagnostic, never a universal gate",
              "promptAdherence": "frozen evaluator score interpreted only against all refreshed seeds of the same prompt",
          },
          "universalMusicQualityScore": None,
      },
      "catastrophicIntegrityFailures": [
          "non-finite token or PCM sample",
          "token or audio duration shorter than the frozen horizon",
          "digital clipping introduced relative to the matched refreshed arm",
          "exact repeated 30-second PCM window",
          "stereo channel collapse relative to the matched refreshed arm",
          "prompt adherence outside the refreshed range for all three seeds of a prompt",
      ],
      "audioJudge": {
          "controlLabels": list(control_labels),
          "validVoteRequires": ["refreshed passes", "corrupted fails"],
          "validVotesPerPrompt": 3,
          "agreement": "unanimous",
          "disagreementVerdict": "inconclusive",
          "neutralContextRequired": True,
          "lineupSeedsByPrompt": {key: list(value) for key, value in LINEUP_SEEDS.items()},
          "lineupOrdersByPrompt": lineup_orders,
          "interpretation": "calibrated model-based perceptual votes, not human participants",
      },
      "failureHandling": {
          "onePromptFailure": "G5 fail; do not delete or replace the prompt",
          "judgeDisagreement": "G5 inconclusive; do not average into pass",
          "thresholdChangesAfterFreeze": "invalidate affected run",
          "referenceFailure": "stop before Core ML or device work",
      },
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES)
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument(
      "--reuse-fixtures",
      action="store_true",
      help="Validate existing fixture bytes without invoking the compiler.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
  args.fixtures_dir.mkdir(parents=True, exist_ok=True)
  fixtures = []
  for slug, prompt in PROMPTS:
    if not args.reuse_fixtures:
      _compile_fixture(slug, prompt, args.fixtures_dir)
    fixtures.append(_fixture_record(slug, prompt, args.fixtures_dir))
  protocol = _protocol(
      fixtures,
      private_commit=_git_commit(REPO_ROOT),
      public_commit=_git_commit(PUBLIC_REPO),
      checkpoint_path=args.checkpoint,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
  print(args.output)


if __name__ == "__main__":
  main()
