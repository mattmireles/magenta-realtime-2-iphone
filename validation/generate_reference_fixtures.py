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

"""Generate MRT2 fixtures for the MRT2 Core ML port.

This script captures the reproducible MLX reference inputs that later Core ML
Depthformer and SpectroStream proofs must match. It deliberately uses the
exported ``mrt2_small.mlxfn`` path because that is the runtime contract driven by
``core/src/mlx_engine.cpp`` today.

The fixture output lives under ``fixtures/`` and is
gitignored. Tracked source only records this deterministic generator and the
proof note that names how to regenerate its outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from magenta_rt import MagentaRT2Mlxfn
from mrt2_coreml import paths


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "fixtures"
DEFAULT_MODEL_NAME = "mrt2_small"
DEFAULT_CHECKPOINT = "mrt2_small.safetensors"
DEFAULT_PROMPT = "lofi house groove"
DEFAULT_DURATION_SECONDS = 1.0
MRT2_FRAME_HZ = 25
MRT2_AUDIO_SAMPLE_RATE = 48_000
MRT2_FRAME_SAMPLES = 1_920
MRT2_RVQ_LEVELS = 12
MRT2_CODEBOOK_SIZE = 1_024
MRT2_EXPORTED_TOKEN_OFFSET = 5
MLXFN_ARGUMENT_NAMES = [
    "conditioning",
    "temperature",
    "top_k",
    "cfg_musiccoca",
    "cfg_notes",
    "cfg_drums",
    "negative_musiccoca_conditioning",
    "negative_notes_conditioning",
    "forced_tokens",
]
PREVIOUS_FRAME_STATE_INDEX = 3


def _run_capture(command: list[str]) -> str:
  """Return stripped stdout for a local tool command, or an error marker."""
  try:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
  except (OSError, subprocess.CalledProcessError) as exc:
    return f"unavailable: {exc}"


def _package_version(name: str) -> str:
  """Return the installed Python package version or ``not-installed``."""
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return "not-installed"


def _sha256(path: Path, *, enabled: bool) -> str:
  """Return a SHA-256 digest for ``path`` when hashing is enabled."""
  if not enabled:
    return "skipped"
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _array_to_numpy(array: Any) -> np.ndarray:
  """Evaluate an MLX array and return a NumPy copy."""
  import mlx.core as mx

  mx.eval(array)
  try:
    return np.array(array)
  except RuntimeError:
    return np.array(array.astype(mx.float32))


def _jsonable_scalar(value: Any) -> Any:
  """Convert NumPy scalar values into JSON-safe Python scalars."""
  if isinstance(value, np.generic):
    return value.item()
  return value


def _state_slice_summary(index: int, array: Any, max_values: int) -> dict[str, Any]:
  """Return a bounded JSON summary for one exported state array."""
  import mlx.core as mx

  mx.eval(array)
  try:
    np_array = np.array(array)
    source_dtype = str(np_array.dtype)
  except RuntimeError:
    np_array = np.array(array.astype(mx.float32))
    source_dtype = f"{array.dtype} cast-to-float32-for-json"
  flat = np_array.reshape(-1)
  values = [_jsonable_scalar(v) for v in flat[:max_values]]
  return {
      "index": index,
      "name": f"state_{index}",
      "shape": list(np_array.shape),
      "dtype": source_dtype,
      "numel": int(np_array.size),
      "first_values": values,
  }


def _raw_tokens_from_previous_frame(state: list[Any]) -> tuple[np.ndarray, np.ndarray]:
  """Extract unique and raw RVQ tokens from the exported previous-frame state.

  ``core/src/mlx_engine.cpp`` identifies the inner decoder ``previous_frame`` as
  state slot 3 for the current ``mrt2_small`` assets. That slot stores unique
  code values. The raw per-codebook tokens are:

  ``unique[k] - k * 1024 - 5``
  """
  import mlx.core as mx

  slot = state[PREVIOUS_FRAME_STATE_INDEX].astype(mx.int32)
  unique = _array_to_numpy(slot).reshape(-1)[:MRT2_RVQ_LEVELS].astype(np.int32)
  offsets = (
      np.arange(MRT2_RVQ_LEVELS, dtype=np.int32) * MRT2_CODEBOOK_SIZE
      + MRT2_EXPORTED_TOKEN_OFFSET
  )
  raw = unique - offsets
  return unique, raw


def _build_metadata(args: argparse.Namespace, asset_paths: dict[str, Path]) -> dict[str, Any]:
  """Build deterministic fixture metadata before running generation."""
  frame_count = int(args.duration_seconds * MRT2_FRAME_HZ)
  if frame_count <= 0:
    raise ValueError(
        "--duration-seconds must produce at least one 25 Hz MRT2 frame."
    )
  return {
      "fixture_schema": "mrt2-coreml-proof-fixtures-v1",
      "created_by": "scripts/generate_mrt2_coreml_reference_fixtures.py",
      "repo_root": str(REPO_ROOT),
      "model_name": args.model,
      "checkpoint": args.checkpoint,
      "prompt": args.prompt,
      "duration_seconds": args.duration_seconds,
      "frame_count": frame_count,
      "temperature": args.temperature,
      "top_k": args.top_k,
      "cfg_musiccoca": args.cfg_musiccoca,
      "cfg_notes": args.cfg_notes,
      "cfg_drums": args.cfg_drums,
      "target_device": {
          "name": args.target_device_name,
          "model": args.target_device_model,
          "udid": args.target_device_udid,
          "os_version": args.target_device_os,
      },
      "toolchain": {
          "python": platform.python_version(),
          "platform": platform.platform(),
          "macos": _run_capture(["/usr/bin/sw_vers"]),
          "xcode": _run_capture(["/usr/bin/xcodebuild", "-version"]),
          "packages": {
              "magenta-rt": _package_version("magenta-rt"),
              "mlx": _package_version("mlx"),
              "jax": _package_version("jax"),
              "flax": _package_version("flax"),
              "safetensors": _package_version("safetensors"),
              "numpy": _package_version("numpy"),
              "scipy": _package_version("scipy"),
              "soundfile": _package_version("soundfile"),
              "torch": _package_version("torch"),
              "coremltools": _package_version("coremltools"),
          },
      },
      "assets": {
          name: {
              "path": str(path),
              "exists": path.exists(),
              "size_bytes": path.stat().st_size if path.exists() else None,
              "sha256": _sha256(path, enabled=not args.skip_asset_hashes)
              if path.exists()
              else None,
          }
          for name, path in asset_paths.items()
      },
      "known_limits": [
          "Logits and pre-sampling tensors are not exposed by the exported MLX function.",
          "Generated tokens are recovered from exported state_3 per the C++ tokens_out contract.",
      ],
  }


def generate_fixtures(args: argparse.Namespace) -> dict[str, Any]:
  """Generate fixture files and return the final metadata dictionary."""
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  model_dir = paths.models_dir() / args.model
  asset_paths = {
      "mlxfn": model_dir / f"{args.model}.mlxfn",
      "mlx_state": model_dir / f"{args.model}_state.safetensors",
      "raw_checkpoint": paths.checkpoints_dir() / args.checkpoint,
  }
  metadata = _build_metadata(args, asset_paths)

  start_load = time.perf_counter()
  mrt = MagentaRT2Mlxfn(
      size=args.model,
      temperature=args.temperature,
      top_k=args.top_k,
      cfg_musiccoca=args.cfg_musiccoca,
      cfg_notes=args.cfg_notes,
      cfg_drums=args.cfg_drums,
  )
  metadata["load_seconds"] = time.perf_counter() - start_load

  style_embedding = mrt.embed_style(args.prompt, use_mapper=True)
  style_tokens = mrt._style_model.tokenize(style_embedding).tolist()
  mlxfn_args = mrt._build_mlxfn_args(
      style_tokens,
      notes=None,
      drums=None,
      cfg_musiccoca=args.cfg_musiccoca,
      cfg_notes=args.cfg_notes,
      cfg_drums=args.cfg_drums,
      temperature=args.temperature,
      top_k=args.top_k,
  )
  conditioning = {
      name: _array_to_numpy(value)
      for name, value in zip(MLXFN_ARGUMENT_NAMES, mlxfn_args, strict=True)
  }
  np.savez(output_dir / "conditioning_inputs.npz", **conditioning)

  initial_state = list(mrt._initial_state)
  with (output_dir / "initial_state_slices.json").open("w") as f:
    json.dump(
        [_state_slice_summary(i, state, args.max_state_values) for i, state in enumerate(initial_state)],
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

  state = list(initial_state)
  audio_frames = []
  unique_tokens = []
  raw_tokens = []
  frame_count = int(args.duration_seconds * MRT2_FRAME_HZ)
  if frame_count <= 0:
    raise ValueError(
        "--duration-seconds must produce at least one 25 Hz MRT2 frame."
    )
  start_generate = time.perf_counter()
  for _ in range(frame_count):
    outputs = mrt._fn(mlxfn_args + state)
    audio_frames.append(_array_to_numpy(outputs[0]))
    state = list(outputs[1:])
    unique, raw = _raw_tokens_from_previous_frame(state)
    unique_tokens.append(unique)
    raw_tokens.append(raw)
  elapsed = time.perf_counter() - start_generate

  generated_unique_tokens = np.stack(unique_tokens, axis=0)
  generated_raw_tokens = np.stack(raw_tokens, axis=0)
  np.save(output_dir / "generated_tokens_unique.npy", generated_unique_tokens)
  np.save(output_dir / "generated_tokens_raw.npy", generated_raw_tokens)

  all_audio = np.concatenate(audio_frames, axis=-1)
  samples = all_audio[0].T.astype(np.float32)
  if np.issubdtype(all_audio.dtype, np.integer):
    samples = samples / 32768.0
  sf.write(output_dir / "reference_pcm.wav", samples, MRT2_AUDIO_SAMPLE_RATE)

  metadata["outputs"] = {
      "conditioning_inputs": "conditioning_inputs.npz",
      "initial_state_slices": "initial_state_slices.json",
      "generated_tokens_unique": "generated_tokens_unique.npy",
      "generated_tokens_raw": "generated_tokens_raw.npy",
      "reference_pcm": "reference_pcm.wav",
  }
  metadata["generation"] = {
      "elapsed_seconds": elapsed,
      "frames": frame_count,
      "steps_per_second": frame_count / elapsed,
      "ms_per_step": elapsed / frame_count * 1000,
      "audio_shape": list(all_audio.shape),
      "audio_dtype": str(all_audio.dtype),
      "pcm_sample_rate": MRT2_AUDIO_SAMPLE_RATE,
      "pcm_frame_samples": MRT2_FRAME_SAMPLES,
  }
  metadata_path = output_dir / "fixture_metadata.json"
  with metadata_path.open("w") as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
    f.write("\n")
  return metadata


def parse_args() -> argparse.Namespace:
  """Parse command-line flags for deterministic fixture generation."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
  parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
  parser.add_argument("--prompt", default=DEFAULT_PROMPT)
  parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
  parser.add_argument("--temperature", type=float, default=1.3)
  parser.add_argument("--top-k", type=int, default=40)
  parser.add_argument("--cfg-musiccoca", type=float, default=3.0)
  parser.add_argument("--cfg-notes", type=float, default=1.0)
  parser.add_argument("--cfg-drums", type=float, default=1.0)
  parser.add_argument("--max-state-values", type=int, default=16)
  parser.add_argument("--skip-asset-hashes", action="store_true")
  parser.add_argument("--target-device-name", default="Webcam")
  parser.add_argument("--target-device-model", default="iPhone 12 Pro (iPhone13,3)")
  parser.add_argument("--target-device-udid", default="00008101-001134561A0A001E")
  parser.add_argument("--target-device-os", default="iOS 26.5 build 23F77")
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  args = parse_args()
  metadata = generate_fixtures(args)
  generation = metadata["generation"]
  print(
      "Generated "
      f"{generation['frames']} frames in {generation['elapsed_seconds']:.2f}s "
      f"({generation['ms_per_step']:.1f} ms/step)."
  )
  print(f"Wrote fixtures to {Path(args.output_dir)}")


if __name__ == "__main__":
  main()
