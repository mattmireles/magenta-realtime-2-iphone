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

"""Export MusicCoCa-derived source conditioning for the MRT2 Core ML port.

The runtime Core ML temporal packages consume ``source_encoded`` with shape
``[1, frames, 256]``. This script keeps MusicCoCa and MRT2 conditioning-token
assembly off the iOS audio path by compiling prompt text on the Mac, running the
same MLX conditioning encoder used by MRT2, and saving little-endian float32
source frames that an iOS host app can bundle and load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from magenta_rt import MagentaRT2Mlxfn
from mrt2_coreml import paths
from magenta_rt.mlx.system import discretize_cfg


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "conditioning"
DEFAULT_MODEL_NAME = "mrt2_small"
SOURCE_DIMENSION = 256
IN_PROCESS_RESERVED_TOKEN_OFFSET = 7
MASKED_TOKEN = -1
NOTE_OFF_TOKEN = 0
NOTE_SUSTAIN_TOKEN = 1
NOTE_ONSET_TOKEN = 2
NOTE_ACTIVE_TOKEN = 3
DEFAULT_NOTE_TOKENS = (MASKED_TOKEN,) * 128
DEFAULT_DRUM_TOKENS = (MASKED_TOKEN,)
DEFAULT_INTENSITY = 0.5

# CFG-token quantization, mirroring ``magenta_rt.mlx.system.discretize_cfg``
# and the in-process ``MagentaRT2System.generate`` path: musiccoca and notes
# scales use a 0.2-per-token step over [-1.0, 7.0] (token 20 == CFG 3.0);
# drums uses a 1.0-per-token step (token 4 == CFG 3.0). The slots are NOT
# interchangeable — a previous version of this script hardcoded (4, 4, 4),
# which decodes to style CFG -0.2 (anti-guidance) — i.e. conditioning that
# actively pushes generation away from the prompt.
CFG_MUSICCOCA_STEP = 0.2
CFG_NOTES_STEP = 0.2
CFG_DRUMS_STEP = 1.0
CFG_MUSICCOCA_MAX_BIN = 40
CFG_NOTES_MAX_BIN = 40
CFG_DRUMS_MAX_BIN = 8


def _cfg_tokens(cfg_musiccoca: float, cfg_notes: float, cfg_drums: float) -> list[int]:
  """Discretize CFG scales into the three MRT2 guidance conditioning tokens."""
  return [
      discretize_cfg(cfg_musiccoca, CFG_MUSICCOCA_STEP, CFG_MUSICCOCA_MAX_BIN),
      discretize_cfg(cfg_notes, CFG_NOTES_STEP, CFG_NOTES_MAX_BIN),
      discretize_cfg(cfg_drums, CFG_DRUMS_STEP, CFG_DRUMS_MAX_BIN),
  ]


def _git_commit() -> str:
  """Return the current git commit hash or an unavailable marker."""
  try:
    return subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
  except (OSError, subprocess.CalledProcessError) as exc:
    return f"unavailable: {exc}"


def _slug(value: str) -> str:
  """Return a stable lowercase filename slug."""
  slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
  return slug[:64] or "prompt"


def _array_to_numpy(array: Any) -> np.ndarray:
  """Evaluate an MLX array and return a NumPy copy."""
  import mlx.core as mx

  mx.eval(array)
  try:
    return np.array(array)
  except RuntimeError:
    return np.array(array.astype(mx.float32))


def _sha256(path: Path) -> str:
  """Return a SHA-256 digest for a file."""
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _parse_midi_notes(value: str | None) -> list[int]:
  """Parse comma-separated MIDI notes and validate the 0...127 range."""
  if not value:
    return []
  notes: list[int] = []
  for raw_note in value.split(","):
    raw_note = raw_note.strip()
    if not raw_note:
      continue
    note = int(raw_note)
    if not 0 <= note <= 127:
      raise ValueError(f"MIDI note out of range 0...127: {note}")
    notes.append(note)
  return sorted(set(notes))


def _parse_midi_note_states(value: str | None) -> dict[int, str]:
  """Parse comma-separated ``pitch:state`` MIDI controls."""
  if not value:
    return {}
  states: dict[int, str] = {}
  for raw_item in value.split(","):
    raw_item = raw_item.strip()
    if not raw_item:
      continue
    if ":" not in raw_item:
      raise ValueError(f"Expected MIDI note state as pitch:state: {raw_item}")
    raw_pitch, state = raw_item.split(":", 1)
    pitch = int(raw_pitch)
    if not 0 <= pitch <= 127:
      raise ValueError(f"MIDI note out of range 0...127: {pitch}")
    if state not in {"off", "sustain", "onset", "auto"}:
      raise ValueError(f"Unsupported MIDI note state for {pitch}: {state}")
    states[pitch] = state
  return states


def _note_tokens(
  note_states: dict[int, str],
  pitch_mask_width: int,
  auto_strum: bool,
) -> list[int]:
  """Build 128 MRT2 pianoroll tokens from MIDI note controls."""
  active_notes = sorted(note for note, state in note_states.items() if state != "off")
  if not active_notes:
    return list(DEFAULT_NOTE_TOKENS)
  width = min(max(pitch_mask_width, 0), 127)
  tokens = [MASKED_TOKEN] * 128
  for note in active_notes:
    start = max(0, note - width)
    end = min(128, note + width + 1)
    for pitch in range(start, end):
      tokens[pitch] = NOTE_OFF_TOKEN
  for note in active_notes:
    state = note_states[note]
    if state == "auto":
      tokens[note] = NOTE_ACTIVE_TOKEN
    elif state == "onset":
      tokens[note] = NOTE_ONSET_TOKEN
    else:
      tokens[note] = NOTE_SUSTAIN_TOKEN
  return tokens


def _drum_tokens(drums_enabled: bool | None) -> list[int]:
  """Build MRT2 drum token: masked/let-model-decide, no-drum, or play-drum."""
  if drums_enabled is None:
    return list(DEFAULT_DRUM_TOKENS)
  return [1 if drums_enabled else 0]


def _parse_weighted_prompt(value: str) -> tuple[str, float]:
  """Parse ``prompt=weight`` or ``weight::prompt`` weighted prompt syntax."""
  if "::" in value:
    raw_weight, prompt = value.split("::", 1)
    return prompt, float(raw_weight)
  if "=" in value:
    prompt, raw_weight = value.rsplit("=", 1)
    return prompt, float(raw_weight)
  return value, 1.0


def _intensity_gain(intensity: float) -> float:
  """Map normalized product intensity to a default-preserving source gain."""
  bounded = min(max(float(intensity), 0.0), 1.0)
  return 0.5 + bounded


def _compile_style_tokens(
  mrt: MagentaRT2Mlxfn,
  prompt: str,
  weighted_prompts: list[str],
) -> tuple[list[int], list[dict[str, Any]]]:
  """Compile a single prompt or weighted prompt surface into MusicCoCa tokens."""
  prompt_specs = [_parse_weighted_prompt(item) for item in weighted_prompts]
  if not prompt_specs:
    prompt_specs = [(prompt, 1.0)]
  total_weight = sum(max(weight, 0.0) for _, weight in prompt_specs)
  if total_weight <= 0:
    raise ValueError("Weighted prompt weights must sum to a positive value")
  embeddings = []
  normalized_specs: list[dict[str, Any]] = []
  for text, weight in prompt_specs:
    normalized_weight = max(weight, 0.0) / total_weight
    embedding = _array_to_numpy(mrt.embed_style(text, use_mapper=True)).astype(np.float32)
    embeddings.append(embedding * normalized_weight)
    normalized_specs.append({
        "text": text,
        "weight": weight,
        "normalized_weight": normalized_weight,
    })
  blended_embedding = np.sum(np.stack(embeddings, axis=0), axis=0)
  style_tokens = mrt._style_model.tokenize(blended_embedding).tolist()
  return style_tokens, normalized_specs


def _build_mlx_sampler():
  """Build and weight-load the in-process MLX MRT2 sampler."""
  import mlx.core as mx

  from magenta_rt.mlx import load_weights as mlx_load_weights
  from magenta_rt.mlx import model
  from magenta_rt.mlx import spectrostream
  from magenta_rt.mlx import system

  exp = model.get_model_class("mrt2_small")()
  exp.compute_dtype = mx.float32
  sampler_config = system.MagentaRT2Sampler.Config(
      depthformer=exp.depthformer_config(),
      spectrostream=spectrostream.stft_spectrostream_40ms_generic_48khz_stereo_config(
          rvq_truncation_level=exp.spectrostream.rvq_truncation_level,
          use_unique_codes=False,
      ),
  )
  sampler = sampler_config.make()
  original_convert = mlx_load_weights.convert_to_bf16
  mlx_load_weights.convert_to_bf16 = lambda module: None
  try:
    mlx_load_weights.load_weights(
        sampler,
        paths.checkpoints_dir() / "mrt2_small.safetensors",
        num_input_channels=exp.input_num_channels,
    )
  finally:
    mlx_load_weights.convert_to_bf16 = original_convert
  return sampler


def export(args: argparse.Namespace) -> dict[str, Any]:
  """Compile prompt conditioning and write binary + metadata artifacts."""
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  basename = args.output_stem or f"mrt2_source_{_slug(args.prompt)}"
  bin_path = output_dir / f"{basename}.bin"
  json_path = output_dir / f"{basename}.json"

  start_load = time.perf_counter()
  mrt = MagentaRT2Mlxfn(
      size=args.model,
      temperature=args.temperature,
      top_k=args.top_k,
      cfg_musiccoca=args.cfg_musiccoca,
      cfg_notes=args.cfg_notes,
      cfg_drums=args.cfg_drums,
  )
  load_seconds = time.perf_counter() - start_load

  start_compile = time.perf_counter()
  style_tokens, prompt_surface = _compile_style_tokens(
      mrt,
      args.prompt,
      args.weighted_prompt,
  )
  midi_notes = _parse_midi_notes(args.midi_notes)
  note_states = _parse_midi_note_states(args.midi_note_states)
  default_active_state = "auto" if args.auto_strum else args.note_state
  for note in midi_notes:
    note_states.setdefault(note, default_active_state)
  note_tokens = _note_tokens(
      note_states,
      args.pitch_mask_width,
      args.auto_strum,
  )
  drums_enabled = None if args.drums_enabled == "masked" else args.drums_enabled == "true"
  drum_tokens = _drum_tokens(drums_enabled)
  sampler = _build_mlx_sampler()
  depth_sampler = sampler.layers[0]
  import mlx.core as mx
  import sequence_layers.mlx as sl

  cfg_tokens = _cfg_tokens(args.cfg_musiccoca, args.cfg_notes, args.cfg_drums)
  conditioning_tokens = np.concatenate(
      [
          np.array(style_tokens, dtype=np.int32),
          np.array(note_tokens, dtype=np.int32),
          np.array(drum_tokens, dtype=np.int32),
          np.array(cfg_tokens, dtype=np.int32),
      ],
      axis=0,
  )
  conditioning_tokens = conditioning_tokens + IN_PROCESS_RESERVED_TOKEN_OFFSET
  block = sl.Sequence(
      mx.array(conditioning_tokens.reshape(1, 1, -1), dtype=mx.int32),
      mx.ones((1, 1), dtype=mx.bool_),
  )
  constants = {
      "temperature": mx.array([args.temperature], dtype=mx.float32),
      "top_k": mx.array([args.top_k], dtype=mx.int32),
  }
  state = depth_sampler.get_initial_state(
      1,
      block.channel_spec,
      constants=constants,
      training=False,
  )
  encoder_state, _, _, _ = state
  encoded, _ = depth_sampler.encoder.body.step(
      block,
      encoder_state,
      training=False,
      constants=constants,
  )
  source = _array_to_numpy(encoded.values).astype("<f4", copy=False)
  source = source.reshape(-1, SOURCE_DIMENSION)
  intensity_gain = _intensity_gain(args.intensity)
  source = (source * intensity_gain).astype("<f4", copy=False)
  if args.frame_count > 1:
    source = np.repeat(source[:1], args.frame_count, axis=0)
  compile_seconds = time.perf_counter() - start_compile

  source.tofile(bin_path)
  metadata = {
      "schema": "mrt2-source-conditioning-v1",
      "created_by": "exporters/export_conditioning.py",
      "source_commit": _git_commit(),
      "prompt": args.prompt,
      "prompt_surface": prompt_surface,
      "model": args.model,
      "style_tokens": style_tokens,
      "midi_notes": midi_notes,
      "midi_note_states": {str(note): state for note, state in sorted(note_states.items())},
      "note_tokens": note_tokens,
      "pitch_mask_width": args.pitch_mask_width,
      "auto_strum": args.auto_strum,
      "note_state": args.note_state,
      "drums_enabled": drums_enabled,
      "drum_tokens": drum_tokens,
      "conditioning_tokens_shape": list(conditioning_tokens.reshape(1, 1, -1).shape),
      "conditioning_contract": "in-process MRT2 sampler block: 12 style + 128 notes + 1 drum + 3 cfg tokens, offset by 7",
      "shape": [int(source.shape[0]), int(source.shape[1])],
      "dtype": "float32-little-endian",
      "temperature": args.temperature,
      "top_k": args.top_k,
      "cfg_musiccoca": args.cfg_musiccoca,
      "cfg_notes": args.cfg_notes,
      "cfg_drums": args.cfg_drums,
      "cfg_tokens": cfg_tokens,
      "intensity": args.intensity,
      "intensity_gain": intensity_gain,
      "seed": args.seed,
      "load_seconds": load_seconds,
      "compile_seconds": compile_seconds,
      "artifacts": {
          "bin": str(bin_path),
          "json": str(json_path),
      },
      "sha256": {
          "bin": _sha256(bin_path),
      },
      "assets": {
          "models_dir": str(paths.models_dir()),
          "checkpoints_dir": str(paths.checkpoints_dir()),
      },
      "known_limits": [
          "This compiles a static prompt/source vector for bundle-time or controller-time use.",
          "It does not run MusicCoCa on iPhone.",
          "MIDI/drum/intensity controls are folded into the exported source vector, but not yet varied frame-by-frame.",
      ],
  }
  json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
  return metadata


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prompt", required=True)
  parser.add_argument(
      "--weighted-prompt",
      action="append",
      default=[],
      help="Prompt-surface point as 'text=weight' or 'weight::text'. Repeat to blend prompts before tokenization.",
  )
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--output-stem", default=None)
  parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
  parser.add_argument("--frame-count", type=int, default=1)
  parser.add_argument("--midi-notes", default="")
  parser.add_argument(
      "--midi-note-states",
      default="",
      help="Comma-separated per-pitch controls such as '60:auto,64:onset,67:sustain'.",
  )
  parser.add_argument("--pitch-mask-width", type=int, default=0)
  parser.add_argument("--auto-strum", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--note-state", choices=("sustain", "onset"), default="sustain")
  parser.add_argument("--drums-enabled", choices=("masked", "true", "false"), default="masked")
  parser.add_argument("--temperature", type=float, default=1.3)
  parser.add_argument("--top-k", type=int, default=40)
  parser.add_argument("--cfg-musiccoca", type=float, default=3.0)
  parser.add_argument("--cfg-notes", type=float, default=1.0)
  parser.add_argument("--cfg-drums", type=float, default=1.0)
  parser.add_argument(
      "--intensity",
      type=float,
      default=DEFAULT_INTENSITY,
      help="Normalized energy macro. 0.5 preserves current output; lower attenuates and higher boosts the source vector.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=None,
      help="Optional seed recorded as source-conditioning provenance. The current MusicCoCa export path is deterministic.",
  )
  return parser.parse_args()


def main() -> None:
  """CLI entry point."""
  metadata = export(parse_args())
  print(f"Wrote {metadata['artifacts']['bin']}")
  print(f"Wrote {metadata['artifacts']['json']}")
  print(f"shape={metadata['shape']} sha256={metadata['sha256']['bin']}")


if __name__ == "__main__":
  main()
