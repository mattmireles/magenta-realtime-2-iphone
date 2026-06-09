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

"""Export the SpectroStream RVQ codebooks needed by the iOS proof app.

The production boundary keeps RVQ detokenization on the host. The iOS probe does
not need the full MRT2 checkpoint; it only needs the first 12 SpectroStream
quantizer tables shaped ``[12, 1024, 256]`` as little-endian float32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import safetensors.flax as safetensors_flax

from mrt2_coreml import paths
from mrt2_coreml.sampling import MRT2_RVQ_LEVELS


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "models"
DEFAULT_BIN_NAME = "spectrostream_rvq_codebooks_12_f32.bin"
DEFAULT_METADATA_NAME = "spectrostream_rvq_codebooks_12_f32.json"
QUANTIZER_KEY = "params/soundstream/quantizer/embedding"
CODEBOOK_SIZE = 1_024
EMBEDDING_DIM = 256


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--checkpoint-path",
      default=str(paths.checkpoints_dir() / "mrt2_small.safetensors"),
  )
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--bin-name", default=DEFAULT_BIN_NAME)
  parser.add_argument("--metadata-name", default=DEFAULT_METADATA_NAME)
  return parser.parse_args()


def main() -> None:
  """Export codebooks and a small metadata sidecar."""
  args = parse_args()
  checkpoint_path = Path(args.checkpoint_path)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  arrays = safetensors_flax.load_file(str(checkpoint_path))
  if QUANTIZER_KEY not in arrays:
    raise KeyError(f"Missing checkpoint key: {QUANTIZER_KEY}")
  codebooks = np.asarray(arrays[QUANTIZER_KEY], dtype="<f4")
  expected_tail = (CODEBOOK_SIZE, EMBEDDING_DIM)
  if codebooks.ndim != 3 or tuple(codebooks.shape[1:]) != expected_tail:
    raise ValueError(f"Unexpected codebook shape: {codebooks.shape}")
  exported = np.ascontiguousarray(codebooks[:MRT2_RVQ_LEVELS], dtype="<f4")

  bin_path = output_dir / args.bin_name
  metadata_path = output_dir / args.metadata_name
  exported.tofile(bin_path)
  metadata = {
      "schema": "spectrostream-rvq-codebooks-ios-v1",
      "checkpoint_path": str(checkpoint_path),
      "quantizer_key": QUANTIZER_KEY,
      "dtype": "little_endian_float32",
      "shape": list(exported.shape),
      "byte_count": int(bin_path.stat().st_size),
  }
  metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
  print(f"Wrote {bin_path}")
  print(f"Wrote {metadata_path}")


if __name__ == "__main__":
  main()
