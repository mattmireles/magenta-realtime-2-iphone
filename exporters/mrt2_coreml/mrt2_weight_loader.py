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

"""Load selected MRT2 Depthformer tensors for the Core ML proof.

This module is the checkpoint-facing companion to
``magenta_rt.coreml.depthformer_wrapper``. The source checkpoint uses JAX-style
slash-delimited safetensors keys. The PyTorch wrapper uses semantic module names
and explicit state names because the later Core ML export must expose names that
an iOS profiler and a future agent can reason about.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Mapping

import numpy as np
import safetensors.numpy as safetensors_numpy

from mrt2_coreml import paths


MRT2_SMALL_CHECKPOINT = "mrt2_small.safetensors"
DEPTHFORMER_PREFIX = "params/depthformer/"
DECODER_PREFIX = f"{DEPTHFORMER_PREFIX}decoder/"
MRT2_TEMPORAL_LAYER_COUNT = 12
MRT2_TEMPORAL_WINDOWS_PER_LAYER = 2
MRT2_TEMPORAL_CACHE_TENSORS_PER_WINDOW = 2
MRT2_TEMPORAL_STATE_TENSOR_COUNT = (
    MRT2_TEMPORAL_LAYER_COUNT
    * MRT2_TEMPORAL_WINDOWS_PER_LAYER
    * MRT2_TEMPORAL_CACHE_TENSORS_PER_WINDOW
)
MRT2_TEMPORAL_CACHE_SHAPE = (1, 41, 8, 128)
MRT2_PREVIOUS_FRAME_STATE_SHAPE = (1, 1, 12)

DECODER_EMBEDDING_KEY = (
    f"{DECODER_PREFIX}decoder_embedding/embedding/embedding"
)
DEPTH_INPUT_ADAPTER_KERNEL_KEY = (
    f"{DECODER_PREFIX}depth_body/depth_input_adapter/kernel"
)
DEPTH_FINAL_LN_SCALE_KEY = f"{DECODER_PREFIX}depth_body/final_ln/scale"
DEPTH_FINAL_LN_BIAS_KEY = f"{DECODER_PREFIX}depth_body/final_ln/bias"
DEPTH_TO_LOGITS_KERNEL_KEY = f"{DECODER_PREFIX}depth_body/to_logits/kernel"
DEPTH_TO_LOGITS_BIAS_KEY = f"{DECODER_PREFIX}depth_body/to_logits/bias"


@dataclasses.dataclass(frozen=True)
class TemporalStateSpec:
  """Names one fixed-shape temporal K/V cache exposed to Core ML.

  The current exported MLX state file flattens these as opaque ``state_N`` keys.
  For the PyTorch/Core ML proof we keep the direct shape but replace positional
  names with layer/window/cache-role names. The mapping remains conservative:
  ``window`` is a stable ordinal until Phase 1 proves whether each pair is
  self-attention or cross-attention in the exact flattened state order.
  """

  name: str
  state_index: int
  shape: tuple[int, int, int, int] = MRT2_TEMPORAL_CACHE_SHAPE


@dataclasses.dataclass(frozen=True)
class DepthformerTensorBundle:
  """Selected checkpoint tensors required by the first logits wrapper.

  These tensors cover the previous-frame embedding, the depth adapter, final
  normalization, and the final logits projection. The full temporal and depth
  transformer bodies are deliberately loaded in later steps; the first wrapper
  uses this bundle to prove names, dimensions, dtypes, and traceable tensor I/O.
  """

  decoder_embedding: np.ndarray
  depth_input_adapter_kernel: np.ndarray
  depth_final_ln_scale: np.ndarray
  depth_final_ln_bias: np.ndarray
  depth_to_logits_kernel: np.ndarray
  depth_to_logits_bias: np.ndarray


def default_mrt2_small_checkpoint_path() -> Path:
  """Return the local raw ``mrt2_small`` checkpoint path."""
  return paths.checkpoints_dir() / MRT2_SMALL_CHECKPOINT


def load_checkpoint_arrays(checkpoint_path: Path | str | None = None) -> Mapping[str, np.ndarray]:
  """Load a safetensors checkpoint as NumPy arrays.

  Args:
    checkpoint_path: Optional explicit checkpoint path. ``None`` uses the local
      ``mrt2_small.safetensors`` downloaded by ``mrt checkpoints download``.

  Returns:
    Mapping from raw safetensors keys to NumPy arrays.
  """
  path = Path(checkpoint_path) if checkpoint_path is not None else default_mrt2_small_checkpoint_path()
  return safetensors_numpy.load_file(str(path))


def load_depthformer_tensor_bundle(
    checkpoint_path: Path | str | None = None,
) -> DepthformerTensorBundle:
  """Load the selected Depthformer tensors for the PyTorch wrapper."""
  arrays = load_checkpoint_arrays(checkpoint_path)
  return DepthformerTensorBundle(
      decoder_embedding=arrays[DECODER_EMBEDDING_KEY],
      depth_input_adapter_kernel=arrays[DEPTH_INPUT_ADAPTER_KERNEL_KEY],
      depth_final_ln_scale=arrays[DEPTH_FINAL_LN_SCALE_KEY],
      depth_final_ln_bias=arrays[DEPTH_FINAL_LN_BIAS_KEY],
      depth_to_logits_kernel=arrays[DEPTH_TO_LOGITS_KERNEL_KEY],
      depth_to_logits_bias=arrays[DEPTH_TO_LOGITS_BIAS_KEY],
  )


def build_temporal_state_specs() -> tuple[TemporalStateSpec, ...]:
  """Return semantic names for the 48 fixed-shape temporal cache tensors.

  In the exported state inventory, temporal cache arrays start at ``state_5``.
  Each temporal window contributes key/value cache tensors followed by mask and
  scalar step state. The Core ML proof names only the K/V tensors here because
  those are the cache buffers that should become ``ct.StateType`` candidates.
  """
  specs: list[TemporalStateSpec] = []
  for layer in range(MRT2_TEMPORAL_LAYER_COUNT):
    for window in range(MRT2_TEMPORAL_WINDOWS_PER_LAYER):
      window_ordinal = layer * MRT2_TEMPORAL_WINDOWS_PER_LAYER + window
      key_state_index = 5 + window_ordinal * 4
      value_state_index = key_state_index + 1
      specs.append(
          TemporalStateSpec(
              name=f"temporal_layer_{layer:02d}_window_{window}_key_cache",
              state_index=key_state_index,
          )
      )
      specs.append(
          TemporalStateSpec(
              name=f"temporal_layer_{layer:02d}_window_{window}_value_cache",
              state_index=value_state_index,
          )
      )
  return tuple(specs)


def selected_depthformer_key_shapes(
    checkpoint_path: Path | str | None = None,
) -> dict[str, tuple[int, ...]]:
  """Return selected Depthformer checkpoint shapes for tests and notes."""
  arrays = load_checkpoint_arrays(checkpoint_path)
  selected_keys = [
      DECODER_EMBEDDING_KEY,
      DEPTH_INPUT_ADAPTER_KERNEL_KEY,
      DEPTH_FINAL_LN_SCALE_KEY,
      DEPTH_FINAL_LN_BIAS_KEY,
      DEPTH_TO_LOGITS_KERNEL_KEY,
      DEPTH_TO_LOGITS_BIAS_KEY,
  ]
  return {key: tuple(arrays[key].shape) for key in selected_keys}
