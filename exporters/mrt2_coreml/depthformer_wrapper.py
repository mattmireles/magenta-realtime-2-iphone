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

"""PyTorch Depthformer wrapper for the MRT2 Core ML port.

Cross-file contract:

- ``magenta_rt/mlx/depthformer.py`` and ``magenta_rt/jax/depthformer.py`` are
  the behavioral references for the full streaming Depthformer.
- ``core/src/mlx_engine.cpp`` is the production runtime that drives the exported
  ``mrt2_small.mlxfn`` state list and recovers ``tokens_out`` from the inner
  ``previous_frame`` slot.
- This file creates the PyTorch/Core ML conversion surface. It starts with the
  traceable previous-frame embedding and depth-logit projection path, while
  leaving RNG, top-k/top-p, CFG control, valid-range masking, and
  ``previous_frame`` mutation outside the graph.

The class below is not yet the full temporal transformer. It is the first
Phase 1 artifact: it proves checkpoint tensor mapping, fixed tensor shapes,
semantic state names, and a pure-tensor ``forward()`` that can be traced before
the full temporal/depth blocks are filled in.
"""

from __future__ import annotations

import torch
from torch import nn

from mrt2_coreml.mrt2_weight_loader import (
    DepthformerTensorBundle,
    MRT2_TEMPORAL_CACHE_SHAPE,
    build_temporal_state_specs,
    load_depthformer_tensor_bundle,
)


MRT2_FRAME_HZ = 25
MRT2_FRAME_MS = 40
MRT2_RVQ_LEVELS = 12
MRT2_CODEBOOK_SIZE = 1_024
MRT2_RESERVED_TOKENS = 6
MRT2_VOCAB_SIZE = MRT2_CODEBOOK_SIZE
MRT2_DEPTHFORMER_LOGIT_SIZE = MRT2_RVQ_LEVELS * MRT2_CODEBOOK_SIZE + MRT2_RESERVED_TOKENS
MRT2_TEMPORAL_LAYERS = 12
MRT2_LOCAL_WINDOW_FRAMES = 41
MRT2_TEMPORAL_HEADS = 8
MRT2_HEAD_DIM = 128
MRT2_MODEL_DIM = 1_024
MRT2_DEPTH_MODEL_DIM = 768
MRT2_DECODER_EMBEDDING_SCALE = MRT2_MODEL_DIM ** 0.5


class DepthformerLogitsWrapper(nn.Module):
  """Traceable PyTorch wrapper for selected MRT2 Depthformer tensors.

  Args:
    tensor_bundle: Selected checkpoint tensors loaded from
      ``mrt2_small.safetensors``. ``None`` loads the local default checkpoint.

  Inputs:
    ``previous_frame_unique_codes`` has shape ``[1, 1, 12]`` and contains the
    Depthformer unique-code previous frame. This is the same token space held by
    the exported MLX inner ``previous_frame`` state before C++ converts it to
    raw 0-1023 RVQ codes.

  Returns:
    A pre-sampling temporal input tensor shaped ``[1, 1, 1024]``. This first
    Phase 1 version proves the previous-frame embedding contract that feeds the
    temporal body. The temporal transformer body and two-layer depth body are
    still the next implementation step before logit parity can be claimed.
  """

  def __init__(self, tensor_bundle: DepthformerTensorBundle | None = None):
    super().__init__()
    bundle = tensor_bundle or load_depthformer_tensor_bundle()

    self.decoder_embedding = nn.Embedding(
        num_embeddings=bundle.decoder_embedding.shape[0],
        embedding_dim=bundle.decoder_embedding.shape[1],
    )
    self.depth_input_adapter = nn.Linear(
        in_features=bundle.depth_input_adapter_kernel.shape[0],
        out_features=bundle.depth_input_adapter_kernel.shape[1],
        bias=False,
    )
    self.depth_final_ln = nn.LayerNorm(MRT2_DEPTH_MODEL_DIM)
    self.depth_to_logits = nn.Linear(
        in_features=bundle.depth_to_logits_kernel.shape[0],
        out_features=bundle.depth_to_logits_kernel.shape[1],
    )

    self._load_selected_weights(bundle)
    self._register_temporal_state_buffers()

  def _load_selected_weights(self, bundle: DepthformerTensorBundle) -> None:
    """Copy selected JAX-layout checkpoint tensors into PyTorch modules."""
    with torch.no_grad():
      self.decoder_embedding.weight.copy_(
          torch.from_numpy(bundle.decoder_embedding)
      )
      self.depth_input_adapter.weight.copy_(
          torch.from_numpy(bundle.depth_input_adapter_kernel.T)
      )
      self.depth_final_ln.weight.copy_(torch.from_numpy(bundle.depth_final_ln_scale))
      self.depth_final_ln.bias.copy_(torch.from_numpy(bundle.depth_final_ln_bias))
      self.depth_to_logits.weight.copy_(torch.from_numpy(bundle.depth_to_logits_kernel.T))
      self.depth_to_logits.bias.copy_(torch.from_numpy(bundle.depth_to_logits_bias))

  def _register_temporal_state_buffers(self) -> None:
    """Register semantic fixed-shape temporal cache buffers for Core ML export.

    Core ML ``ct.StateType`` names must match PyTorch ``register_buffer`` names.
    The registered buffers are zero-initialized shape placeholders for Phase 2
    export plumbing; Phase 1 tests assert that all 48 temporal K/V cache names
    exist and have the direct ``[1, 41, 8, 128]`` shape from the state inventory.
    """
    for spec in build_temporal_state_specs():
      self.register_buffer(
          spec.name,
          torch.zeros(MRT2_TEMPORAL_CACHE_SHAPE, dtype=torch.float16),
      )

  def temporal_pre_sampling(
      self, previous_frame_unique_codes: torch.Tensor
  ) -> torch.Tensor:
    """Return the temporal-body input from previous-frame unique codes.

    This is the first deterministic pre-sampling tensor in
    ``magenta_rt/mlx/depthformer.py``:

    ``decoder_embedding(previous_frame) -> scale(32) -> mean(axis=RVQ)``.

    Args:
      previous_frame_unique_codes: Integer tensor shaped ``[1, 1, 12]``.

    Returns:
      Float tensor shaped ``[1, 1, 1024]``.
    """
    embedded_frame = self.decoder_embedding(previous_frame_unique_codes.long())
    temporal_input = (embedded_frame * MRT2_DECODER_EMBEDDING_SCALE).mean(dim=-2)
    return self._touch_temporal_state_buffers(temporal_input)

  def _touch_temporal_state_buffers(self, temporal_input: torch.Tensor) -> torch.Tensor:
    """Read all temporal state buffers without changing wrapper output.

    ``coremltools`` rejects ``ct.StateType`` entries when the matching PyTorch
    buffers are registered but unused by ``forward()``. The full temporal
    transformer will read and update these buffers directly in a later phase.
    Until then, this method keeps the stateful export surface alive by reading
    one scalar from each state and multiplying it by a runtime zero derived from
    ``temporal_input``.
    """
    runtime_zero = temporal_input.sum() * 0.0
    state_zero = runtime_zero
    for spec in build_temporal_state_specs():
      state_buffer = getattr(self, spec.name)
      state_zero = state_zero + state_buffer.reshape(-1)[0].to(
          temporal_input.dtype
      ) * runtime_zero
    return temporal_input + state_zero

  def logits_smoke(self, previous_frame_unique_codes: torch.Tensor) -> torch.Tensor:
    """Return a logits-shaped smoke tensor from selected mapped weights.

    This helper proves adapter/final-projection weight loading, but it is not
    full Depthformer logit parity because the temporal transformer and depth
    transformer bodies are not implemented in PyTorch yet.
    """
    temporal_input = self.temporal_pre_sampling(previous_frame_unique_codes)
    depth_input = self.depth_input_adapter(temporal_input)
    normalized = self.depth_final_ln(depth_input)
    return self.depth_to_logits(normalized)

  def forward(self, previous_frame_unique_codes: torch.Tensor) -> torch.Tensor:
    """Return the traceable Phase 1 pre-sampling tensor."""
    return self.temporal_pre_sampling(previous_frame_unique_codes)
