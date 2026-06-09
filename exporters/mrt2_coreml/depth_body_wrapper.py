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

"""PyTorch depth-body logits wrapper for the MRT2 Core ML port.

This module ports the small MRT2 depth transformer body from
``magenta_rt/mlx/transformer.py`` into a traceable PyTorch module. It deliberately
does not include the 12-layer temporal transformer or temporal K/V cache. The
input is the fixed depth-input sequence that the MLX decoder builds after the
temporal step:

``[temporal_output, embedded sampled_rvq_0, ..., embedded sampled_rvq_10]``.

That isolates the next proof boundary: real logits from fixed depth inputs,
with CPU-owned sampling still outside Core ML.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from mrt2_coreml.depthformer_wrapper import (
    MRT2_DEPTHFORMER_LOGIT_SIZE,
    MRT2_DEPTH_MODEL_DIM,
    MRT2_MODEL_DIM,
    MRT2_RVQ_LEVELS,
)
from mrt2_coreml.mrt2_weight_loader import load_checkpoint_arrays


DEPTH_BODY_PREFIX = "params/depthformer/decoder/depth_body"
DEPTH_TRANSFORMER_PREFIX = f"{DEPTH_BODY_PREFIX}/transformer"
DEPTH_BODY_LAYERS = 2
DEPTH_BODY_HEADS = 6
DEPTH_BODY_HEAD_DIM = 128
DEPTH_BODY_FFN_DIM = 3_072
DEPTH_BODY_MAX_PAST_HORIZON = MRT2_RVQ_LEVELS
DEPTHFORMER_SOFT_CAP_LOGITS = 30.0
DEPTH_BODY_EPSILON = 1e-6
QUERY_SCALE_SOFTPLUS_ZERO_RECIP = 1.442695041


def _tensor(array) -> torch.Tensor:
  """Convert a NumPy checkpoint array to a float32 tensor."""
  return torch.from_numpy(array).to(dtype=torch.float32)


def _rms_norm(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  """Apply SequenceLayers RMSNormalization over the final axis."""
  values = x.to(dtype=torch.float32)
  mean_square = torch.mean(values * values, dim=-1, keepdim=True)
  return values * torch.rsqrt(mean_square + DEPTH_BODY_EPSILON) * scale


def _layer_norm(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
  """Apply SequenceLayers LayerNormalization over the final axis."""
  values = x.to(dtype=torch.float32)
  mean = torch.mean(values, dim=-1, keepdim=True)
  variance = torch.mean((values - mean) * (values - mean), dim=-1, keepdim=True)
  return (values - mean) * torch.rsqrt(variance + DEPTH_BODY_EPSILON) * scale + bias


def _causal_depth_mask(time: int, device: torch.device) -> torch.Tensor:
  """Return the local self-attention mask used by the depth body in layer mode."""
  row = torch.arange(time, device=device)[:, None]
  col = torch.arange(time, device=device)[None, :]
  return (
      (col >= row - DEPTH_BODY_MAX_PAST_HORIZON)
      & (col <= row)
  ).reshape(1, 1, time, time)


class DepthBodyTransformerLayer(nn.Module):
  """One MRT2 depth transformer layer ported from SequenceLayers MLX."""

  def __init__(self, arrays, layer_index: int):
    super().__init__()
    prefix = f"{DEPTH_TRANSFORMER_PREFIX}/x_layers_{layer_index}"
    self.register_buffer(
        "attn_pre_norm_scale",
        _tensor(arrays[f"{prefix}/self_attention/pre_norm/scale"]),
    )
    self.register_buffer(
        "attn_post_norm_scale",
        _tensor(arrays[f"{prefix}/self_attention/post_norm/scale"]),
    )
    self.register_buffer(
        "query_kernel",
        _tensor(arrays[f"{prefix}/self_attention/attention/query_projection/kernel"]),
    )
    self.register_buffer(
        "key_kernel",
        _tensor(arrays[f"{prefix}/self_attention/attention/key_projection/kernel"]),
    )
    self.register_buffer(
        "value_kernel",
        _tensor(arrays[f"{prefix}/self_attention/attention/value_projection/kernel"]),
    )
    self.register_buffer(
        "per_dim_scale",
        _tensor(arrays[f"{prefix}/self_attention/attention/per_dim_scale"]),
    )
    self.register_buffer(
        "output_projection_kernel",
        _tensor(arrays[f"{prefix}/self_attention/output_projection/kernel"]),
    )
    self.register_buffer(
        "ffn_pre_norm_scale",
        _tensor(arrays[f"{prefix}/ffn/pre_norm/scale"]),
    )
    self.register_buffer(
        "ffn_post_norm_scale",
        _tensor(arrays[f"{prefix}/ffn/post_norm/scale"]),
    )
    self.register_buffer(
        "ffn_layer1_kernel",
        _tensor(arrays[f"{prefix}/ffn/ffn_layer1/kernel"]),
    )
    self.register_buffer(
        "ffn_layer1_bias",
        _tensor(arrays[f"{prefix}/ffn/ffn_layer1/bias"]),
    )
    self.register_buffer(
        "ffn_layer2_kernel",
        _tensor(arrays[f"{prefix}/ffn/ffn_layer2/kernel"]),
    )
    self.register_buffer(
        "ffn_layer2_bias",
        _tensor(arrays[f"{prefix}/ffn/ffn_layer2/bias"]),
    )

  def _attention(self, x: torch.Tensor) -> torch.Tensor:
    """Run local self-attention and output projection."""
    normed = _rms_norm(x, self.attn_pre_norm_scale)
    query = torch.einsum("btd,dnh->btnh", normed, self.query_kernel)
    key = torch.einsum("btd,dnh->btnh", normed, self.key_kernel)
    value = torch.einsum("btd,dnh->btnh", normed, self.value_kernel)

    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    query_scale = 1.0 / math.sqrt(DEPTH_BODY_HEAD_DIM)
    per_dim = (
        QUERY_SCALE_SOFTPLUS_ZERO_RECIP
        * query_scale
        * F.softplus(self.per_dim_scale)
    ).reshape(1, 1, 1, DEPTH_BODY_HEAD_DIM)
    scores = torch.matmul(query * per_dim, key.transpose(-1, -2))
    mask = _causal_depth_mask(x.shape[1], x.device)
    scores = torch.where(mask, scores, torch.full_like(scores, -1e9))
    weights = torch.softmax(scores.to(dtype=torch.float32), dim=-1).to(value.dtype)
    context = torch.matmul(weights, value).permute(0, 2, 1, 3)
    projected = torch.einsum(
        "btnh,dnh->btd", context, self.output_projection_kernel
    )
    return _rms_norm(projected, self.attn_post_norm_scale)

  def _ffn(self, x: torch.Tensor) -> torch.Tensor:
    """Run the non-gated MRT2 depth FFN."""
    normed = _rms_norm(x, self.ffn_pre_norm_scale)
    hidden = torch.matmul(normed, self.ffn_layer1_kernel) + self.ffn_layer1_bias
    hidden = F.gelu(hidden, approximate="tanh")
    output = torch.matmul(hidden, self.ffn_layer2_kernel) + self.ffn_layer2_bias
    return _rms_norm(output, self.ffn_post_norm_scale)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply self-attention residual followed by FFN residual."""
    x = x + self._attention(x)
    return x + self._ffn(x)


class DepthBodyLogitsWrapper(nn.Module):
  """Traceable PyTorch wrapper for MRT2 depth-body logits.

  Inputs:
    ``depth_inputs`` has shape ``[1, 12, 1024]``. Position 0 is the temporal
    transformer output for the frame. Positions 1-11 are decoder embeddings for
    already selected RVQ levels when running autoregressively.

  Returns:
    Soft-capped logits shaped ``[1, 12, 12294]``.
  """

  def __init__(self):
    super().__init__()
    arrays = load_checkpoint_arrays()
    self.register_buffer(
        "depth_input_adapter_kernel",
        _tensor(arrays[f"{DEPTH_BODY_PREFIX}/depth_input_adapter/kernel"]),
    )
    self.layers = nn.ModuleList(
        [DepthBodyTransformerLayer(arrays, layer) for layer in range(DEPTH_BODY_LAYERS)]
    )
    self.register_buffer(
        "final_ln_scale",
        _tensor(arrays[f"{DEPTH_BODY_PREFIX}/final_ln/scale"]),
    )
    self.register_buffer(
        "final_ln_bias",
        _tensor(arrays[f"{DEPTH_BODY_PREFIX}/final_ln/bias"]),
    )
    self.register_buffer(
        "to_logits_kernel",
        _tensor(arrays[f"{DEPTH_BODY_PREFIX}/to_logits/kernel"]),
    )
    self.register_buffer(
        "to_logits_bias",
        _tensor(arrays[f"{DEPTH_BODY_PREFIX}/to_logits/bias"]),
    )

  def forward(self, depth_inputs: torch.Tensor) -> torch.Tensor:
    """Return soft-capped full-vocabulary logits for fixed depth inputs."""
    x = torch.matmul(depth_inputs.to(dtype=torch.float32), self.depth_input_adapter_kernel)
    for layer in self.layers:
      x = layer(x)
    x = _layer_norm(x, self.final_ln_scale, self.final_ln_bias)
    logits = torch.matmul(x, self.to_logits_kernel) + self.to_logits_bias
    return torch.tanh(logits / DEPTHFORMER_SOFT_CAP_LOGITS) * DEPTHFORMER_SOFT_CAP_LOGITS


def deterministic_depth_body_input(seed: int = 1234) -> torch.Tensor:
  """Return a stable non-random-looking depth input for parity tests."""
  generator = torch.Generator().manual_seed(seed)
  return torch.randn(
      (1, MRT2_RVQ_LEVELS, MRT2_MODEL_DIM),
      generator=generator,
      dtype=torch.float32,
  )
