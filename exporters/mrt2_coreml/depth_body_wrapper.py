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
    MRT2_CODEBOOK_SIZE,
    MRT2_DEPTHFORMER_LOGIT_SIZE,
    MRT2_DEPTH_MODEL_DIM,
    MRT2_MODEL_DIM,
    MRT2_RESERVED_TOKENS,
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


DEPTH_ROLLOUT_TOP_K = 40
# Additive sentinel for tokens outside the top-k set. Must stay finite and
# fp16-representable (|x| < 65504) while sitting far below any reachable
# perturbed logit: |logit/T| <= 30 / 0.05 = 600 and Gumbel(0,1) noise from a
# clamped uniform stays within ~(-3, 17), so -1e4 can never win an argmax.
DEPTH_ROLLOUT_MASK_VALUE = -1e4
DECODER_EMBEDDING_KEY = (
    "params/depthformer/decoder/decoder_embedding/embedding/embedding"
)
# Fixed sl.Scale layer applied after the embedding lookup (sqrt(model dim));
# baked into the in-graph table exactly like
# scripts/export_mrt2_depth_embedder_for_ios.py bakes it into the Swift table.
DECODER_EMBEDDER_SCALE = 32.0


class DepthBodyRolloutWrapper(nn.Module):
  """Whole-frame in-graph 12-level depth rollout with host-noise sampling.

  Why this exists: every out-of-graph rollout makes 12 depth predictions per
  frame, and on phones the depth path is weight-BANDWIDTH-bound — each
  prediction streams the full depth weight set from DRAM, so 12 predictions
  cost ~12 weight streams (~40 ms/frame on iPhone 12 Pro) no matter how few
  positions they compute. Both shipped designs paid that price:
  ``DepthBodyLogitsWrapper`` (12 full passes) and
  ``DepthBodyStepLogitsWrapper`` (12 stateful single-position calls measured
  at the SAME per-call cost, falsifying the "one pass + overhead" model). See
  SEQUEL 2 in
  README/Notes/aperture-device-quality-root-cause-feedback-and-strides.md.

  This wrapper moves the entire autoregressive rollout INSIDE one prediction:
  sample level k, embed the sampled token, feed it to level k+1, twelve times,
  in one graph execution. The host makes one Core ML call instead of 12 (also
  deleting the per-level CPU sampling round-trips and the ``MLState`` K/V
  caches — depth state is purely intra-frame, so a stateless graph is correct
  by construction). Honest bandwidth accounting (device-measured 2026-06-10):
  the ``to_logits`` slices stream once per frame, but the transformer LAYER
  weights still stream once per LEVEL (sequential dependence; they do not fit
  in SLC), so FLOAT32 is ~750 MB/frame ≈ 37 ms on iPhone 12 Pro — better
  than the 12-call ~1.2 GB but still over budget. FLOAT16 weights halve it:
  12.7 ms/frame (iPhone 12 Pro), 8.4 ms (iPhone 15 Pro Max, zero underruns
  composed). FLOAT16 is the ship precision; FLOAT32 is the exactness
  reference.

  Determinism stays host-owned via the Gumbel-max trick: sampling a token
  from ``softmax(logits/T)`` restricted to the top-k set is EXACTLY
  ``argmax(logits/T + g)`` over that set when ``g`` is iid Gumbel(0,1) noise.
  The host supplies the noise (from its seeded RNG) and the inverse
  temperature as inputs; the graph contributes only deterministic math, so
  the same inputs always produce the same tokens. Top-k is the static
  ``DEPTH_ROLLOUT_TOP_K`` = 40 (the MRT2 MLX default) selected on RAW
  soft-capped logits before temperature, matching the Swift host sampler.

  Inputs:
    ``temporal_frame`` ``[1, 1, 1024]``: the temporal transformer output for
    the frame (the old ``depth_inputs`` position 0).
    ``gumbel_noise`` ``[12, 1024]``: per-level, per-code Gumbel(0,1) noise,
    ``-log(-log(u))`` with ``u`` drawn from the host RNG and clamped away
    from 0.
    ``inverse_temperature`` ``[1]``: ``1 / max(0.05, temperature)``. Hosts
    pass the reciprocal so the graph multiplies instead of divides.

  Outputs:
    ``sampled_codes`` ``[12]`` int32: codebook-LOCAL codes (0..1023) per RVQ
    level. The unique id is ``6 + level*1024 + code``.
    ``temporal_feedback`` ``[1, 1024]`` float32: mean of the 12 sampled token
    embeddings (decoder embedder rows, x32 scale baked in) — the next frame's
    ``temporal_inputs`` row, so the host no longer reads its embedder table
    on the hot path.

  Weight-bandwidth notes (the whole point of this graph):
    * ``to_logits`` is pre-sliced per level to the level's valid 1024-column
      range, so the big ``[768, 12294]`` projection is touched exactly once
      per frame instead of 12 times (the reserved columns 0..5 are never
      computed; the old graphs computed all 12294 columns per level).
    * Token-embedding feedback is an in-graph ``index_select`` on per-level
      ``[1024, 1024]`` table slices — one row read per level, negligible
      bandwidth.
    * Attention K/V are plain Python lists across the unrolled levels; the
      trace turns them into intermediate tensors. No state I/O at all.
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
    to_logits_kernel = _tensor(arrays[f"{DEPTH_BODY_PREFIX}/to_logits/kernel"])
    to_logits_bias = _tensor(arrays[f"{DEPTH_BODY_PREFIX}/to_logits/bias"])
    embedding_table = (
        _tensor(arrays[DECODER_EMBEDDING_KEY]) * DECODER_EMBEDDER_SCALE
    )
    for level in range(MRT2_RVQ_LEVELS):
      start = MRT2_RESERVED_TOKENS + level * MRT2_CODEBOOK_SIZE
      end = start + MRT2_CODEBOOK_SIZE
      self.register_buffer(
          self._logits_kernel_name(level),
          to_logits_kernel[:, start:end].contiguous(),
      )
      self.register_buffer(
          self._logits_bias_name(level),
          to_logits_bias[start:end].contiguous(),
      )
      self.register_buffer(
          self._embed_table_name(level),
          embedding_table[start:end].contiguous(),
      )

  @staticmethod
  def _logits_kernel_name(level: int) -> str:
    return f"to_logits_kernel_{level:02d}"

  @staticmethod
  def _logits_bias_name(level: int) -> str:
    return f"to_logits_bias_{level:02d}"

  @staticmethod
  def _embed_table_name(level: int) -> str:
    return f"embed_table_{level:02d}"

  def _rollout_attention(
      self,
      layer: DepthBodyTransformerLayer,
      x: torch.Tensor,
      key_list: list[torch.Tensor],
      value_list: list[torch.Tensor],
  ) -> torch.Tensor:
    """One-position attention over the in-graph K/V grown so far.

    No causal mask is needed: position k's K/V lists hold exactly positions
    0..k (the same visible set the full-pass mask allows — the depth horizon
    of 12 covers all intra-frame positions).
    """
    normed = _rms_norm(x, layer.attn_pre_norm_scale)
    query = torch.einsum("btd,dnh->btnh", normed, layer.query_kernel)
    key = torch.einsum("btd,dnh->btnh", normed, layer.key_kernel)
    value = torch.einsum("btd,dnh->btnh", normed, layer.value_kernel)
    key_list.append(key)
    value_list.append(value)
    keys = torch.cat(key_list, dim=1) if len(key_list) > 1 else key
    values = torch.cat(value_list, dim=1) if len(value_list) > 1 else value

    query_scale = 1.0 / math.sqrt(DEPTH_BODY_HEAD_DIM)
    per_dim = (
        QUERY_SCALE_SOFTPLUS_ZERO_RECIP
        * query_scale
        * F.softplus(layer.per_dim_scale)
    ).reshape(1, 1, 1, DEPTH_BODY_HEAD_DIM)
    query_heads = (query * per_dim).permute(0, 2, 1, 3)
    key_heads = keys.permute(0, 2, 1, 3)
    value_heads = values.permute(0, 2, 1, 3)
    scores = torch.matmul(query_heads, key_heads.transpose(-1, -2))
    weights = torch.softmax(scores.to(dtype=torch.float32), dim=-1)
    context = torch.matmul(weights, value_heads).permute(0, 2, 1, 3)
    projected = torch.einsum(
        "btnh,dnh->btd", context, layer.output_projection_kernel
    )
    return _rms_norm(projected, layer.attn_post_norm_scale)

  def forward(
      self,
      temporal_frame: torch.Tensor,
      gumbel_noise: torch.Tensor,
      inverse_temperature: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample all 12 RVQ levels of one frame in a single graph execution."""
    inv_temp = inverse_temperature.reshape(1, 1, 1).to(dtype=torch.float32)
    key_lists: list[list[torch.Tensor]] = [[] for _ in self.layers]
    value_lists: list[list[torch.Tensor]] = [[] for _ in self.layers]
    current = temporal_frame
    codes: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    for level in range(MRT2_RVQ_LEVELS):
      x = torch.matmul(
          current.to(dtype=torch.float32), self.depth_input_adapter_kernel
      )
      for layer_index, layer in enumerate(self.layers):
        x = x + self._rollout_attention(
            layer, x, key_lists[layer_index], value_lists[layer_index]
        )
        x = x + layer._ffn(x)
      x = _layer_norm(x, self.final_ln_scale, self.final_ln_bias)
      logits = (
          torch.matmul(x, getattr(self, self._logits_kernel_name(level)))
          + getattr(self, self._logits_bias_name(level))
      )
      logits = (
          torch.tanh(logits / DEPTHFORMER_SOFT_CAP_LOGITS)
          * DEPTHFORMER_SOFT_CAP_LOGITS
      )
      # Top-k on RAW soft-capped logits (before temperature), then Gumbel-max
      # over the surviving set — the host sampler's semantics exactly.
      threshold = torch.topk(logits, DEPTH_ROLLOUT_TOP_K, dim=-1).values[..., -1:]
      perturbed = logits * inv_temp + gumbel_noise[level].reshape(
          1, 1, MRT2_CODEBOOK_SIZE
      )
      masked = torch.where(
          logits >= threshold,
          perturbed,
          torch.full_like(perturbed, DEPTH_ROLLOUT_MASK_VALUE),
      )
      code = torch.argmax(masked, dim=-1).reshape(1)
      codes.append(code)
      embedding = torch.index_select(
          getattr(self, self._embed_table_name(level)), 0, code
      )
      embeddings.append(embedding)
      current = embedding.reshape(1, 1, MRT2_MODEL_DIM)
    sampled_codes = torch.cat(codes, dim=0).to(dtype=torch.int32)
    temporal_feedback = torch.mean(
        torch.cat(embeddings, dim=0), dim=0, keepdim=True
    )
    return sampled_codes, temporal_feedback


def gumbel_topk_sample_reference(
    logits: torch.Tensor,
    gumbel_row: torch.Tensor,
    inverse_temperature: float,
) -> int:
  """Reference (non-traced) Gumbel-max top-k sample over one level's logits.

  Mirrors the in-graph math bit for bit so validators and tests can predict
  ``DepthBodyRolloutWrapper`` tokens from full-pass reference logits.
  ``logits`` and ``gumbel_row`` are ``[1024]`` tensors for ONE level's valid
  code range; returns the codebook-local code.
  """
  threshold = torch.topk(logits, DEPTH_ROLLOUT_TOP_K).values[-1]
  perturbed = logits * inverse_temperature + gumbel_row
  masked = torch.where(
      logits >= threshold,
      perturbed,
      torch.full_like(perturbed, DEPTH_ROLLOUT_MASK_VALUE),
  )
  return int(torch.argmax(masked).item())


def level_onehot(level: int) -> torch.Tensor:
  """Return the ``[1, 12]`` float one-hot selector for one intra-frame level."""
  if not 0 <= level < MRT2_RVQ_LEVELS:
    raise ValueError(f"level must be in [0, {MRT2_RVQ_LEVELS}), got {level}")
  onehot = torch.zeros((1, MRT2_RVQ_LEVELS), dtype=torch.float32)
  onehot[0, level] = 1.0
  return onehot
