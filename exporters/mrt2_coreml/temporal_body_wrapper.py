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

"""PyTorch temporal-body step wrapper for the MRT2 Core ML proof.

This module ports the streaming ``decoder.temporal_body`` step from
``magenta_rt/mlx/depthformer.py`` and ``magenta_rt/mlx/transformer.py``. It is
the Phase 3B bridge between the already-exported previous-frame embedding proof
and the depth-body logits proof:

``previous_frame -> temporal_input -> temporal_body_step -> depth_body_logits``.

The wrapper deliberately takes the encoder output as ``source_encoded`` instead
of converting the conditioning encoder. This port keeps conditioning and
MusicCoCa host-owned; the Core ML risk under test here is the temporal
transformer K/V update and math, not prompt-token assembly.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
from torch.nn import functional as F

from mrt2_coreml.depth_body_wrapper import DEPTH_BODY_EPSILON
from mrt2_coreml.depthformer_wrapper import (
    MRT2_HEAD_DIM,
    MRT2_LOCAL_WINDOW_FRAMES,
    MRT2_MODEL_DIM,
    MRT2_TEMPORAL_HEADS,
    MRT2_TEMPORAL_LAYERS,
)
from mrt2_coreml.mrt2_weight_loader import load_checkpoint_arrays


TEMPORAL_BODY_PREFIX = "params/depthformer/decoder/temporal_body/transformer"
TEMPORAL_FFN_DIM = 4_096
TEMPORAL_SOURCE_DIM = 256
TEMPORAL_SINKS = 1
QUERY_SCALE_SOFTPLUS_ZERO_RECIP = 1.442695041
TEMPORAL_COREML_SLOT_COUNT = MRT2_LOCAL_WINDOW_FRAMES


@dataclasses.dataclass(frozen=True)
class TemporalAttentionState:
  """State for one streaming temporal attention block."""

  key_cache: torch.Tensor
  value_cache: torch.Tensor
  mask: torch.Tensor
  step: torch.Tensor


@dataclasses.dataclass(frozen=True)
class TemporalLayerState:
  """Self-attention and streaming cross-attention state for one layer."""

  self_attention: TemporalAttentionState
  cross_attention: TemporalAttentionState


def _tensor(array) -> torch.Tensor:
  """Convert a checkpoint NumPy array to float32 torch tensor."""
  return torch.from_numpy(array).to(dtype=torch.float32)


def _rms_norm(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  """Apply SequenceLayers RMSNormalization over the final axis."""
  values = x.to(dtype=torch.float32)
  mean_square = torch.mean(values * values, dim=-1, keepdim=True)
  return values * torch.rsqrt(mean_square + DEPTH_BODY_EPSILON) * scale


def _attention_query_scale(per_dim_scale: torch.Tensor) -> torch.Tensor:
  """Return the learned SequenceLayers per-dimension query scale."""
  query_scale = 1.0 / math.sqrt(MRT2_HEAD_DIM)
  return (
      QUERY_SCALE_SOFTPLUS_ZERO_RECIP * query_scale * F.softplus(per_dim_scale)
  ).reshape(1, 1, 1, MRT2_HEAD_DIM)


def _project_heads(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
  """Project ``[B, T, D]`` values with a ``[D, N, H]`` checkpoint kernel."""
  return torch.einsum("btd,dnh->btnh", x.to(dtype=torch.float32), kernel)


def _output_projection(
    context: torch.Tensor, kernel: torch.Tensor
) -> torch.Tensor:
  """Project attention context ``[B, T, N, H]`` back to ``[B, T, D]``."""
  return torch.einsum("btnh,dnh->btd", context, kernel)


def _scaled_dot_product_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    valid_bias: torch.Tensor | None = None,
    per_dim_scale: torch.Tensor,
    sink_key: torch.Tensor,
    sink_value: torch.Tensor,
) -> torch.Tensor:
  """Run MRT2 attention including SequenceLayers attention sinks."""
  scale = _attention_query_scale(per_dim_scale)
  query_heads = (query * scale).permute(0, 2, 1, 3)
  key_heads = key.permute(0, 2, 1, 3)
  value_heads = value.permute(0, 2, 1, 3)

  sink_key_heads = (sink_key / scale.reshape(MRT2_HEAD_DIM)).permute(1, 0, 2)
  sink_value_heads = sink_value.permute(1, 0, 2)
  sink_key_heads = sink_key_heads.unsqueeze(0).expand(
      query_heads.shape[0], -1, -1, -1
  )
  sink_value_heads = sink_value_heads.unsqueeze(0).expand(
      value_heads.shape[0], -1, -1, -1
  )
  key_heads = torch.cat([sink_key_heads, key_heads], dim=2)
  value_heads = torch.cat([sink_value_heads, value_heads], dim=2)

  scores = torch.matmul(query_heads, key_heads.transpose(-1, -2))
  if valid_bias is not None:
    scores = scores + valid_bias.to(device=scores.device, dtype=scores.dtype)
  else:
    if valid_mask is None:
      raise ValueError("valid_mask is required when valid_bias is not provided")
    sink_mask = torch.ones(
        valid_mask.shape[:-1] + (TEMPORAL_SINKS,),
        dtype=torch.bool,
        device=valid_mask.device,
    )
    valid_mask = torch.cat([sink_mask, valid_mask], dim=-1)
    scores = torch.where(valid_mask, scores, torch.full_like(scores, -1e9))
  weights = torch.softmax(scores.to(dtype=torch.float32), dim=-1)
  context = torch.matmul(weights.to(dtype=value_heads.dtype), value_heads)
  return context.permute(0, 2, 1, 3)


class TemporalAttentionBlock(nn.Module):
  """One MRT2 temporal self-attention or streaming cross-attention block."""

  def __init__(self, arrays, layer_index: int, kind: str):
    super().__init__()
    if kind not in {"self_attention", "cross_attention"}:
      raise ValueError(f"Unsupported temporal attention kind: {kind}")
    self.kind = kind
    prefix = f"{TEMPORAL_BODY_PREFIX}/x_layers_{layer_index}/{kind}"
    self.register_buffer(
        "pre_norm_scale", _tensor(arrays[f"{prefix}/pre_norm/scale"])
    )
    self.register_buffer(
        "post_norm_scale", _tensor(arrays[f"{prefix}/post_norm/scale"])
    )
    self.register_buffer(
        "query_kernel",
        _tensor(arrays[f"{prefix}/attention/query_projection/kernel"]),
    )
    self.register_buffer(
        "key_kernel",
        _tensor(arrays[f"{prefix}/attention/key_projection/kernel"]),
    )
    self.register_buffer(
        "value_kernel",
        _tensor(arrays[f"{prefix}/attention/value_projection/kernel"]),
    )
    self.register_buffer(
        "per_dim_scale",
        _tensor(arrays[f"{prefix}/attention/per_dim_scale"]),
    )
    self.register_buffer(
        "sink_key",
        _tensor(arrays[f"{prefix}/attention/sink_key_embeddings"]),
    )
    self.register_buffer(
        "sink_value",
        _tensor(arrays[f"{prefix}/attention/sink_value_embeddings"]),
    )
    self.register_buffer(
        "output_kernel",
        _tensor(arrays[f"{prefix}/output_projection/kernel"]),
    )

  def initial_state(self, batch_size: int = 1) -> TemporalAttentionState:
    """Return an empty 41-frame K/V window for this attention block."""
    key_shape = (
        batch_size,
        MRT2_LOCAL_WINDOW_FRAMES,
        MRT2_TEMPORAL_HEADS,
        MRT2_HEAD_DIM,
    )
    return TemporalAttentionState(
        key_cache=torch.zeros(key_shape, dtype=torch.float32),
        value_cache=torch.zeros(key_shape, dtype=torch.float32),
        mask=torch.zeros(
            (batch_size, MRT2_LOCAL_WINDOW_FRAMES), dtype=torch.bool
        ),
        step=torch.zeros((batch_size,), dtype=torch.int32),
    )

  def forward(
      self,
      x: torch.Tensor,
      source: torch.Tensor,
      state: TemporalAttentionState,
  ) -> tuple[torch.Tensor, TemporalAttentionState]:
    """Run one streaming attention step and return output plus updated state."""
    normed = _rms_norm(x, self.pre_norm_scale)
    query = _project_heads(normed, self.query_kernel)
    key_source = normed if self.kind == "self_attention" else source
    key = _project_heads(key_source, self.key_kernel)
    value = _project_heads(key_source, self.value_kernel)

    combined_key = torch.cat([state.key_cache, key], dim=1)
    combined_value = torch.cat([state.value_cache, value], dim=1)
    input_mask = torch.ones(
        (x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device
    )
    combined_mask = torch.cat(
        [state.mask.to(device=x.device), input_mask], dim=1
    )
    valid_mask = combined_mask[:, None, None, :]
    if self.kind == "self_attention":
      valid_mask = valid_mask & self._self_visibility_mask(state, x.shape[1])
    else:
      valid_mask = valid_mask & self._cross_visibility_mask(
          x.shape[1], combined_key.shape[1]
      )

    context = _scaled_dot_product_attention(
        query=query,
        key=combined_key,
        value=combined_value,
        valid_mask=valid_mask,
        per_dim_scale=self.per_dim_scale,
        sink_key=self.sink_key,
        sink_value=self.sink_value,
    )
    projected = _output_projection(context, self.output_kernel)
    output = x + _rms_norm(projected, self.post_norm_scale)

    if self.kind == "self_attention":
      next_key, next_value, next_mask = self._ring_update(
          state, key, value, input_mask
      )
    else:
      next_key = combined_key[:, -MRT2_LOCAL_WINDOW_FRAMES:]
      next_value = combined_value[:, -MRT2_LOCAL_WINDOW_FRAMES:]
      next_mask = combined_mask[:, -MRT2_LOCAL_WINDOW_FRAMES:]
    next_state = TemporalAttentionState(
        key_cache=next_key,
        value_cache=next_value,
        mask=next_mask,
        step=state.step + x.shape[1],
    )
    return output, next_state

  def _self_visibility_mask(
      self,
      state: TemporalAttentionState,
      x_time: int,
  ) -> torch.Tensor:
    """Mirror SequenceLayers ring-buffer self-attention visibility."""
    kv_size = state.key_cache.shape[1]
    t0 = state.step[0].to(device=state.key_cache.device)
    newest_time_old = t0 - 1
    newest_pos_old = torch.remainder(newest_time_old, kv_size)
    phys_old = torch.arange(kv_size, device=state.key_cache.device)
    dist_old = torch.remainder(newest_pos_old - phys_old + kv_size, kv_size)
    temporal_old = newest_time_old - dist_old
    temporal_new = t0 + torch.arange(x_time, device=state.key_cache.device)
    temporal = torch.cat([temporal_old, temporal_new], dim=0)
    q_times = t0 + torch.arange(x_time, device=state.key_cache.device)
    causal = temporal[None, :] <= q_times[:, None]
    finite = temporal[None, :] >= q_times[:, None] - MRT2_LOCAL_WINDOW_FRAMES
    return (causal & finite).reshape(1, 1, x_time, kv_size + x_time)

  def _cross_visibility_mask(self, x_time: int, key_time: int) -> torch.Tensor:
    """Return the streaming cross-attention step visibility mask."""
    return torch.ones((1, 1, x_time, key_time), dtype=torch.bool)

  def _ring_update(
      self,
      state: TemporalAttentionState,
      key: torch.Tensor,
      value: torch.Tensor,
      mask: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update the self-attention cache with SequenceLayers ring semantics."""
    key_cache = state.key_cache.clone()
    value_cache = state.value_cache.clone()
    mask_cache = state.mask.clone()
    for offset in range(key.shape[1]):
      position = int((state.step[0].item() + offset) % key_cache.shape[1])
      key_cache[:, position] = key[:, offset]
      value_cache[:, position] = value[:, offset]
      mask_cache[:, position] = mask[:, offset]
    return key_cache, value_cache, mask_cache


class TemporalFFNBlock(nn.Module):
  """One non-gated MRT2 temporal feed-forward residual block."""

  def __init__(self, arrays, layer_index: int):
    super().__init__()
    prefix = f"{TEMPORAL_BODY_PREFIX}/x_layers_{layer_index}/ffn"
    self.register_buffer(
        "pre_norm_scale", _tensor(arrays[f"{prefix}/pre_norm/scale"])
    )
    self.register_buffer(
        "post_norm_scale", _tensor(arrays[f"{prefix}/post_norm/scale"])
    )
    self.register_buffer(
        "layer1_kernel", _tensor(arrays[f"{prefix}/ffn_layer1/kernel"])
    )
    self.register_buffer(
        "layer1_bias", _tensor(arrays[f"{prefix}/ffn_layer1/bias"])
    )
    self.register_buffer(
        "layer2_kernel", _tensor(arrays[f"{prefix}/ffn_layer2/kernel"])
    )
    self.register_buffer(
        "layer2_bias", _tensor(arrays[f"{prefix}/ffn_layer2/bias"])
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply RMSNorm, GELU MLP, post RMSNorm, and residual add."""
    normed = _rms_norm(x, self.pre_norm_scale)
    hidden = torch.matmul(normed, self.layer1_kernel) + self.layer1_bias
    hidden = F.gelu(hidden, approximate="tanh")
    output = torch.matmul(hidden, self.layer2_kernel) + self.layer2_bias
    return x + _rms_norm(output, self.post_norm_scale)


class TemporalTransformerLayer(nn.Module):
  """One MRT2 temporal layer: self-attn, cross-attn, then FFN."""

  def __init__(self, arrays, layer_index: int):
    super().__init__()
    self.self_attention = TemporalAttentionBlock(
        arrays, layer_index, "self_attention"
    )
    self.cross_attention = TemporalAttentionBlock(
        arrays, layer_index, "cross_attention"
    )
    self.ffn = TemporalFFNBlock(arrays, layer_index)

  def initial_state(self, batch_size: int = 1) -> TemporalLayerState:
    """Return empty self and cross K/V windows for this layer."""
    return TemporalLayerState(
        self_attention=self.self_attention.initial_state(batch_size),
        cross_attention=self.cross_attention.initial_state(batch_size),
    )

  def forward(
      self,
      x: torch.Tensor,
      source: torch.Tensor,
      state: TemporalLayerState,
  ) -> tuple[torch.Tensor, TemporalLayerState]:
    """Run one temporal layer step."""
    x, self_state = self.self_attention(x, source, state.self_attention)
    x, cross_state = self.cross_attention(x, source, state.cross_attention)
    x = self.ffn(x)
    return x, TemporalLayerState(self_state, cross_state)


class TemporalBodyStepWrapper(nn.Module):
  """PyTorch parity implementation of one MRT2 temporal-body step.

  This class proves the temporal math and state ordering against MLX. The later
  Core ML export wrapper must express state mutation as FP16 fixed-slice
  ``ct.StateType`` writes; this parity wrapper intentionally keeps readable
  Python dataclass state.
  """

  def __init__(self):
    super().__init__()
    arrays = load_checkpoint_arrays()
    self.layers = nn.ModuleList([
        TemporalTransformerLayer(arrays, layer_index)
        for layer_index in range(MRT2_TEMPORAL_LAYERS)
    ])

  def initial_state(
      self, batch_size: int = 1
  ) -> tuple[TemporalLayerState, ...]:
    """Return empty temporal K/V state for all 12 layers."""
    return tuple(layer.initial_state(batch_size) for layer in self.layers)

  def forward(
      self,
      temporal_input: torch.Tensor,
      source_encoded: torch.Tensor,
      state: tuple[TemporalLayerState, ...],
  ) -> tuple[torch.Tensor, tuple[TemporalLayerState, ...]]:
    """Run one temporal step from ``[1,1,1024]`` input and ``[1,1,256]`` source."""
    if temporal_input.shape[-1] != MRT2_MODEL_DIM:
      raise ValueError(f"Expected temporal input dim {MRT2_MODEL_DIM}")
    if source_encoded.shape[-1] != TEMPORAL_SOURCE_DIM:
      raise ValueError(f"Expected source encoded dim {TEMPORAL_SOURCE_DIM}")
    x = temporal_input.to(dtype=torch.float32)
    source = source_encoded.to(dtype=torch.float32)
    next_states = []
    for layer, layer_state in zip(self.layers, state, strict=True):
      x, next_state = layer(x, source, layer_state)
      next_states.append(next_state)
    return x, tuple(next_states)


class TemporalBodyCoreMLSlotWrapper(nn.Module):
  """Core ML export wrapper for one fixed temporal cache slot.

  Core ML accepts FP16 ``ct.StateType`` fixed-slice writes for MRT2-shaped
  ``[1, 41, 8, 128]`` K/V buffers, but this toolchain does not convert dynamic
  tensor-index writes. This wrapper is therefore bucketed by ``slot_index``:
  one exported function writes one fixed cache slot. A state-continuous 25-frame
  proof still needs either a single unrolled graph or explicit host carry;
  separate fixed-slot packages do not by themselves share ``MLState``.
  """

  def __init__(self, slot_index: int):
    super().__init__()
    if not 0 <= slot_index < TEMPORAL_COREML_SLOT_COUNT:
      raise ValueError(
          f"slot_index must be in [0, {TEMPORAL_COREML_SLOT_COUNT}), got"
          f" {slot_index}"
      )
    self.slot_index = int(slot_index)
    arrays = load_checkpoint_arrays()
    self.layers = nn.ModuleList([
        TemporalTransformerLayer(arrays, layer_index)
        for layer_index in range(MRT2_TEMPORAL_LAYERS)
    ])
    self._register_state_buffers()

  def _register_state_buffers(self) -> None:
    """Register FP16 K/V buffers that will become Core ML ``StateType`` values."""
    state_shape = (
        1,
        MRT2_LOCAL_WINDOW_FRAMES,
        MRT2_TEMPORAL_HEADS,
        MRT2_HEAD_DIM,
    )
    for layer_index in range(MRT2_TEMPORAL_LAYERS):
      for kind in ("self", "cross"):
        self.register_buffer(
            self._state_name(layer_index, kind, "key"),
            torch.zeros(state_shape, dtype=torch.float16),
        )
        self.register_buffer(
            self._state_name(layer_index, kind, "value"),
            torch.zeros(state_shape, dtype=torch.float16),
        )

  @staticmethod
  def _state_name(layer_index: int, kind: str, role: str) -> str:
    """Return a semantic Core ML state name for one temporal K/V cache."""
    return f"temporal_layer_{layer_index:02d}_{kind}_{role}_cache"

  @classmethod
  def state_names(cls) -> tuple[str, ...]:
    """Return all Core ML state names in forward execution order."""
    names: list[str] = []
    for layer_index in range(MRT2_TEMPORAL_LAYERS):
      for kind in ("self", "cross"):
        for role in ("key", "value"):
          names.append(cls._state_name(layer_index, kind, role))
    return tuple(names)

  def _state(
      self, layer_index: int, kind: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one attention cache pair as float32 for attention math."""
    key = getattr(self, self._state_name(layer_index, kind, "key")).to(
        torch.float32
    )
    value = getattr(self, self._state_name(layer_index, kind, "value")).to(
        torch.float32
    )
    return key, value

  def _write_state(
      self,
      *,
      layer_index: int,
      kind: str,
      key: torch.Tensor,
      value: torch.Tensor,
      slot_index: int | None = None,
  ) -> None:
    """Write current K/V tensors to this wrapper's fixed Core ML state slot."""
    key_cache = getattr(self, self._state_name(layer_index, kind, "key"))
    value_cache = getattr(self, self._state_name(layer_index, kind, "value"))
    if slot_index is None:
      slot_index = self.slot_index
    start = slot_index
    end = slot_index + 1
    key_cache[:, start:end] = key.to(torch.float16)
    value_cache[:, start:end] = value.to(torch.float16)

  def _valid_mask(
      self,
      device: torch.device,
      slot_index: int | None = None,
  ) -> torch.Tensor:
    """Return a no-wrap history-plus-current attention mask for this slot."""
    if slot_index is None:
      slot_index = self.slot_index
    valid = torch.zeros(
        (1, MRT2_LOCAL_WINDOW_FRAMES + 1),
        dtype=torch.bool,
        device=device,
    )
    if slot_index > 0:
      valid[:, :slot_index] = True
    valid[:, -1] = True
    return valid[:, None, None, :]

  def _valid_bias(
      self,
      device: torch.device,
      slot_index: int | None = None,
  ) -> torch.Tensor:
    """Return a fixed additive attention bias including the sink position."""
    if slot_index is None:
      slot_index = self.slot_index
    values = [0.0]
    for index in range(MRT2_LOCAL_WINDOW_FRAMES):
      values.append(0.0 if index < slot_index else -1e9)
    values.append(0.0)
    return torch.tensor(values, dtype=torch.float32, device=device).reshape(
        1,
        1,
        1,
        TEMPORAL_SINKS + MRT2_LOCAL_WINDOW_FRAMES + 1,
    )

  def _run_attention(
      self,
      *,
      block: TemporalAttentionBlock,
      x: torch.Tensor,
      source: torch.Tensor,
      old_key: torch.Tensor,
      old_value: torch.Tensor,
      slot_index: int | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one attention block using old cache plus current K/V."""
    normed = _rms_norm(x, block.pre_norm_scale)
    query = _project_heads(normed, block.query_kernel)
    key_source = normed if block.kind == "self_attention" else source
    key = _project_heads(key_source, block.key_kernel)
    value = _project_heads(key_source, block.value_kernel)
    combined_key = torch.cat([old_key, key], dim=1)
    combined_value = torch.cat([old_value, value], dim=1)
    context = _scaled_dot_product_attention(
        query=query,
        key=combined_key,
        value=combined_value,
        valid_bias=self._valid_bias(x.device, slot_index),
        per_dim_scale=block.per_dim_scale,
        sink_key=block.sink_key,
        sink_value=block.sink_value,
    )
    projected = _output_projection(context, block.output_kernel)
    output = x + _rms_norm(projected, block.post_norm_scale)
    return output, key, value

  def forward(
      self,
      temporal_input: torch.Tensor,
      source_encoded: torch.Tensor,
  ) -> torch.Tensor:
    """Run one fixed-slot temporal step and mutate 48 FP16 K/V states."""
    x = temporal_input.to(dtype=torch.float32)
    source = source_encoded.to(dtype=torch.float32)
    for layer_index, layer in enumerate(self.layers):
      self_key, self_value = self._state(layer_index, "self")
      x, new_self_key, new_self_value = self._run_attention(
          block=layer.self_attention,
          x=x,
          source=source,
          old_key=self_key,
          old_value=self_value,
          slot_index=self.slot_index,
      )
      self._write_state(
          layer_index=layer_index,
          kind="self",
          key=new_self_key,
          value=new_self_value,
          slot_index=self.slot_index,
      )

      cross_key, cross_value = self._state(layer_index, "cross")
      x, new_cross_key, new_cross_value = self._run_attention(
          block=layer.cross_attention,
          x=x,
          source=source,
          old_key=cross_key,
          old_value=cross_value,
          slot_index=self.slot_index,
      )
      self._write_state(
          layer_index=layer_index,
          kind="cross",
          key=new_cross_key,
          value=new_cross_value,
          slot_index=self.slot_index,
      )
      x = layer.ffn(x)
    return x


class TemporalBodyCoreMLUnrolledWrapper(TemporalBodyCoreMLSlotWrapper):
  """Core ML export wrapper for a no-wrap multi-frame temporal proof.

  The fixed-slot wrapper proves one state write. This wrapper keeps one model
  and one state object, then unrolls a small fixed number of frames inside one
  traced graph so later frames can depend on slots written earlier in the same
  prediction. It is a proof tool for Phase 3B, not the production decode API:
  the real-time path still wants one prediction per 40 ms frame after state
  continuity and residency are proven.
  """

  def __init__(self, frame_count: int):
    if not 1 <= frame_count <= TEMPORAL_COREML_SLOT_COUNT:
      raise ValueError(
          f"frame_count must be in [1, {TEMPORAL_COREML_SLOT_COUNT}], got"
          f" {frame_count}"
      )
    self.frame_count = int(frame_count)
    super().__init__(slot_index=0)

  def forward(
      self,
      temporal_inputs: torch.Tensor,
      source_encoded: torch.Tensor,
  ) -> torch.Tensor:
    """Run ``frame_count`` no-wrap temporal steps and mutate 48 K/V states."""
    outputs = []
    for slot_index in range(self.frame_count):
      x = temporal_inputs[:, slot_index : slot_index + 1].to(
          dtype=torch.float32
      )
      source = source_encoded[:, slot_index : slot_index + 1].to(
          dtype=torch.float32
      )
      for layer_index, layer in enumerate(self.layers):
        self_key, self_value = self._state(layer_index, "self")
        x, new_self_key, new_self_value = self._run_attention(
            block=layer.self_attention,
            x=x,
            source=source,
            old_key=self_key,
            old_value=self_value,
            slot_index=slot_index,
        )
        self._write_state(
            layer_index=layer_index,
            kind="self",
            key=new_self_key,
            value=new_self_value,
            slot_index=slot_index,
        )

        cross_key, cross_value = self._state(layer_index, "cross")
        x, new_cross_key, new_cross_value = self._run_attention(
            block=layer.cross_attention,
            x=x,
            source=source,
            old_key=cross_key,
            old_value=cross_value,
            slot_index=slot_index,
        )
        self._write_state(
            layer_index=layer_index,
            kind="cross",
            key=new_cross_key,
            value=new_cross_value,
            slot_index=slot_index,
        )
        x = layer.ffn(x)
      outputs.append(x)
    return torch.cat(outputs, dim=1)


class TemporalBodyCoreMLCarryWrapper(TemporalBodyCoreMLSlotWrapper):
  """Core ML export wrapper with host-owned temporal K/V cache tensors.

  This is the Phase 1 escape hatch after stateful multi-frame unrolls hit the
  iPhone ANE compiler cliff. It removes ``ct.StateType`` completely: all 48 K/V
  caches enter as ordinary tensors, and the graph returns one K/V update slice
  per cache for each frame in the fixed burst. The first proof uses a no-wrap
  prefix window with fixed ``history_length`` so Core ML sees only static tensor
  shapes while host code owns cache lifetime.
  """

  def __init__(self, frame_count: int, history_length: int = 0):
    if not 1 <= frame_count <= TEMPORAL_COREML_SLOT_COUNT:
      raise ValueError(
          f"frame_count must be in [1, {TEMPORAL_COREML_SLOT_COUNT}], got"
          f" {frame_count}"
      )
    if not 0 <= history_length <= TEMPORAL_COREML_SLOT_COUNT:
      raise ValueError(
          "history_length must be in "
          f"[0, {TEMPORAL_COREML_SLOT_COUNT}], got {history_length}"
      )
    if history_length + frame_count > TEMPORAL_COREML_SLOT_COUNT:
      raise ValueError(
          "history_length + frame_count must fit in the local window"
      )
    self.frame_count = int(frame_count)
    self.history_length = int(history_length)
    super().__init__(slot_index=0)

  @classmethod
  def cache_input_names(cls) -> tuple[str, ...]:
    """Return normal input names for host-owned K/V caches."""
    return tuple(f"{name}_in" for name in cls.state_names())

  @classmethod
  def cache_update_output_names(cls) -> tuple[str, ...]:
    """Return normal output names for host-owned K/V update slices."""
    return tuple(f"{name}_updates" for name in cls.state_names())

  @staticmethod
  def _attention_visibility_mask(
      *,
      device: torch.device,
      history_length: int,
      frame_index: int,
  ) -> torch.Tensor:
    """Return static no-wrap visibility for host-owned cache plus new frames."""
    valid = torch.zeros(
        (1, MRT2_LOCAL_WINDOW_FRAMES + frame_index + 1),
        dtype=torch.bool,
        device=device,
    )
    if history_length > 0:
      valid[:, :history_length] = True
    valid[:, MRT2_LOCAL_WINDOW_FRAMES:] = True
    return valid[:, None, None, :]

  def _run_carry_attention(
      self,
      *,
      block: TemporalAttentionBlock,
      x: torch.Tensor,
      source: torch.Tensor,
      old_key: torch.Tensor,
      old_value: torch.Tensor,
      new_keys: list[torch.Tensor],
      new_values: list[torch.Tensor],
      frame_index: int,
      valid_bias: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one attention block using host cache plus in-burst K/V updates."""
    normed = _rms_norm(x, block.pre_norm_scale)
    query = _project_heads(normed, block.query_kernel)
    key_source = normed if block.kind == "self_attention" else source
    key = _project_heads(key_source, block.key_kernel)
    value = _project_heads(key_source, block.value_kernel)
    if new_keys:
      combined_key = torch.cat([old_key, *new_keys, key], dim=1)
      combined_value = torch.cat([old_value, *new_values, value], dim=1)
    else:
      combined_key = torch.cat([old_key, key], dim=1)
      combined_value = torch.cat([old_value, value], dim=1)
    context = _scaled_dot_product_attention(
        query=query,
        key=combined_key,
        value=combined_value,
        valid_mask=(
            None
            if valid_bias is not None
            else self._attention_visibility_mask(
                device=x.device,
                history_length=self.history_length,
                frame_index=frame_index,
            )
        ),
        valid_bias=valid_bias,
        per_dim_scale=block.per_dim_scale,
        sink_key=block.sink_key,
        sink_value=block.sink_value,
    )
    projected = _output_projection(context, block.output_kernel)
    output = x + _rms_norm(projected, block.post_norm_scale)
    return output, key, value

  def forward(
      self,
      temporal_inputs: torch.Tensor,
      source_encoded: torch.Tensor,
      *cache_inputs: torch.Tensor,
  ) -> tuple[torch.Tensor, ...]:
    """Run a fixed burst and return temporal outputs plus K/V update slices."""
    expected_cache_count = len(self.cache_input_names())
    if len(cache_inputs) != expected_cache_count:
      raise ValueError(
          f"Expected {expected_cache_count} cache inputs, got"
          f" {len(cache_inputs)}"
      )
    cache_by_name = {
        name: cache.to(dtype=torch.float32)
        for name, cache in zip(self.state_names(), cache_inputs, strict=True)
    }
    updates_by_name: dict[str, list[torch.Tensor]] = {
        name: [] for name in self.state_names()
    }
    outputs = []
    for frame_index in range(self.frame_count):
      x = temporal_inputs[:, frame_index : frame_index + 1].to(
          dtype=torch.float32
      )
      source = source_encoded[:, frame_index : frame_index + 1].to(
          dtype=torch.float32
      )
      for layer_index, layer in enumerate(self.layers):
        self_key_name = self._state_name(layer_index, "self", "key")
        self_value_name = self._state_name(layer_index, "self", "value")
        x, new_self_key, new_self_value = self._run_carry_attention(
            block=layer.self_attention,
            x=x,
            source=source,
            old_key=cache_by_name[self_key_name],
            old_value=cache_by_name[self_value_name],
            new_keys=updates_by_name[self_key_name],
            new_values=updates_by_name[self_value_name],
            frame_index=frame_index,
        )
        updates_by_name[self_key_name].append(new_self_key.to(torch.float16))
        updates_by_name[self_value_name].append(
            new_self_value.to(torch.float16)
        )

        cross_key_name = self._state_name(layer_index, "cross", "key")
        cross_value_name = self._state_name(layer_index, "cross", "value")
        x, new_cross_key, new_cross_value = self._run_carry_attention(
            block=layer.cross_attention,
            x=x,
            source=source,
            old_key=cache_by_name[cross_key_name],
            old_value=cache_by_name[cross_value_name],
            new_keys=updates_by_name[cross_key_name],
            new_values=updates_by_name[cross_value_name],
            frame_index=frame_index,
        )
        updates_by_name[cross_key_name].append(new_cross_key.to(torch.float16))
        updates_by_name[cross_value_name].append(
            new_cross_value.to(torch.float16)
        )
        x = layer.ffn(x)
      outputs.append(x)

    temporal_outputs = torch.cat(outputs, dim=1)
    cache_updates = tuple(
        torch.cat(updates_by_name[name], dim=1) for name in self.state_names()
    )
    return (temporal_outputs, *cache_updates)


class TemporalBodyCoreMLStreamingCarryWrapper(TemporalBodyCoreMLCarryWrapper):
  """One-frame pure-function temporal step with host-owned rolling state.

  The fixed-history carry exports are useful compiler probes but require one
  separately compiled model for every history bucket. This shipping boundary
  instead accepts a static-shape additive attention bias. The host changes the
  *values* of that tensor while every graph shape remains fixed, so one model
  handles cold start, the 41-frame fill, and steady-state wraparound.

  Cache inputs are chronological: during warmup, valid entries occupy the
  prefix selected by ``cache_valid_bias``; once full, the host shifts each
  cache left by one frame and appends the returned update. The graph itself
  never mutates, gathers, scatters, rolls, or concatenates cache outputs.
  """

  cache_valid_bias_name = "cache_valid_bias"
  attention_extent = TEMPORAL_SINKS + MRT2_LOCAL_WINDOW_FRAMES + 1

  def __init__(self):
    super().__init__(frame_count=1, history_length=0)

  def forward(
      self,
      temporal_inputs: torch.Tensor,
      source_encoded: torch.Tensor,
      cache_valid_bias: torch.Tensor,
      *cache_inputs: torch.Tensor,
  ) -> tuple[torch.Tensor, ...]:
    """Run one temporal frame and return 48 K/V update slices."""
    expected_cache_count = len(self.cache_input_names())
    if len(cache_inputs) != expected_cache_count:
      raise ValueError(
          f"Expected {expected_cache_count} cache inputs, got"
          f" {len(cache_inputs)}"
      )
    cache_by_name = {
        name: cache.to(dtype=torch.float32)
        for name, cache in zip(self.state_names(), cache_inputs, strict=True)
    }
    updates_by_name: dict[str, list[torch.Tensor]] = {
        name: [] for name in self.state_names()
    }
    x = temporal_inputs[:, :1].to(dtype=torch.float32)
    source = source_encoded[:, :1].to(dtype=torch.float32)
    bias = cache_valid_bias.to(dtype=torch.float32)
    for layer_index, layer in enumerate(self.layers):
      self_key_name = self._state_name(layer_index, "self", "key")
      self_value_name = self._state_name(layer_index, "self", "value")
      x, new_self_key, new_self_value = self._run_carry_attention(
          block=layer.self_attention,
          x=x,
          source=source,
          old_key=cache_by_name[self_key_name],
          old_value=cache_by_name[self_value_name],
          new_keys=updates_by_name[self_key_name],
          new_values=updates_by_name[self_value_name],
          frame_index=0,
          valid_bias=bias,
      )
      updates_by_name[self_key_name].append(new_self_key.to(torch.float16))
      updates_by_name[self_value_name].append(new_self_value.to(torch.float16))

      cross_key_name = self._state_name(layer_index, "cross", "key")
      cross_value_name = self._state_name(layer_index, "cross", "value")
      x, new_cross_key, new_cross_value = self._run_carry_attention(
          block=layer.cross_attention,
          x=x,
          source=source,
          old_key=cache_by_name[cross_key_name],
          old_value=cache_by_name[cross_value_name],
          new_keys=updates_by_name[cross_key_name],
          new_values=updates_by_name[cross_value_name],
          frame_index=0,
          valid_bias=bias,
      )
      updates_by_name[cross_key_name].append(new_cross_key.to(torch.float16))
      updates_by_name[cross_value_name].append(
          new_cross_value.to(torch.float16)
      )
      x = layer.ffn(x)

    cache_updates = tuple(
        torch.cat(updates_by_name[name], dim=1) for name in self.state_names()
    )
    return (x, *cache_updates)
