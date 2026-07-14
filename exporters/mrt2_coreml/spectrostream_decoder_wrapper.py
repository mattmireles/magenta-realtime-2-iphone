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

"""PyTorch mirror of the MLX SpectroStream decoder conv boundary.

This module exists only as a Core ML conversion source for the MRT2 Core ML port.
The behavioral reference remains
``magenta_rt.mlx.spectrostream.modeling.spectrostream_decoder_config`` plus the
weights loaded by ``magenta_rt.mlx.spectrostream.load_weights``.

The wrapper deliberately mirrors the MLX SequenceLayers **layer-mode** decoder
from host-owned RVQ embeddings ``[B, T, 256]`` to the pre-iSTFT STFT tensor
``[B, T * 4 - 4, 480, 4]`` for the current 40 ms config. RVQ lookup and iSTFT /
overlap-add stay outside this graph per ``README/Guides/RVQ-codec-decoder-guide.md``.
Streaming conv state is a later runtime design; this proof exports a fixed-shape
chunk baseline first so Core ML compatibility and GPU placement can be measured
without hiding gather or iSTFT fallback inside the graph.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SPECTROSTREAM_EMBEDDING_DIM = 256
SPECTROSTREAM_NUM_BINS = 480
SPECTROSTREAM_OUTPUT_CHANNELS = 4
SPECTROSTREAM_DECODER_TIME_STRIDE = 4
SPECTROSTREAM_DECODER_LOOKAHEAD_FRAMES = 1


def _effective_kernel_size(kernel_size: int, dilation_rate: int) -> int:
  """Return the SequenceLayers effective kernel size."""
  return (kernel_size - 1) * dilation_rate + 1


def _explicit_padding(
    padding: str | tuple[int, int],
    kernel_size: int,
    stride: int,
    dilation_rate: int,
) -> tuple[int, int]:
  """Return ``(left, right)`` padding matching SequenceLayers MLX."""
  if not isinstance(padding, str):
    return tuple(int(v) for v in padding)

  effective_kernel = _effective_kernel_size(kernel_size, dilation_rate)
  if padding in ("causal_valid", "causal"):
    return (effective_kernel - 1, 0)
  if padding == "semicausal":
    left = max(effective_kernel - stride, 0)
    return (left, effective_kernel - 1 - left)
  if padding in ("reverse_causal_valid", "reverse_causal"):
    return (0, effective_kernel - 1)
  if padding == "same":
    amount = effective_kernel - 1
    left = amount // 2
    return (left, amount - left)
  if padding == "valid":
    return (0, 0)
  if padding == "semicausal_full":
    return (effective_kernel - stride, effective_kernel - 1)
  raise ValueError(f"Unsupported padding mode: {padding}")


def _transpose_conv_trim(
    padding: str | tuple[int, int],
    kernel_size: int,
    stride: int,
    dilation_rate: int,
) -> tuple[int, int]:
  """Return SequenceLayers transpose-conv output trim."""
  if not isinstance(padding, str):
    return tuple(int(v) for v in padding)

  effective_kernel = _effective_kernel_size(kernel_size, dilation_rate)
  if padding == "valid":
    return (0, 0)
  if padding == "same":
    trim = effective_kernel - stride
    left = trim // 2
    return (left, trim - left)
  if padding == "causal":
    return (0, max(0, effective_kernel - stride))
  if padding == "reverse_causal":
    return (max(0, effective_kernel - stride), 0)
  raise ValueError(f"Unsupported transpose padding mode: {padding}")


def _as_numpy(value: object) -> np.ndarray:
  """Materialize an MLX array-like object as a NumPy array."""
  return np.asarray(value, dtype=np.float32)


def _ensure_inner(layer: object) -> object:
  """Unwrap deferred SequenceLayers wrappers after they have been built."""
  inner = getattr(layer, "inner", None)
  return inner if inner is not None else layer


class TorchSerial(nn.Module):
  """Run child modules in SequenceLayers serial order."""

  def __init__(self, layers: Iterable[nn.Module]):
    super().__init__()
    self.layers = nn.ModuleList(layers)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      values = layer(values)
    return values


class TorchResidual(nn.Module):
  """Mirror SequenceLayers ``Residual``: ``body(x) + shortcut(x)``."""

  def __init__(self, body: nn.Module, shortcut: nn.Module):
    super().__init__()
    self.body = body
    self.shortcut = shortcut

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    return self.body(values) + self.shortcut(values)


class TorchIdentity(nn.Module):
  """Identity layer for missing SequenceLayers shortcuts."""

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    return values


class TorchElu(nn.Module):
  """ELU activation with the alpha used by SequenceLayers."""

  def __init__(self, alpha: float):
    super().__init__()
    self.alpha = float(alpha)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    return F.elu(values, alpha=self.alpha)


class TorchScale(nn.Module):
  """Elementwise constant multiply (one FP16-safe ``mul`` op after tracing).

  Used in pairs by ``apply_fp16_safe_rescale``: ``TorchScale(1/S)`` enters the
  rescaled region and ``TorchScale(S)`` restores native magnitudes after it.
  """

  def __init__(self, scale: float):
    super().__init__()
    self.scale = float(scale)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    return values * self.scale


class TorchScaledElu(nn.Module):
  """Exact ELU for a ``1/S``-scaled stream, FP16-overflow-safe by construction.

  On a stream ``y = x/S`` this computes ``elu_alpha(x)/S`` as::

    relu(y) + (alpha/S) * (exp(clamp(y, max=0) * S) - 1)

  The clamp keeps the exponent argument <= 0 so ``exp`` never overflows; if
  the FP16 multiply saturates a large-magnitude negative to ``-inf``,
  ``exp(-inf) = 0`` and the branch lands exactly on its mathematical limit
  ``-alpha/S``. Replaces ``TorchElu`` inside the region chosen by
  ``apply_fp16_safe_rescale``.

  Why not just retune ELU's alpha to ``alpha/S``: that changes the exponent
  slope (``e**y`` vs ``e**(S*y)``) and the resulting transition-band error,
  amplified through the downstream convs, measured 35.6 dB SNR vs the FP32
  reference on the standard fixture (2026-06-12). This formulation is exact.
  """

  def __init__(self, alpha: float, scale: float):
    super().__init__()
    self.alpha = float(alpha)
    self.scale = float(scale)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    negative = torch.clamp(values, max=0.0) * self.scale
    return F.relu(values) + (self.alpha / self.scale) * (
        torch.exp(negative) - 1.0
    )


class TorchExpandDims(nn.Module):
  """Expand channel dimensions while preserving batch/time axes."""

  def __init__(self, axes: Iterable[int]):
    super().__init__()
    self.axes = tuple(int(axis) for axis in axes)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    rank = values.dim() - 2
    axes = sorted({axis + rank + 1 if axis < 0 else axis for axis in self.axes})
    for axis in axes:
      values = values.unsqueeze(2 + axis)
    return values


class TorchReshape(nn.Module):
  """Reshape SequenceLayers channel dimensions only."""

  def __init__(self, output_shape: Iterable[int]):
    super().__init__()
    self.output_shape = tuple(int(value) for value in output_shape)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    batch, time = values.shape[:2]
    return values.reshape((batch, time) + self.output_shape)


class TorchUpsample2D(nn.Module):
  """Nearest-neighbor 2D upsample over time and frequency dimensions."""

  def __init__(self, rate: tuple[int, int]):
    super().__init__()
    self.rate = (int(rate[0]), int(rate[1]))

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    nchw = values.permute(0, 3, 1, 2).contiguous()
    upsampled = F.interpolate(nchw, scale_factor=self.rate, mode="nearest")
    return upsampled.permute(0, 2, 3, 1).contiguous()


class TorchUpsample2DNCHW(nn.Module):
  """Nearest-neighbor 2D upsample for already-channel-first tensors."""

  def __init__(self, source: TorchUpsample2D):
    super().__init__()
    self.rate = source.rate

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    return F.interpolate(values, scale_factor=self.rate, mode="nearest")


class TorchLookahead(nn.Module):
  """Layer-mode SpectroStream lookahead drops leading decoder frames."""

  def __init__(self, length: int):
    super().__init__()
    self.length = int(length)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    if self.length == 0:
      return values
    return values[:, self.length :]


class TorchConv2D(nn.Module):
  """MLX channel-last Conv2D expressed through PyTorch ``conv2d``."""

  def __init__(self, mlx_layer: object):
    super().__init__()
    layer = _ensure_inner(mlx_layer)
    kernel = _as_numpy(layer.kernel)
    self.register_buffer(
        "weight",
        torch.from_numpy(np.transpose(kernel, (0, 3, 1, 2)).copy()),
    )
    bias = getattr(layer, "bias", None)
    if bias is None:
      self.bias = None
    else:
      self.register_buffer("bias", torch.from_numpy(_as_numpy(bias).copy()))
    self.strides = tuple(int(value) for value in layer.strides)
    self.dilation_rate = tuple(int(value) for value in layer.dilation_rate)
    self.time_padding = layer.time_padding
    self.spatial_padding = layer.spatial_padding
    self.groups = int(layer.groups)
    self.kernel_size = tuple(int(value) for value in layer.kernel_size)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    time_pad = _explicit_padding(
        self.time_padding,
        self.kernel_size[0],
        self.strides[0],
        self.dilation_rate[0],
    )
    spatial_pad = _explicit_padding(
        self.spatial_padding,
        self.kernel_size[1],
        self.strides[1],
        self.dilation_rate[1],
    )
    nchw = values.permute(0, 3, 1, 2).contiguous()
    nchw = F.pad(nchw, (spatial_pad[0], spatial_pad[1], time_pad[0], time_pad[1]))
    out = F.conv2d(
        nchw,
        self.weight,
        self.bias,
        stride=self.strides,
        padding=0,
        dilation=self.dilation_rate,
        groups=self.groups,
    )
    return out.permute(0, 2, 3, 1).contiguous()


class TorchConv2DNCHW(nn.Module):
  """Conv2D that keeps the whole operation in NCHW layout."""

  def __init__(self, source: TorchConv2D):
    super().__init__()
    self.register_buffer("weight", source.weight.detach().clone())
    if source.bias is None:
      self.bias = None
    else:
      self.register_buffer("bias", source.bias.detach().clone())
    self.strides = source.strides
    self.dilation_rate = source.dilation_rate
    self.time_padding = source.time_padding
    self.spatial_padding = source.spatial_padding
    self.groups = source.groups
    self.kernel_size = source.kernel_size

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    time_pad = _explicit_padding(
        self.time_padding,
        self.kernel_size[0],
        self.strides[0],
        self.dilation_rate[0],
    )
    spatial_pad = _explicit_padding(
        self.spatial_padding,
        self.kernel_size[1],
        self.strides[1],
        self.dilation_rate[1],
    )
    values = F.pad(values, (spatial_pad[0], spatial_pad[1], time_pad[0], time_pad[1]))
    return F.conv2d(
        values,
        self.weight,
        self.bias,
        stride=self.strides,
        padding=0,
        dilation=self.dilation_rate,
        groups=self.groups,
    )


class TorchConv2DTranspose(nn.Module):
  """MLX channel-last Conv2DTranspose expressed through PyTorch."""

  def __init__(self, mlx_layer: object):
    super().__init__()
    layer = _ensure_inner(mlx_layer)
    kernel = _as_numpy(layer.kernel)
    self.register_buffer(
        "weight",
        torch.from_numpy(np.transpose(kernel, (3, 0, 1, 2)).copy()),
    )
    bias = getattr(layer, "bias", None)
    if bias is None:
      self.bias = None
    else:
      self.register_buffer("bias", torch.from_numpy(_as_numpy(bias).copy()))
    self.strides = tuple(int(value) for value in layer.strides)
    self.dilation_rate = tuple(int(value) for value in layer.dilation_rate)
    self.time_padding = layer.time_padding
    self.spatial_padding = layer.spatial_padding
    self.groups = int(layer.groups)
    self.kernel_size = tuple(int(value) for value in layer.kernel_size)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    nchw = values.permute(0, 3, 1, 2).contiguous()
    out = F.conv_transpose2d(
        nchw,
        self.weight,
        self.bias,
        stride=self.strides,
        padding=0,
        dilation=self.dilation_rate,
        groups=self.groups,
    )
    out = out.permute(0, 2, 3, 1).contiguous()
    time_trim = _transpose_conv_trim(
        self.time_padding,
        self.kernel_size[0],
        self.strides[0],
        self.dilation_rate[0],
    )
    spatial_trim = _transpose_conv_trim(
        self.spatial_padding,
        self.kernel_size[1],
        self.strides[1],
        self.dilation_rate[1],
    )
    if time_trim[0] > 0:
      out = out[:, time_trim[0] :]
    if time_trim[1] > 0:
      out = out[:, :-time_trim[1]]
    if spatial_trim[0] > 0:
      out = out[:, :, spatial_trim[0] :]
    if spatial_trim[1] > 0:
      out = out[:, :, :-spatial_trim[1]]
    return out


class TorchConv2DTransposeNCHW(nn.Module):
  """Conv2DTranspose that keeps the whole operation in NCHW layout."""

  def __init__(self, source: TorchConv2DTranspose):
    super().__init__()
    self.register_buffer("weight", source.weight.detach().clone())
    if source.bias is None:
      self.bias = None
    else:
      self.register_buffer("bias", source.bias.detach().clone())
    self.strides = source.strides
    self.dilation_rate = source.dilation_rate
    self.time_padding = source.time_padding
    self.spatial_padding = source.spatial_padding
    self.groups = source.groups
    self.kernel_size = source.kernel_size

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    out = F.conv_transpose2d(
        values,
        self.weight,
        self.bias,
        stride=self.strides,
        padding=0,
        dilation=self.dilation_rate,
        groups=self.groups,
    )
    time_trim = _transpose_conv_trim(
        self.time_padding,
        self.kernel_size[0],
        self.strides[0],
        self.dilation_rate[0],
    )
    spatial_trim = _transpose_conv_trim(
        self.spatial_padding,
        self.kernel_size[1],
        self.strides[1],
        self.dilation_rate[1],
    )
    if time_trim[0] > 0:
      out = out[:, :, time_trim[0] :]
    if time_trim[1] > 0:
      out = out[:, :, :-time_trim[1]]
    if spatial_trim[0] > 0:
      out = out[:, :, :, spatial_trim[0] :]
    if spatial_trim[1] > 0:
      out = out[:, :, :, :-spatial_trim[1]]
    return out


class TorchParallelChannels(nn.Module):
  """Apply one shared child module to channel groups and concatenate outputs."""

  def __init__(self, child: nn.Module, num_groups: int):
    super().__init__()
    self.child = child
    self.num_groups = int(num_groups)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    group_size = values.shape[-1] // self.num_groups
    groups = torch.split(values, group_size, dim=-1)
    return torch.cat([self.child(group) for group in groups], dim=-1)


class TorchParallelChannelsNCHW(nn.Module):
  """Apply one shared child module to NCHW channel groups."""

  def __init__(self, child: nn.Module, num_groups: int):
    super().__init__()
    self.child = child
    self.num_groups = int(num_groups)

  def forward(self, values: torch.Tensor) -> torch.Tensor:
    group_size = values.shape[1] // self.num_groups
    groups = torch.split(values, group_size, dim=1)
    return torch.cat([self.child(group) for group in groups], dim=1)


def _to_nchw_layer(layer: nn.Module) -> nn.Module:
  """Convert a channel-last decoder subgraph to channel-first internally."""
  if isinstance(layer, TorchSerial):
    return TorchSerial(_to_nchw_layer(child) for child in layer.layers)
  if isinstance(layer, TorchResidual):
    return TorchResidual(_to_nchw_layer(layer.body), _to_nchw_layer(layer.shortcut))
  if isinstance(layer, TorchIdentity):
    return TorchIdentity()
  if isinstance(layer, TorchElu):
    return TorchElu(layer.alpha)
  if isinstance(layer, TorchUpsample2D):
    return TorchUpsample2DNCHW(layer)
  if isinstance(layer, TorchParallelChannels):
    return TorchParallelChannelsNCHW(_to_nchw_layer(layer.child), layer.num_groups)
  if isinstance(layer, TorchConv2D):
    return TorchConv2DNCHW(layer)
  if isinstance(layer, TorchConv2DTranspose):
    return TorchConv2DTransposeNCHW(layer)
  raise TypeError(f"Unsupported NCHW decoder layer type: {type(layer).__name__}")


class SpectroStreamDecoderConvWrapper(nn.Module):
  """Traceable fixed-chunk SpectroStream decoder conv wrapper."""

  def __init__(self, decoder: nn.Module):
    super().__init__()
    self.decoder = decoder

  def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
    """Decode host-owned RVQ embeddings to the pre-iSTFT STFT tensor."""
    return self.decoder(embeddings)

  @classmethod
  def from_mlx_decoder(cls, mlx_decoder: object) -> "SpectroStreamDecoderConvWrapper":
    """Build a PyTorch wrapper from a weight-loaded MLX decoder."""
    return cls(_from_mlx_layer(mlx_decoder))


class SpectroStreamDecoderPrefixWrapper(nn.Module):
  """Traceable prefix of the SpectroStream decoder conv stack.

  This is an ANE placement probe for the early decoder layers. It deliberately
  stops before the full decoder output so we can determine whether a future
  decoder split should keep early dense/conv work in Core ML and leave the
  numerically fragile upsampling tail elsewhere.
  """

  def __init__(self, decoder: TorchSerial, layer_count: int):
    super().__init__()
    if layer_count <= 0 or layer_count > len(decoder.layers):
      raise ValueError("layer_count must select a non-empty decoder prefix")
    self.layer_count = int(layer_count)
    self.layers = nn.ModuleList(list(decoder.layers[: self.layer_count]))

  def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
    """Decode host-owned RVQ embeddings through the configured prefix."""
    values = embeddings
    for layer in self.layers:
      values = layer(values)
    return values

  @classmethod
  def from_mlx_decoder(
      cls,
      mlx_decoder: object,
      layer_count: int,
  ) -> "SpectroStreamDecoderPrefixWrapper":
    """Build a decoder-prefix wrapper from a weight-loaded MLX decoder."""
    decoder = _from_mlx_layer(mlx_decoder)
    if not isinstance(decoder, TorchSerial):
      raise TypeError(f"Expected TorchSerial decoder, got {type(decoder).__name__}")
    return cls(decoder, layer_count)


class SpectroStreamDecoderNCHWParallelPrefixWrapper(nn.Module):
  """Decoder prefix that keeps one ParallelChannels layer channel-first.

  This is a focused ANE placement probe for the large SpectroStream upsampling
  block. It preserves the public channel-last prefix input/output contract while
  deleting repeated channel-last/NCHW conversions inside the selected parallel
  layer.
  """

  def __init__(self, decoder: TorchSerial, parallel_layer: int):
    super().__init__()
    if parallel_layer <= 0 or parallel_layer >= len(decoder.layers):
      raise ValueError("parallel_layer must select a non-initial decoder layer")
    selected = decoder.layers[parallel_layer]
    if not isinstance(selected, TorchParallelChannels):
      raise TypeError(
          "parallel_layer must select a TorchParallelChannels layer, "
          f"got {type(selected).__name__}"
      )
    self.parallel_layer = int(parallel_layer)
    self.prefix_layers = nn.ModuleList(list(decoder.layers[: self.parallel_layer]))
    self.parallel_nchw = _to_nchw_layer(selected)

  def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
    """Run layers before the parallel block, then the block in NCHW."""
    values = embeddings
    for layer in self.prefix_layers:
      values = layer(values)
    nchw = values.permute(0, 3, 1, 2).contiguous()
    nchw = self.parallel_nchw(nchw)
    return nchw.permute(0, 2, 3, 1).contiguous()

  @classmethod
  def from_mlx_decoder(
      cls,
      mlx_decoder: object,
      parallel_layer: int,
  ) -> "SpectroStreamDecoderNCHWParallelPrefixWrapper":
    """Build an NCHW-parallel prefix wrapper from a weight-loaded MLX decoder."""
    decoder = _from_mlx_layer(mlx_decoder)
    if not isinstance(decoder, TorchSerial):
      raise TypeError(f"Expected TorchSerial decoder, got {type(decoder).__name__}")
    return cls(decoder, parallel_layer)


class SpectroStreamDecoderNCHWParallelWrapper(nn.Module):
  """Full decoder wrapper with one ParallelChannels layer kept channel-first."""

  def __init__(self, decoder: TorchSerial, parallel_layer: int):
    super().__init__()
    if parallel_layer <= 0 or parallel_layer >= len(decoder.layers):
      raise ValueError("parallel_layer must select a non-initial decoder layer")
    selected = decoder.layers[parallel_layer]
    if not isinstance(selected, TorchParallelChannels):
      raise TypeError(
          "parallel_layer must select a TorchParallelChannels layer, "
          f"got {type(selected).__name__}"
      )
    self.parallel_layer = int(parallel_layer)
    self.prefix_layers = nn.ModuleList(list(decoder.layers[: self.parallel_layer]))
    self.parallel_nchw = _to_nchw_layer(selected)
    self.suffix_layers = nn.ModuleList(list(decoder.layers[self.parallel_layer + 1 :]))

  def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
    """Run a full decoder while keeping the selected parallel block in NCHW."""
    values = embeddings
    for layer in self.prefix_layers:
      values = layer(values)
    nchw = values.permute(0, 3, 1, 2).contiguous()
    nchw = self.parallel_nchw(nchw)
    values = nchw.permute(0, 2, 3, 1).contiguous()
    for layer in self.suffix_layers:
      values = layer(values)
    return values

  @classmethod
  def from_mlx_decoder(
      cls,
      mlx_decoder: object,
      parallel_layer: int,
  ) -> "SpectroStreamDecoderNCHWParallelWrapper":
    """Build an NCHW-parallel full wrapper from a weight-loaded MLX decoder."""
    decoder = _from_mlx_layer(mlx_decoder)
    if not isinstance(decoder, TorchSerial):
      raise TypeError(f"Expected TorchSerial decoder, got {type(decoder).__name__}")
    return cls(decoder, parallel_layer)


#: Default stream scale for ``apply_fp16_safe_rescale``. The hot region peaks
#: at ~2.65e6 on the standard 25-frame fixture; 1/128 brings that to ~20.7k,
#: a 3.2x margin under FP16 max (65504), while keeping the region's quiet
#: content far above the FP16 subnormal floor.
FP16_RESCALE_DEFAULT_SCALE = 128.0
#: A top-level child block whose subtree exceeds this absmax joins the
#: rescaled region (FP16 max is 65504; this leaves ~2x content headroom).
FP16_RESCALE_ENTER_THRESHOLD = 30000.0
#: The restore boundary must satisfy ``native_absmax * margin <= 65504`` so
#: the exit ``TorchScale(S)`` multiply cannot itself overflow.
FP16_RESCALE_RESTORE_MARGIN = 2.0
_FP16_MAX = 65504.0


def _replace_elus_with_scaled(module: nn.Module, scale: float) -> int:
  """Recursively swap every ``TorchElu`` under ``module`` for the exact
  ``TorchScaledElu`` equivalent. Returns the number swapped."""
  swapped = 0
  for name, child in module.named_children():
    if isinstance(child, TorchElu):
      setattr(module, name, TorchScaledElu(child.alpha, scale))
      swapped += 1
    else:
      swapped += _replace_elus_with_scaled(child, scale)
  return swapped


def apply_fp16_safe_rescale(
    wrapper: "SpectroStreamDecoderNCHWParallelWrapper",
    example: torch.Tensor,
    scale: float = FP16_RESCALE_DEFAULT_SCALE,
    enter_threshold: float = FP16_RESCALE_ENTER_THRESHOLD,
    restore_margin: float = FP16_RESCALE_RESTORE_MARGIN,
) -> dict:
  """Make the decoder FP16-convertible by rescaling its hot mid-network.

  Why this exists: the SpectroStream decoder's residual upsampling blocks
  produce activations up to ~2.65e6 on real content — 40x past FP16 max — so
  a plain ``compute_precision=FLOAT16`` Core ML conversion emits NaN/Inf (the
  2026-06-08 ``finite_ratio=0.843`` failure documented in
  ``README/Notes/aperture-v0-phase5-real-audio.md``). FP32 cannot run on the
  ANE at all, which is why this transform exists: it is the gate between the
  decoder and the Apple Neural Engine.

  The transform is EXACT in FP32 (measured 180 dB SNR vs the untouched
  wrapper on the standard fixture; FP16 then measures ~62 dB, the FP16 grid
  itself). It rescales the stream by ``1/scale`` across the hot region only:

  * ``TorchScale(1/scale)`` / ``TorchScale(scale)`` are inserted at TOP-LEVEL
    serial boundaries of ``wrapper.parallel_nchw.child`` — the only points
    that dominate all dataflow. Scaling inside a residual branch mixes scales
    at the residual add and destroys the function (measured -6.5 dB SNR).
  * Region conv/convT WEIGHTS are untouched (their input is already scaled)
    but every region BIAS is multiplied by ``1/scale`` — bias adds at output
    scale; forgetting the shortcuts' conv biases cost 0.5-1.5% error.
  * Region ``TorchElu``s become ``TorchScaledElu`` (exact, see its docstring).

  Boundary selection runs live on ``example`` (use the standard 25-frame
  fixture from ``scripts/convert_spectrostream_decoder_conv_coreml.py``):
  entry before the first top-level block whose subtree worst absmax exceeds
  ``enter_threshold``; exit after the last such block, advanced until the
  boundary's native absmax restores safely. Returns a metadata dict (region,
  scale, swap/bias counts, measured boundary maxima) for the export report.
  """
  child = wrapper.parallel_nchw.child
  if not isinstance(child, TorchSerial):
    raise TypeError(f"Expected TorchSerial child, got {type(child).__name__}")
  blocks = list(child.layers)

  # Profile: worst absmax per top-level block subtree + each block's output.
  subtree_worst = [0.0] * len(blocks)
  boundary_out = [0.0] * len(blocks)

  def _make_subtree_hook(index: int):
    def hook(_module, _inputs, output):
      if isinstance(output, torch.Tensor):
        subtree_worst[index] = max(
            subtree_worst[index], float(output.detach().abs().max())
        )
    return hook

  def _make_boundary_hook(index: int):
    def hook(_module, _inputs, output):
      if isinstance(output, torch.Tensor):
        boundary_out[index] = max(
            boundary_out[index], float(output.detach().abs().max())
        )
    return hook

  handles = []
  for index, block in enumerate(blocks):
    handles.append(block.register_forward_hook(_make_boundary_hook(index)))
    for module in block.modules():
      handles.append(module.register_forward_hook(_make_subtree_hook(index)))
  wrapper.eval()
  with torch.no_grad():
    wrapper(example)
  for handle in handles:
    handle.remove()

  hot = [i for i, worst in enumerate(subtree_worst) if worst > enter_threshold]
  if not hot:
    raise RuntimeError(
        f"No decoder block exceeds {enter_threshold}; rescale is unnecessary"
    )
  entry = hot[0]
  exit_block = hot[-1]
  while boundary_out[exit_block] * restore_margin > _FP16_MAX:
    exit_block += 1
    if exit_block >= len(blocks):
      raise RuntimeError("No safe restore boundary inside the decoder child")

  region = blocks[entry : exit_block + 1]
  swapped = sum(_replace_elus_with_scaled(block, scale) for block in region)
  biased = 0
  seen: set[int] = set()
  for block in region:
    for module in block.modules():
      if id(module) in seen:
        continue
      seen.add(id(module))
      bias = getattr(module, "bias", None)
      if isinstance(bias, torch.Tensor):
        bias.mul_(1.0 / scale)
        biased += 1

  child.layers = nn.ModuleList(
      blocks[:entry]
      + [TorchScale(1.0 / scale)]
      + region
      + [TorchScale(scale)]
      + blocks[exit_block + 1 :]
  )
  return {
      "scale": float(scale),
      "entry_block": int(entry),
      "exit_block": int(exit_block),
      "elus_swapped": int(swapped),
      "biases_scaled": int(biased),
      "subtree_worst_absmax": [float(value) for value in subtree_worst],
      "boundary_out_absmax": [float(value) for value in boundary_out],
  }


class SpectroStreamDecoderTailWrapper(nn.Module):
  """Traceable suffix of the SpectroStream decoder conv stack.

  The tail wrapper pairs with ``SpectroStreamDecoderPrefixWrapper`` for split
  decoder experiments: run the early conv/upsample prefix as an ANE island, then
  run the numerically sensitive tail with a safer precision/compute policy.
  """

  def __init__(self, decoder: TorchSerial, start_layer: int):
    super().__init__()
    if start_layer <= 0 or start_layer >= len(decoder.layers):
      raise ValueError("start_layer must leave a non-empty decoder tail")
    self.start_layer = int(start_layer)
    self.layers = nn.ModuleList(list(decoder.layers[self.start_layer :]))

  def forward(self, prefix_output: torch.Tensor) -> torch.Tensor:
    """Decode an intermediate prefix tensor through the remaining decoder tail."""
    values = prefix_output
    for layer in self.layers:
      values = layer(values)
    return values

  @classmethod
  def from_mlx_decoder(
      cls,
      mlx_decoder: object,
      start_layer: int,
  ) -> "SpectroStreamDecoderTailWrapper":
    """Build a decoder-tail wrapper from a weight-loaded MLX decoder."""
    decoder = _from_mlx_layer(mlx_decoder)
    if not isinstance(decoder, TorchSerial):
      raise TypeError(f"Expected TorchSerial decoder, got {type(decoder).__name__}")
    return cls(decoder, start_layer)


def _from_mlx_layer(layer: object) -> nn.Module:
  """Convert a built MLX SequenceLayer object into a PyTorch module."""
  layer = _ensure_inner(layer)
  name = type(layer).__name__

  if name in ("Serial", "SerialModules"):
    return TorchSerial(_from_mlx_layer(child) for child in layer.layers)
  if name == "Residual":
    return TorchResidual(_from_mlx_layer(layer.body), _from_mlx_layer(layer.shortcut))
  if name == "Identity":
    return TorchIdentity()
  if name == "Elu":
    return TorchElu(getattr(layer, "_alpha", 1.0))
  if name == "ExpandDims":
    return TorchExpandDims(getattr(layer, "_axis"))
  if name == "Reshape":
    return TorchReshape(getattr(layer, "_output_shape"))
  if name == "Upsample2D":
    return TorchUpsample2D(getattr(layer, "_rate"))
  if name == "Lookahead":
    return TorchLookahead(getattr(layer, "length"))
  if name == "ParallelChannels":
    combination = int(getattr(layer, "_combination"))
    if combination != 2:
      raise ValueError(f"Unsupported ParallelChannels combination: {combination}")
    return TorchParallelChannels(_from_mlx_layer(layer.child), getattr(layer, "_num_groups"))
  if name == "Conv2D":
    return TorchConv2D(layer)
  if name == "Conv2DTranspose":
    return TorchConv2DTranspose(layer)

  raise TypeError(f"Unsupported MLX decoder layer type: {name}")


def decoder_output_frames(input_frames: int) -> int:
  """Return layer-mode pre-iSTFT decoder frames for the fixed 40 ms config."""
  return input_frames * SPECTROSTREAM_DECODER_TIME_STRIDE - (
      SPECTROSTREAM_DECODER_LOOKAHEAD_FRAMES * SPECTROSTREAM_DECODER_TIME_STRIDE
  )


def count_torch_conv_layers(module: nn.Module) -> dict[str, int]:
  """Count converted convolutional layers for metadata sanity checks."""
  conv2d = sum(isinstance(child, TorchConv2D) for child in module.modules())
  transpose = sum(isinstance(child, TorchConv2DTranspose) for child in module.modules())
  return {"conv2d": conv2d, "conv2d_transpose": transpose}
