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
