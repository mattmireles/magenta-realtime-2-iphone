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

"""CPU-owned Depthformer sampling for the Core ML proof.

Cross-file contract:

- ``magenta_rt/mlx/depthformer.py`` keeps sampling outside the heavy temporal
  math: each RVQ level samples from the full vocabulary after applying a
  level-specific valid range.
- ``magenta_rt/coreml/depthformer_wrapper.py`` and the Core ML export keep RNG,
  top-k/top-p, CFG, and valid-range masking outside the model graph.

This module is the host-side replacement for that dynamic sampling logic. It
operates on NumPy arrays so the same code can be called from Python validation
scripts and later from a Swift/C++ port with the same constants.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np
import numpy.typing as npt


MRT2_RVQ_LEVELS = 12
MRT2_CODEBOOK_SIZE = 1_024
MRT2_RESERVED_TOKENS = 6
MRT2_DEPTHFORMER_LOGIT_SIZE = MRT2_RVQ_LEVELS * MRT2_CODEBOOK_SIZE + MRT2_RESERVED_TOKENS
MRT2_DEFAULT_TOP_K = 40
MRT2_DEFAULT_TEMPERATURE = 1.3
MRT2_DETERMINISTIC_TEMPERATURE = 0.0


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
  """Sampling parameters for one CPU-owned Depthformer frame.

  ``temperature=0`` selects deterministic argmax after masking. Positive
  temperatures match the MLX implementation's Gumbel-max form:
  ``argmax(logits + gumbel_noise * temperature)``.
  """

  temperature: float = MRT2_DEFAULT_TEMPERATURE
  top_k: int | None = MRT2_DEFAULT_TOP_K
  top_p: float | None = None
  seed: int = 0

  def __post_init__(self) -> None:
    """Reject invalid sampler configurations early."""
    if self.temperature < 0.0:
      raise ValueError("temperature must be non-negative")
    if self.top_k is not None and self.top_k <= 0:
      raise ValueError("top_k must be positive when provided")
    if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
      raise ValueError("top_p must be in (0, 1] when provided")
    if self.top_k is not None and self.top_p is not None:
      raise ValueError("Only one of top_k or top_p may be set")


def valid_range_for_rvq_level(
    rvq_level: int,
    *,
    reserved_tokens: int = MRT2_RESERVED_TOKENS,
    codebook_size: int = MRT2_CODEBOOK_SIZE,
) -> tuple[int, int]:
  """Return the half-open valid vocabulary range for one RVQ level."""
  if not 0 <= rvq_level < MRT2_RVQ_LEVELS:
    raise ValueError(f"rvq_level must be in [0, {MRT2_RVQ_LEVELS}), got {rvq_level}")
  start = reserved_tokens + rvq_level * codebook_size
  return start, start + codebook_size


def unique_token_to_raw_code(
    unique_tokens: npt.ArrayLike,
    rvq_levels: Iterable[int] | None = None,
) -> np.ndarray:
  """Convert Depthformer unique-code tokens to raw 0-1023 RVQ code values."""
  tokens = np.asarray(unique_tokens, dtype=np.int64)
  levels = np.arange(tokens.shape[-1]) if rvq_levels is None else np.asarray(list(rvq_levels))
  if levels.shape[0] != tokens.shape[-1]:
    raise ValueError("rvq_levels length must match the token array's last axis")
  raw = tokens - MRT2_RESERVED_TOKENS - levels.reshape((1,) * (tokens.ndim - 1) + (-1,)) * MRT2_CODEBOOK_SIZE
  return raw.astype(np.int64)


def _large_negative(dtype: np.dtype) -> float:
  """Return the MLX/JAX-style large negative sentinel for masked logits."""
  if np.issubdtype(dtype, np.floating):
    return float(np.finfo(dtype).max * -0.7)
  return float(np.iinfo(dtype).max * -0.7)


def mask_logits_for_rvq_level(
    logits: npt.ArrayLike,
    rvq_level: int,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
) -> np.ndarray:
  """Apply valid-range and optional top-k/top-p filtering to full-vocab logits.

  Args:
    logits: Array shaped ``[..., 12294]``.
    rvq_level: RVQ level whose valid range should remain sampleable.
    top_k: Optional top-k threshold after valid-range masking.
    top_p: Optional nucleus threshold after valid-range masking.

  Returns:
    A float array with invalid positions replaced by a large negative sentinel.
  """
  if top_k is not None and top_p is not None:
    raise ValueError("Only one of top_k or top_p may be set")
  values = np.asarray(logits)
  if values.shape[-1] != MRT2_DEPTHFORMER_LOGIT_SIZE:
    raise ValueError(
        f"Expected logits last dimension {MRT2_DEPTHFORMER_LOGIT_SIZE}, got {values.shape[-1]}"
    )
  masked = values.astype(np.float32, copy=True)
  negative = _large_negative(masked.dtype)
  start, end = valid_range_for_rvq_level(rvq_level)
  valid = np.zeros(masked.shape[-1], dtype=bool)
  valid[start:end] = True
  masked[..., ~valid] = negative

  if top_k is not None:
    k = min(max(int(top_k), 1), end - start)
    threshold = np.partition(masked, -k, axis=-1)[..., -k][..., np.newaxis]
    masked = np.where(masked >= threshold, masked, negative)

  if top_p is not None:
    masked = _apply_top_p(masked, float(top_p), negative)
  return masked


def _apply_top_p(masked_logits: np.ndarray, top_p: float, negative: float) -> np.ndarray:
  """Apply nucleus filtering while preserving original vocabulary order."""
  sorted_indices = np.argsort(masked_logits, axis=-1)[..., ::-1]
  sorted_logits = np.take_along_axis(masked_logits, sorted_indices, axis=-1)
  shifted = sorted_logits - np.max(sorted_logits, axis=-1, keepdims=True)
  probs = np.exp(shifted)
  probs /= np.sum(probs, axis=-1, keepdims=True)
  cumulative = np.cumsum(probs, axis=-1)
  keep_sorted = cumulative - probs < top_p
  keep_sorted[..., 0] = True
  keep = np.zeros_like(keep_sorted, dtype=bool)
  np.put_along_axis(keep, sorted_indices, keep_sorted, axis=-1)
  return np.where(keep, masked_logits, negative)


def sample_logits_for_rvq_level(
    logits: npt.ArrayLike,
    rvq_level: int,
    config: SamplingConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
  """Sample one RVQ level from full-vocabulary logits on the CPU."""
  masked = mask_logits_for_rvq_level(
      logits,
      rvq_level,
      top_k=config.top_k,
      top_p=config.top_p,
  )
  if config.temperature == MRT2_DETERMINISTIC_TEMPERATURE:
    return np.argmax(masked, axis=-1).astype(np.int64)

  generator = rng or np.random.default_rng(config.seed)
  gumbel = generator.gumbel(size=masked.shape).astype(np.float32)
  return np.argmax(masked + gumbel * config.temperature, axis=-1).astype(np.int64)


def sample_rvq_frame_logits(
    logits_by_level: npt.ArrayLike,
    config: SamplingConfig,
) -> np.ndarray:
  """Sample a 12-level RVQ frame from level-major full-vocab logits.

  Args:
    logits_by_level: Array shaped ``[12, batch, time, 12294]``.
    config: Host sampling parameters.

  Returns:
    Unique-code tokens shaped ``[batch, time, 12]``.
  """
  logits = np.asarray(logits_by_level)
  expected_rank = 4
  if logits.ndim != expected_rank:
    raise ValueError(f"Expected rank {expected_rank} logits, got shape {logits.shape}")
  if logits.shape[0] != MRT2_RVQ_LEVELS:
    raise ValueError(f"Expected {MRT2_RVQ_LEVELS} RVQ levels, got {logits.shape[0]}")
  if logits.shape[-1] != MRT2_DEPTHFORMER_LOGIT_SIZE:
    raise ValueError(
        f"Expected logits last dimension {MRT2_DEPTHFORMER_LOGIT_SIZE}, got {logits.shape[-1]}"
    )

  rng = np.random.default_rng(config.seed)
  samples = [
      sample_logits_for_rvq_level(logits[level], level, config, rng)
      for level in range(MRT2_RVQ_LEVELS)
  ]
  return np.stack(samples, axis=-1).astype(np.int64)
