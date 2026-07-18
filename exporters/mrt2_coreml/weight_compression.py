"""Post-training Core ML weight-compression helpers for MRT2 exports."""

from __future__ import annotations

import os
import time
from typing import Any

import coremltools as ct


WEIGHT_COMPRESSION_CHOICES = (
  "none",
  "int8-linear",
  "palettize-6bit",
  "palettize-4bit",
)


def compress_weights(
  model: ct.models.MLModel,
  variant: str,
) -> tuple[ct.models.MLModel, dict[str, Any]]:
  """Compress constant weights and return the model plus receipt metadata."""
  if variant not in WEIGHT_COMPRESSION_CHOICES:
    raise ValueError(f"unsupported weight compression: {variant}")
  if variant == "none":
    return model, {
      "variant": variant,
      "algorithm": None,
      "optimization_seconds": 0.0,
    }

  optimize = ct.optimize.coreml
  if variant == "int8-linear":
    op_config = optimize.OpLinearQuantizerConfig(
      mode="linear_symmetric",
      dtype="int8",
      granularity="per_channel",
      weight_threshold=2_048,
    )
    transform = optimize.linear_quantize_weights
    algorithm = "linear_symmetric_int8_per_channel"
    parameters: dict[str, Any] = {
      "dtype": "int8",
      "granularity": "per_channel",
      "weight_threshold": 2_048,
    }
  else:
    nbits = 6 if variant == "palettize-6bit" else 4
    workers = max(1, min(8, os.cpu_count() or 1))
    op_config = optimize.OpPalettizerConfig(
      mode="kmeans",
      nbits=nbits,
      granularity="per_tensor",
      num_kmeans_workers=workers,
      weight_threshold=2_048,
    )
    transform = optimize.palettize_weights
    algorithm = f"kmeans_{nbits}bit_per_tensor"
    parameters = {
      "nbits": nbits,
      "granularity": "per_tensor",
      "num_kmeans_workers": workers,
      "weight_threshold": 2_048,
    }

  config = optimize.OptimizationConfig(global_config=op_config)
  started = time.perf_counter()
  compressed = transform(model, config=config)
  return compressed, {
    "variant": variant,
    "algorithm": algorithm,
    "parameters": parameters,
    "optimization_seconds": time.perf_counter() - started,
    "minimum_deployment_target": "iOS18",
  }
