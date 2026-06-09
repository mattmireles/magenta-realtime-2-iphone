# MRT2 Temporal Body Unrolled Core ML Validation

- Boundary: `temporal_body_unrolled_no_wrap`
- Frames: 1
- State count: 48
- Core ML vs MLX max error: 0.1178550720
- Core ML vs MLX mean error: 0.0197259826
- Core ML vs MLX correlation: 0.999985904188
- Core ML CPU_ONLY predict smoke: 75.110 ms

Per-frame Core ML vs MLX:
- Frame 0: max 0.1178550720, mean 0.0197259826, corr 0.999985904188

Temporal Core ML + depth-body FLOAT16:
- Core ML vs MLX max: 0.0652890205
- Core ML vs MLX mean: 0.0112627821
- Core ML vs MLX corr: 0.999998250871
- Deterministic sample mismatches: 0 / 12

Temporal Core ML + depth-body FLOAT32:
- Core ML vs MLX max: 0.0652890205
- Core ML vs MLX mean: 0.0112627821
- Core ML vs MLX corr: 0.999998250871
- Deterministic sample mismatches: 0 / 12

Known limits:
- Unrolls a fixed no-wrap frame count into one prediction.
- Frame 1 proves read-after-write across slots inside this unrolled graph.
- This is not the final one-prediction-per-40-ms-frame runtime API.
- The conditioning encoder remains host-owned through source_encoded.
- Depth-body logits remain in a separate Core ML package.
- This is not the 25-frame full temporal-plus-depth logits loop.
