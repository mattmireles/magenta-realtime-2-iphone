# MRT2 Temporal Body Carry Core ML Validation

- Boundary: `host_owned_kv_cache_inputs_and_update_outputs`
- Frames: 2
- Cache inputs: 48
- Cache update outputs: 48
- Core ML vs MLX max error: 0.3322525024
- Core ML vs MLX mean error: 0.0202849621
- Core ML vs MLX correlation: 0.999981748233
- Core ML CPU_ONLY predict smoke: 93.119 ms
- Cache updates all finite: True

Known limits:
- This first carry proof uses empty host-owned caches and history_length=0.
- It validates a no-wrap burst, not full rolling host cache placement.
- Depth-body logits remain in a separate Core ML package.
