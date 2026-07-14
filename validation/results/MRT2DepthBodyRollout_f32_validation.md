# MRT2 Depth-Body In-Graph Rollout Core ML Validation

- Model: build/models/mrt2_depth_body_rollout_f32.mlpackage (FLOAT32, CPU_ONLY)
- Gate: PASS — FLOAT32: zero token mismatches vs reference on all arms; temporal_feedback max |err| <= 0.05
- argmax_frames: 0 / 300 token mismatches (rate 0.0000), feedback max |err| 0.000000, p50 16.21 ms (CPU smoke)
- noise_frames_t10: 0 / 300 token mismatches (rate 0.0000), feedback max |err| 0.000000, p50 16.15 ms (CPU smoke)
- noise_frames_t13: 0 / 300 token mismatches (rate 0.0000), feedback max |err| 0.000000, p50 15.48 ms (CPU smoke)
