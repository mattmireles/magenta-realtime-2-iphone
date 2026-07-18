# MRT2 Depth-Body In-Graph Rollout Core ML Validation

- Model: build/models/mrt2_depth_body_rollout_int8-linear.mlpackage (FLOAT16, CPU_ONLY)
- Gate: FAIL — FLOAT16: token mismatch rate <= 0.02 per arm (fp16 near-tie flips sample an equally-likely top-k neighbor; distribution unchanged); temporal_feedback max |err| <= 0.05
- argmax_frames: 139 / 300 token mismatches (rate 0.4633), feedback max |err| 1.841667, p50 6.89 ms (CPU smoke)
- noise_frames_t10: 100 / 300 token mismatches (rate 0.3333), feedback max |err| 2.511558, p50 6.68 ms (CPU smoke)
- noise_frames_t13: 87 / 300 token mismatches (rate 0.2900), feedback max |err| 2.082430, p50 6.88 ms (CPU smoke)
