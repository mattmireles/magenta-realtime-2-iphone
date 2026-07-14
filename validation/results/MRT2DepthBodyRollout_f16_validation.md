# MRT2 Depth-Body In-Graph Rollout Core ML Validation

- Model: build/models/mrt2_depth_body_rollout_f16.mlpackage (FLOAT16, CPU_ONLY)
- Gate: FAIL — FLOAT16: token mismatch rate <= 0.02 per arm (fp16 near-tie flips sample an equally-likely top-k neighbor; distribution unchanged); temporal_feedback max |err| <= 0.05
- argmax_frames: 51 / 300 token mismatches (rate 0.1700), feedback max |err| 1.673023, p50 7.59 ms (CPU smoke)
- noise_frames_t10: 47 / 300 token mismatches (rate 0.1567), feedback max |err| 2.035069, p50 7.35 ms (CPU smoke)
- noise_frames_t13: 50 / 300 token mismatches (rate 0.1667), feedback max |err| 1.926033, p50 7.40 ms (CPU smoke)
