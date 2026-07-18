# MRT2 Depth-Body In-Graph Rollout Core ML Validation

- Model: build/models/mrt2_depth_body_rollout_palettize-4bit.mlpackage (FLOAT16, CPU_ONLY)
- Gate: FAIL — FLOAT16: token mismatch rate <= 0.02 per arm (fp16 near-tie flips sample an equally-likely top-k neighbor; distribution unchanged); temporal_feedback max |err| <= 0.05
- argmax_frames: 282 / 300 token mismatches (rate 0.9400), feedback max |err| 2.739432, p50 6.43 ms (CPU smoke)
- noise_frames_t10: 275 / 300 token mismatches (rate 0.9167), feedback max |err| 2.554337, p50 6.91 ms (CPU smoke)
- noise_frames_t13: 267 / 300 token mismatches (rate 0.8900), feedback max |err| 3.238492, p50 6.83 ms (CPU smoke)
