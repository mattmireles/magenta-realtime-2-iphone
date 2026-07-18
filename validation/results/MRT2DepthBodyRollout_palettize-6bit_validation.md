# MRT2 Depth-Body In-Graph Rollout Core ML Validation

- Model: build/models/mrt2_depth_body_rollout_palettize-6bit.mlpackage (FLOAT16, CPU_ONLY)
- Gate: FAIL — FLOAT16: token mismatch rate <= 0.02 per arm (fp16 near-tie flips sample an equally-likely top-k neighbor; distribution unchanged); temporal_feedback max |err| <= 0.05
- argmax_frames: 206 / 300 token mismatches (rate 0.6867), feedback max |err| 1.944876, p50 6.88 ms (CPU smoke)
- noise_frames_t10: 172 / 300 token mismatches (rate 0.5733), feedback max |err| 1.971295, p50 7.27 ms (CPU smoke)
- noise_frames_t13: 144 / 300 token mismatches (rate 0.4800), feedback max |err| 2.583538, p50 6.60 ms (CPU smoke)
