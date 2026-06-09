# SpectroStream Decoder Conv Core ML Baseline

- Frames: 25
- Boundary: host RVQ embeddings to pre-iSTFT SpectroStream decoder tensor
- Layout mode: `nchw-parallel`
- Input shape: `[1, 25, 256]`
- Output shape: `[1, 96, 480, 4]`
- Conv layers: `{'conv2d': 7, 'conv2d_transpose': 1}`
- Weight norm: standard MRT2 SpectroStream config has global_weight_norm=False; no fusion needed
- PyTorch vs MLX max error: 0.0032958984
- PyTorch vs MLX mean error: 0.0000067176
- PyTorch vs MLX SNR: 119.701 dB
- PyTorch vs MLX log-spectral distance: 0.001008 dB
- Core ML vs MLX max error: 0.0030517578
- Core ML vs MLX mean error: 0.0000076265
- Core ML vs MLX SNR: 118.850 dB
- Core ML vs MLX log-spectral distance: 0.000722 dB
- Core ML CPU_AND_GPU predict smoke p50/p99: 25.647 / 26.437 ms
- MLPackage: `SpectroStreamDecoder.mlpackage`

Known limits:
- Exports a fixed chunk decoder conv baseline, not per-frame streaming Core ML state.
- RVQ lookup remains host CPU owned.
- Inverse STFT and PCM overlap state remain host owned.
- CPU_AND_GPU predict timing on Mac is only a local smoke signal; iPhone profiling is Phase 5 authority.
