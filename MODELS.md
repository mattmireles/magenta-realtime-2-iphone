# Model artifacts

All artifacts are hosted on Hugging Face:
[mattmireles/magenta-realtime-2-iphone](https://huggingface.co/mattmireles/magenta-realtime-2-iphone).
Every artifact below was produced by an exporter in `exporters/` and is covered
by a validation receipt in `validation/results/` (mirrored in the HF repo's
`validation/` folder). All Core ML models are `mlprogram`, minimum deployment
target **iOS 18 / macOS 15**.

## Core ML packages

| Artifact | Size | Precision | Compute target | Role |
| --- | --- | --- | --- | --- |
| `MRT2TemporalBody.mlpackage` | 349 MB | FLOAT16 | ANE (stateful) | Temporal transformer, one 40 ms frame per call. 48 `ct.StateType` KV buffers (fp16, `[1, 41, 8, 128]` per layer-head group). |
| `MRT2DepthBody.mlpackage` | 93 MB | FLOAT32 | CPU/GPU | Depth transformer → RVQ logits. Sampling (Gumbel + top-k) stays on the host. |
| `SpectroStreamDecoder.mlpackage` | 136 MB | FLOAT32 | GPU | RVQ embeddings → pre-iSTFT tensor. FLOAT16 overflows (15.7 % NaN/Inf) — do not re-export at fp16. |

## Host-side binaries

| Artifact | Size | Role |
| --- | --- | --- |
| `SpectroStreamRVQCodebooks.f32.bin` (+ `.json` shape sidecar) | 12.6 MB | 12 RVQ levels × 1024 codes × 256 dims, float32 little-endian. Loaded once; codebook gather runs on the CPU. |
| `examples/test_vector_smooth_electronic.bin` (+ `.json` provenance) | 1 KB | One certified conditioning vector (`[1, 256]`, prompt "smooth electronic", CFG tokens (20, 10, 2)). Deterministic: `exporters/export_conditioning.py` reproduces it byte-for-byte. |

## Tensor contracts

### MRT2TemporalBody

| Tensor | Shape | dtype |
| --- | --- | --- |
| `temporal_inputs` (in) | `[1, 1, 1024]` | float32 |
| `source_encoded` (in) | `[1, 1, 256]` | float32 |
| `temporal_outputs` (out) | `[1, 1, 1024]` | float32 |
| KV state ×48 | `[1, 41, 8, 128]` | float16 (Core ML state) |

### MRT2DepthBody

| Tensor | Shape | dtype |
| --- | --- | --- |
| `depth_inputs` (in) | `[1, 12, 1024]` | float32 |
| `depth_logits` (out) | `[1, 12, 12294]` | float32 |

### SpectroStreamDecoder

| Tensor | Shape | dtype |
| --- | --- | --- |
| RVQ embeddings (in) | `[1, 25, 256]` | float32 |
| pre-iSTFT tensor (out) | `[1, 96, 480, 4]` | float32 |

## Checksums (sha256)

| File | sha256 |
| --- | --- |
| `MRT2TemporalBody.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `6a662d011b4b2438a78c152a4308cffb4a54904f8249379b2739fd8bf090ed43` |
| `MRT2TemporalBody.mlpackage/Data/com.apple.CoreML/model.mlmodel` | `04153ebbeac518cf684c9fd95f0cf30fe0861343a71879940cbc7111aa1ea2cc` |
| `MRT2DepthBody.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `ff1554bb1ddaf9d74c5449bc7687bfe5db1dfd90a476390ba2a0938c59fd3a86` |
| `MRT2DepthBody.mlpackage/Data/com.apple.CoreML/model.mlmodel` | `9d93fb43a8d278faf9950759a5afb6d5bcc5f083c72bbe978a654942063ee7c3` |
| `SpectroStreamDecoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `38cbdf5ca97fe1744fa4d5a5caee23d50170a7eeb18daed1c40fe3a73aed9852` |
| `SpectroStreamDecoder.mlpackage/Data/com.apple.CoreML/model.mlmodel` | `e69658faae08a659a39c5d7afe25bc7bb6c6a5620b169b90d058ca38dcf93598` |
| `SpectroStreamRVQCodebooks.f32.bin` | `4e236269d4194ffe2d7463c483a1a36f4aff7d619c34f8e8bfe451c7af92d496` |
| `examples/test_vector_smooth_electronic.bin` | `08dfec990345e92f7dbffa7f3c349b9bebf61967c67c653c069550a5c6132039` |

## Provenance

Converted from the [google/magenta-realtime-2](https://huggingface.co/google/magenta-realtime-2)
`mrt2_small` checkpoint (CC-BY-4.0). The conversion is content-preserving:
precision and memory-layout transforms only; no fine-tuning, pruning, or
distillation. Per-model export receipts (coremltools version, deployment
target, conversion timings, full I/O schemas) ship in the HF repo's
`metadata/` folder.
