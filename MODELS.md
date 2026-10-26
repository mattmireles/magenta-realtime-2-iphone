# Model artifacts

Every Core ML model below is produced by an exporter in `exporters/` and covered
by a validation receipt in `validation/results/`. All Core ML models are
`mlprogram`, minimum deployment target **iOS 18 / macOS 15**, converted from the
[google/magenta-realtime-2](https://huggingface.co/google/magenta-realtime-2)
`mrt2_small` checkpoint (CC-BY-4.0). The conversion is content-preserving:
precision and memory-layout transforms only; no fine-tuning, pruning, or
distillation.

This repository documents **two generations** of artifacts, because the
supersession itself is a result reported in the paper (*Surgical Inference*,
§5.1–5.3):

- **Corrected generation** — the exporters and receipts that reproduce the
  paper's three headline findings. This is the current, correct code path.
- **Superseded generation** — the earlier artifacts, retained as
  **negative-result evidence**: in-graph state mutation fails ANE admission
  (§5.1), and 12 depth predictions per frame are weight-bandwidth-bound
  (§5.3). See "Superseded generation" below.

The corrected `.mlpackage` binaries are reproduced locally by the commands in
the [README](README.md#re-export-from-scratch) (checksums below); every
corrected package was regenerated from the exporters in this repo and its
weight payload matched the certified original byte-for-byte. The Hugging Face
mirror
([mattmireles/magenta-realtime-2-iphone](https://huggingface.co/mattmireles/magenta-realtime-2-iphone))
hosts all five corrected Core ML packages listed below, plus the RVQ codebook
binary, the test vector, fixtures, export metadata, and validation receipts. Its `superseded/`
directory holds the superseded artifacts' export metadata, and validation
receipts for the temporal graphs, not their packages.

---

## Corrected generation (paper §5.1–5.3)

### System-paper compression ladder (2026-07-18)

The real-time system uses the one-step
`mrt2_temporal_body_streaming_carry_01` specialization: one ordinary-input /
ordinary-output cache-update graph plus a host-owned 41-frame chronological
ring. Post-training weight compression was tested only after the 64-step
(23 post-wrap steps) state fixture was frozen.

| Component | Variant | Package vs. baseline | Deterministic result | Decision |
| --- | --- | ---: | --- | --- |
| Temporal | int8 linear | 0.502x | corr 0.997328; max error 4.585 | Reject |
| Temporal | 6-bit palette | 0.376x | corr 0.976950; max error 13.212 | Reject |
| Temporal | 4-bit palette | 0.252x | corr 0.887407; max error 23.949 | Reject |
| Depth | int8 linear | 0.505x | argmax token mismatch 0.463 | Reject |
| Depth | 6-bit palette | 0.380x | argmax token mismatch 0.687 | Reject |
| Depth | 4-bit palette | 0.256x | argmax token mismatch 0.940 | Reject |

All candidates remained finite but failed deterministic-reference parity.
Following the declared finite → parity → device → blind-audio gate order, none
was installed on a phone or rendered for listening. The selected system stays
uncompressed temporal plus the existing FP16 depth artifact. Consequently the
HF mirror is unchanged and this work makes no compression-causes-speedup
claim. Exact package bytes, weight hashes, all three depth arms, and early-stop
statuses are in
`validation/results/MRT2WeightCompressionLadder.{json,md}`.

| Artifact | File | Exporter | Precision | Compute target | Role |
| --- | --- | --- | --- | --- | --- |
| `MRT2TemporalBodyStreamingCarry` | `MRT2TemporalBodyStreamingCarry.mlpackage` | `convert_temporal_body_carry.py --frames 1 --streaming` | FLOAT16 | **ANE** (10/10 cold launches on both phones, ANE cost share 1.000) | Stateless one-frame temporal step, the shipped temporal graph. 48 K/V caches are ordinary **inputs**, 48 one-token cache updates are ordinary **outputs**, plus a fixed-shape `cache_valid_bias`; the host owns a 41-frame ring. No `ct.StateType`. The exporter's default 2-frame bucket (same weights) falls back to an all-CPU plan under `.cpuAndNeuralEngine` and is superseded. |
| `MRT2DepthBodyRollout` | `MRT2DepthBodyRollout.f16.mlpackage` (ship) / `.f32.mlpackage` (reference) | `convert_depth_body_rollout.py` | FLOAT16 (ship) / FLOAT32 (reference) | CPU in the system runtime (bandwidth-bound on every unit) | In-graph depth rollout: all 12 RVQ levels sampled in **one** prediction from host-supplied Gumbel noise. |
| `SpectroStreamDecoder` (NCHW FP16) | `SpectroStreamDecoder.f16.mlpackage` (ship) / `SpectroStreamDecoder.mlpackage` (FP32 reference) | `convert_spectrostream_decoder.py --nchw-parallel-layer 5 --fp16-rescale --compute-precision FLOAT16` | FLOAT16 (fp16-safe rescale) / FLOAT32 (reference) | FP16: **ANE** (`.cpuAndNeuralEngine`) on the A17 Pro. FP32: the A14 decoder; its compute plan under `.cpuAndNeuralEngine` is entirely CPU | RVQ embeddings → pre-iSTFT tensor. Channels-first internal layout, channels-last public I/O. The only FP16 variant usable on device (A17 Pro only). |

### Why each is the corrected artifact

**`MRT2TemporalBodyStreamingCarry` — state mutation, not attention, is the ANE cliff
(§5.1).** The complete 12-layer stateless stack for one frame (all attention, all FFN,
all 48 cache reads, all 48 one-token cache-update outputs; the host-cache update
stack, not the 2-frame checksummed package) compiles to a **single
ANE-resident graph** and matches or beats CPU-only and GPU placement — on iPhone 12
Pro, `MLComputePlan` reports `preferredCounts=ane:1033,cpu:2`,
`costWeights=ane:1.000`, p99 **14.991 ms**. The one condition is that no cache
mutation happens *inside* the graph. Every multi-frame or (on the A14) rolling-history
`ct.StateType` variant fails ANE compilation (the one-frame fixed-slot graphs compiled only as partial ANE islands, and the one the app used masked its history): the 25-frame stateful unrolled graph reproduces
`MILCompilerForANE … ANECCompile() FAILED`, Core ML **error −14**, on both
phones and under both `.cpuAndNeuralEngine` and `.all`. Core ML vs MLX temporal
correlation is 0.999975 (25-frame) / 0.999984 (2-frame carry). **Placement in
the app (paper §5.1):** for a time the temporal stage shipped on `.cpuAndGPU`
after earlier app builds failed to reproduce harness ANE placement. The system
runtime now ships this stateless boundary as the one-frame streaming-carry
graph (`--frames 1 --streaming`) with a Swift-owned 41-frame ring; it refuses to
start unless `MLComputePlan` puts it on the ANE, and it passes that gate in
10/10 cold launches on both the A17 Pro and the A14 (ANE cost share 1.000).

**`MRT2DepthBodyRollout` — weight bandwidth is the invariant that shapes the
graph (§5.3).** Twelve depth predictions per frame cost ~12 weight streams
(~40 ms/frame on A14) no matter how few positions they compute, because per-call
cost ≈ weight bytes ÷ DRAM bandwidth on every compute unit. Moving the entire
12-level autoregressive rollout inside **one** prediction removes 11 host/Core
ML boundaries, but it does not collapse the dependent transformer passes into
one weight traversal. The measured A14 FLOAT32 comparison is ~40.2 ms/frame for
12 separate full-pass predictions versus ~37 ms/frame for the one-call graph.
The large improvement comes from the one-call graph's FLOAT16 artifact:
**12.7 ms/frame on A14 and 8.4 ms on A17 Pro**. Determinism stays host-owned via
the Gumbel-max identity (the host supplies per-level Gumbel noise and inverse
temperature; top-k and valid-range masks are baked constants; embedder feedback
between levels is an in-graph gather). FLOAT32 export is token-for-token exact
(**0/900 mismatches**); FP16 flips fp16 near-tie tokens (~148/900) without
changing the sampling distribution, and was shipped after device quality and
paired-listening gates.

**`SpectroStreamDecoder` (NCHW FP16) — FP16 range, not layout, gates
correctness (§5.2).** The decoder's internal activations exceed the FP16
maximum on every real window. Without the rescale, FP16 is wrong in either
layout: in an October 2026 device matrix (both phones, iOS 27.0.1, SNR against
the FP32 decoder on the same device's CPU) channels-last and NCHW FP16 both
return non-finite output on CPU and GPU, and both run fully on the ANE and
return finite output at ~0 dB SNR (correlation ≤ 0.46). Layout decided neither
ANE placement nor correctness. The exact-in-FP32 mid-network rescale
(`apply_fp16_safe_rescale`, which this exporter applies only to the NCHW
layout) is what makes FP16 usable: on the A17 Pro it scores 41–46 dB on its
calibration window and 35–48 dB on a held-out window across engines (38.8 dB
on the ANE). It still fails where activations exceed FP16 range after the
rescale (1 of 600 windows by host scan; 8.5 dB on the A17 Pro ANE, non-finite on
CPU and GPU). On the A14 the same artifact is wrong on the ANE (−12 to
−21 dB) while its CPU gives 44.0 / 34.7 dB, so the A14 runs the FP32 decoder.
The June iPhone 12 Pro probe that found FP16 "finite only on the ANE"
(184,320/184,320 at 25-frame, p99 24.77 ms; 30,720/30,720 at 5-frame) measured
the unrescaled NCHW graph on a constant input; the 25-frame readout reproduces
byte-for-byte, and its output is wrong. FP32 NCHW parity vs MLX: SNR **118.85 dB**.

### Tensor contracts (corrected generation)

#### MRT2TemporalBodyStreamingCarry (published, F = 1) and carry exports (`frames` = F)

| Tensor | Shape | dtype |
| --- | --- | --- |
| `temporal_inputs` (in) | `[1, F, 1024]` | float32 |
| `source_encoded` (in) | `[1, F, 256]` | float32 |
| `cache_valid_bias` (in; `--streaming` only) | `[1, 1, 1, 43]` | float16 |
| `temporal_layer_XX_{self,cross}_{key,value}_cache_in` ×48 (in) | `[1, 41, 8, 128]` | float16 |
| `temporal_outputs` (out) | `[1, F, 1024]` | Core ML selected |
| `temporal_layer_XX_{self,cross}_{key,value}_cache_updates` ×48 (out) | `[1, F, 8, 128]` | float16 |

#### MRT2DepthBodyRollout

| Tensor | Shape | dtype |
| --- | --- | --- |
| `temporal_frame` (in) | `[1, 1, 1024]` | float32 |
| `gumbel_noise` (in) | `[12, 1024]` | float32 |
| `inverse_temperature` (in) | `[1]` | float32 |
| `sampled_codes` (out) | `[12]` | int32 |
| `temporal_feedback` (out) | `[1, 1024]` | Core ML selected |

`sampled_codes` are codebook-local codes (0–1023); the unique id is
`6 + level*1024 + code`. `temporal_feedback` is the mean of the 12 sampled token
embeddings (×32 scale baked in) — the next frame's `temporal_inputs` row.

#### SpectroStreamDecoder

| Tensor | Shape | dtype |
| --- | --- | --- |
| `decoder_embeddings` (in) | `[1, frames, 256]` | float32 |
| `decoder_stft` (out) | `[1, 96, 480, 4]` | float32 (channels-last public contract) |

Input is the host CPU RVQ lookup sum; output is consumed by host inverse
STFT / overlap-add. Audio-rate tensors never enter the Core ML graph.

### Validation receipts (corrected generation)

| Artifact | Receipt |
| --- | --- |
| `MRT2TemporalBodyStreamingCarry` (64 steps across the 41-frame ring wrap; Core ML vs reference corr 0.999312, fresh vs warmed state diverge) | `validation/results/MRT2TemporalBodyStreamingCarry_validation.{json,md}` |
| Superseded 2-frame carry (Core ML vs MLX carry parity, corr 0.999982, all cache updates finite) | `validation/results/MRT2TemporalBodyCarry_validation.{json,md}` |
| `MRT2DepthBodyRollout` FP32 (0/900 token parity) | `validation/results/MRT2DepthBodyRollout_f32_validation.{json,md}` |
| `MRT2DepthBodyRollout` FP16 (near-tie flips, distribution unchanged) | `validation/results/MRT2DepthBodyRollout_f16_validation.{json,md}` |
| `SpectroStreamDecoder` NCHW FP32 parity (SNR 118.85 dB, finite 1.0) | `validation/results/SpectroStreamDecoder_validation.{json,md}` |
| `SpectroStreamDecoder` NCHW FP16 (fp16-rescale; finite 184,320/184,320 on Mac, SNR 59.44 dB) | `validation/results/SpectroStreamDecoder_f16_validation.{json,md}` |

The temporal stateless-boundary ANE placement (`ane:1033,cpu:2`, p99 14.991 ms),
the stateful `ANECCompile −14` failures, and the decoder's on-device FP16
placement are on-device `MLComputePlan`/Instruments measurements, and its
accuracy (§5.2) is on-device SNR against the FP32 decoder; see [`docs/validation-receipts.md`](docs/validation-receipts.md)
for the device evidence and per-run log pointers. The October 2026 decoder
matrix's packages and per-cell logs are archived by the author and available on
request.

---

## Superseded generation (metadata under `superseded/` on Hugging Face)

The HF repo's `superseded/` directory holds these artifacts' export metadata,
and validation receipts for the temporal graphs; the packages themselves are not on
the mirror and are reproduced by the superseded exporters. They are retained as negative-result
evidence; do **not** treat them as the paper's shipped configuration.

| Artifact | Size | Precision | Compute target | Superseded by |
| --- | --- | --- | --- | --- |
| `superseded/MRT2TemporalBody.mlpackage` | 349 MB | FLOAT16 | Partial ANE island (`ane:0.69`, A14; stateful, 48 `ct.StateType`) | Stateless carry graph (§5.1): multi-frame and rolling-history state variants fail `ANECCompile −14`; this one-frame fixed-slot graph compiles only as a partial ANE island, and its baked bias masks all 41 history slots. |
| `MRT2TemporalBodyCarry.mlpackage` (2-frame bucket; metadata and receipt under `superseded/`) | 366 MB | FLOAT16 | CPU (all-CPU plan under `.cpuAndNeuralEngine`) | One-frame streaming carry (`MRT2TemporalBodyStreamingCarry`, same weights): the 2-frame graph prints `ANECCompile() FAILED` and falls back. |
| `superseded/MRT2DepthBody.mlpackage` | 93 MB | FLOAT32 | CPU/GPU (host sampling) | In-graph FP16 rollout (§5.3): 12 predictions/frame are weight-bandwidth-doomed. |

**Decoder note.** The earlier generation's `SpectroStreamDecoder.mlpackage`
turned out to be the NCHW FP32 build all along — its weight payload is
byte-identical to the certified NCHW FP32 reference, and its receipt is the
118.85 dB NCHW parity run. What §5.2 supersedes is the *shipping decision*
("FP32, keep it off the ANE, do not re-export at fp16") on the A17 Pro only,
not the binary, which remains the A14 decoder. It therefore stays at the
mirror's top level as the FP32 reference and A14 decoder, next to the FP16 A17
Pro variant `SpectroStreamDecoder.f16.mlpackage`.

### Host-side binaries

| Artifact | Size | Role |
| --- | --- | --- |
| `SpectroStreamRVQCodebooks.f32.bin` (+ `.json` shape sidecar) | 12.6 MB | 12 RVQ levels × 1024 codes × 256 dims, float32 little-endian. Loaded once; codebook gather runs on the CPU. |
| `examples/test_vector_smooth_electronic.bin` (+ `.json` provenance) | 1 KB | One certified conditioning vector (`[1, 256]`, prompt "smooth electronic", CFG tokens (20, 10, 2)). Deterministic: `exporters/export_conditioning.py` reproduces it byte-for-byte. |

### Checksums (sha256) — corrected binaries

| File | sha256 |
| --- | --- |
| `MRT2TemporalBodyStreamingCarry.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `02657e24a19402fed8210e431e1a67baa0c8196d4d23f0c78dd931668a9c5a64` |
| `MRT2DepthBodyRollout.f16.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `6c69ebae8cd7401fcb9927e3ff558d5b240cebdae2e29972d731ee08bd81c4f8` |
| `MRT2DepthBodyRollout.f32.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `af3ec4f9fadb84991b5dabbac58acd44294df7514226af2db63aa79659e304a9` |
| `SpectroStreamDecoder.f16.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `ceb3ed7d5b0286e69a937a17db253ce5a95e6bfa5560a187eb4c17de9e080f5d` |
| `SpectroStreamDecoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `38cbdf5ca97fe1744fa4d5a5caee23d50170a7eeb18daed1c40fe3a73aed9852` |
| `SpectroStreamRVQCodebooks.f32.bin` | `4e236269d4194ffe2d7463c483a1a36f4aff7d619c34f8e8bfe451c7af92d496` |
| `examples/test_vector_smooth_electronic.bin` | `08dfec990345e92f7dbffa7f3c349b9bebf61967c67c653c069550a5c6132039` |

### Checksums (sha256) — superseded binaries (reproduced locally; not hosted)

| File | sha256 |
| --- | --- |
| `superseded/MRT2TemporalBody.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `6a662d011b4b2438a78c152a4308cffb4a54904f8249379b2739fd8bf090ed43` |
| `superseded/MRT2TemporalBody.mlpackage/Data/com.apple.CoreML/model.mlmodel` | `04153ebbeac518cf684c9fd95f0cf30fe0861343a71879940cbc7111aa1ea2cc` |
| `superseded/MRT2DepthBody.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `ff1554bb1ddaf9d74c5449bc7687bfe5db1dfd90a476390ba2a0938c59fd3a86` |
| `superseded/MRT2DepthBody.mlpackage/Data/com.apple.CoreML/model.mlmodel` | `9d93fb43a8d278faf9950759a5afb6d5bcc5f083c72bbe978a654942063ee7c3` |

The superseded temporal export is `convert_temporal_body.py` (the stateful
unrolled graph) and the superseded depth export is `convert_depth_body.py`
(FP32 logits, host sampling). Both are kept in `exporters/` and headed as
superseded, with pointers to the corrected exporters and the paper findings.

## Provenance

Per-model export receipts (coremltools version, deployment target, conversion
timings, full I/O schemas) are emitted next to each `.mlpackage` as
`*_export_metadata.json` when you run an exporter.
