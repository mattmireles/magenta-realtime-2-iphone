# MRT2 Small Graph Teardown

**Date:** 2026-06-05
**Status:** Reference (conversion-readiness map for the *superseded* first
generation; see the correction note below)

> **Correction (paper §6.3–6.5).** The "shipped" artifacts named in this
> teardown — a 1-frame stateful FP16 temporal body, a FLOAT32 depth body, and a
> FLOAT32 conv decoder — are the *superseded* first generation. The corrected
> generation replaces them: a **stateless** host-owned-cache temporal step
> (in-graph state mutation fails `ANECCompile −14`; §6.3), an **in-graph FP16
> depth rollout** (weight-bandwidth invariant; §6.5), and an **NCHW FP16**
> decoder that is finite and ANE-resident (§6.4). See `MODELS.md` and
> `docs/validation-receipts.md` §0. This teardown remains an accurate op-level
> map of the per-frame graph.

## Purpose

This document is the op-level conversion-readiness map for porting `mrt2_small`
(Google DeepMind's Magenta RealTime 2 small model) to Core ML and the Apple
Neural Engine. It records the actual per-frame inference graph, tensor shapes,
streaming state contract, Core ML/ANE risk ranking, and conversion-source
verdict that motivated the pipeline split shipped in this repo: a 1-frame
stateful FP16 temporal body (`MRT2TemporalBody.mlpackage`), a FLOAT32 depth
logits body (`MRT2DepthBody.mlpackage`), a FLOAT32 conv decoder
(`SpectroStreamDecoder.mlpackage`), and host-owned RVQ codebooks
(`SpectroStreamRVQCodebooks.f32.bin`). The exporters that implement these cuts
live in `exporters/`, with PyTorch wrappers in `exporters/mrt2_coreml/`.

The first rule for this document is simple: every conclusion must point to a
source file, command, asset, or explicit blocker. Do not infer shapes or state
behavior from architecture names alone.

## Source Freeze

### Toolchain

| Tool | Version |
| --- | --- |
| Python | 3.11 |
| macOS | 26.5, build 25F71 |
| Xcode | 26.5, build 17F42 |
| `magenta-rt` | 2.0.2 |
| `mlx` | 0.31.2 |
| `jax` | 0.10.1 |
| `flax` | 0.12.7 |
| `safetensors` | 0.7.0 |
| `numpy` | 2.3.5 |
| `scipy` | 1.17.1 |
| `huggingface_hub` | 1.17.0 |
| `soundfile` | 0.13.1 |

### Asset Commands

The required `mrt2_small` assets were fetched with the upstream `mrt` CLI:

```bash
mrt models init --source=hf
mrt models download mrt2_small --source=hf
mrt checkpoints download mrt2_small.safetensors --source=hf
```

Asset paths (relative to the Magenta asset root, `$MAGENTA_HOME`):

| Asset | Path | Size |
| --- | --- | --- |
| Exported MLX function | `$MAGENTA_HOME/models/mrt2_small/mrt2_small.mlxfn` | 435 MB |
| Exported MLX state | `$MAGENTA_HOME/models/mrt2_small/mrt2_small_state.safetensors` | 8.3 MB |
| Raw checkpoint | `$MAGENTA_HOME/checkpoints/mrt2_small.safetensors` | 1.1 GB |
| Baseline output | `$MAGENTA_HOME/outputs/output_audio_mlx_mrt2_small.wav` | 188 KB |

No required gated `mrt2_small` file was missing at the time of this freeze.

### Baseline Generation

Baseline command:

```bash
mrt mlx generate \
  --model=mrt2_small \
  --prompt='lofi house groove' \
  --duration=1 \
  --bits=8
```

Observed result (Apple Silicon Mac):

```text
Loaded mlxfn and 165 state arrays.
Warm-up 5 steps done in 1.0s.
Generated 25 frames in 0.7s, 36.9 steps/s, 27.1 ms/step.
Target: 25 steps/s, 40 ms/step.
Saved output_audio_mlx_mrt2_small.wav.
```

This proves only the local Mac MLX reference path for the frozen assets. It
does not prove Core ML conversion, ANE residency, iPhone p99 latency, or
thermal sustain.

## Investigation Log

### 2026-06-05

- **Hypothesis:** The upstream package and downloaded assets are sufficient for
  a teardown of `mrt2_small` without starting conversion work.
- **Tried:** Downloaded the `mrt2_small` MLX export, MLX state, raw checkpoint,
  MusicCoCa resources, and SpectroStream resources; ran a one-second MLX
  generation through the packaged CLI.
- **Outcome:** The local MLX path works and exposes 165 exported state arrays.
  Component clocks could then be mapped from source instead of guessed from the
  model card alone.

## Component and Clock Inventory

MRT2 has three named model components in the upstream model card (`MODEL.md`):
SpectroStream, MusicCoCa, and a decoder-only transformer LLM. For this port's
first phase, the conversion-critical path is narrower than the product
pipeline: one 25 Hz frame enters the exported Depthformer/SpectroStream step
and returns stereo PCM.

| Component | Source evidence | Params or shape facts | Duty cycle | Input tensors | Output tensors | Likely Core ML surface | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MusicCoCa text/audio embedding | `MODEL.md:84-88`; `magenta_rt/mlx/system.py:589-606`; MLX reference implementation | 768-d style embedding quantized to 12 RVQ tokens; the reference engine runs TFLite/CPU on prompt encode. | Prompt/control change, not every 40 ms frame. | Text or 16 kHz mono audio; later represented as 12 style tokens. | 12 MusicCoCa RVQ tokens. | Do not convert first. CPU/TFLite or existing prompt-token path is sufficient for the teardown. | Low for 25 Hz latency; medium product risk if prompt-change latency matters later. |
| Conditioning token assembly | `magenta_rt/mlx/system.py:608-672`; MLX reference implementation | Positive conditioning is `[1, 1, 141]`: 12 MusicCoCa tokens, 128 MIDI note tokens, 1 drum token. CFG scales and negative condition blocks are separate exported args. | Every frame in the reference engine; once per `generate(...)` call in the Python exported wrapper unless notes/control change. | Style tokens, notes, drum, CFG scales, temperature, top-k, empty forced tokens `[1, 0, 12]`. | Flat argument list before state append. | CPU-owned setup tensors feeding Core ML. Keep this out of the heavy graph unless a conversion wrapper needs constants. | Medium: token offsets, masks, and CFG negatives are easy to get wrong. |
| Depthformer / decoder-only transformer | `MODEL.md:89-100`; `magenta_rt/jax/model.py:468-476`; `magenta_rt/mlx/system.py:134-145` | `mrt2_small` has 12 temporal decoder layers, 41-frame local horizons, 12 generated RVQ codebooks, vocab 1024 plus reserved/dropout tokens. | Every 40 ms frame. | Conditioning, `previous_frame`, temporal/cross-attention state, sampling params, RNG/top-k path. | One generated frame of 12 RVQ tokens, then updated streaming state. | Primary Core ML stateful `mlprogram` candidate. Use fixed-shape `ct.StateType` if the real state ledger supports it. | High: streaming state, local attention masks, sampling, and token gather paths can trigger fallback or force CPU islands. |
| SpectroStream RVQ detokenization | `MODEL.md:80-83`; `magenta_rt/config.py:184-190`; `magenta_rt/mlx/system.py:138-145` | 64-level codec, truncated to 12 generated levels for MRT2; 1024-code vocab. | Every 40 ms frame after Depthformer samples 12 codes. | 12 raw/unique RVQ codes. | Feature embeddings for the decoder. | Default split is host/CPU lookup before decoder conv graph. Consider Core ML only if gather path is proven clean. | High: embedding/gather/indexing is a likely ANE fallback. |
| SpectroStream waveform decoder | `magenta_rt/mlx/system.py:144-149`; MLX reference implementation | Produces one stereo frame `[1, 2, 1920]`, i.e. 40 ms at 48 kHz. | Every 40 ms frame. | RVQ embeddings/features plus decoder streaming state/lookback. | Stereo PCM frame. | Separate decoder conversion project. Baseline should be GPU or split conv-only Core ML, with iSTFT/overlap state owned deliberately. | High: conv transpose, iSTFT, overlap-add, layout conversion, and audio-quality drift. |
| Prefill / context seeding | MLX reference implementation (prefill/context path) | Feeds historical 12-code frames and seeds `previous_frame`; `tokens_out` exposes generated raw codes. | Context-only; not required for the normal next-frame hot loop. | Token arrays or encoded audio context. | Updated transformer state and optional PCM during prefill. | Not a first conversion target. Use as behavioral evidence for state ownership and parity tests. | Medium for live looping later; low for the first conversion proof. |
| Export/offline model construction | `magenta_rt/mlx/system.py:490-550`; `magenta_rt/mlx/export.py` | `.mlxfn` and `_state.safetensors` bake model construction and initial state. | Load-time/offline. | Exported function path plus 165 state arrays. | Callable exported function and initial state list. | Use as evidence, not as the recommended conversion source by itself. | Medium: reverse-engineering `.mlxfn` is a tempting but brittle route. |

### Clock Cut

The hot loop is the reference engine's per-frame generation path:

```text
cached MusicCoCa tokens + live MIDI/drum/control
  -> condition args and negatives
  -> exported function + state
  -> audio [1, 2, 1920] + new state
  -> optional tokens_out from previous_frame state
```

Source evidence:

- The MLX reference implementation describes the runtime as MusicCoCa
  TFLite/CPU followed by a transformer step returning `[1, 2, 1920]` at 25 Hz,
  with `generate_frame` defined as one 1920-sample stereo frame plus optional
  12-code `tokens_out`.
- The reference per-frame step updates conditioning, CFG scales, and state,
  calls the compiled function, copies `[1, 2, 1920]` into left/right PCM
  buffers, and replaces the transformer state from outputs `1...N`.
- `magenta_rt/mlx/system.py:704-732` tokenizes style before the frame loop,
  builds args once, then repeatedly calls the exported function and appends
  `outputs[0]` shaped `(1, 2, 1920)`.

MusicCoCa conversion is cut from the first phase. The hot-loop dependency is
the 12-token style prefix already in conditioning memory, not the text/audio
encoder itself. The first Core ML proof should spend its risk budget on the
stateful Depthformer and the SpectroStream decode boundary.

## Per-Frame Dataflow and Tensor Shape Ledger

This ledger uses the exported `.mlxfn` runtime contract because that is the
path the reference engine actually drives. The in-process Python sampler is
still useful for source names, but it includes CFG tokens inside the
conditioning block; the exported path moves CFG scales and negative condition
blocks to separate arguments.

### One-Frame Diagram

```text
Controller / CPU
  style_tokens: int32[12]
  notes:        int32[128]
  drum:         int32[1]
  temperature: float32[1]
  top_k:       int32[1]
  cfg scales:  float32[1] x 3
  negatives:   int32[1, 1, 141] x 2
  forced:      int32[1, 0, 12]
        |
        v
condition args
  cond:         int32[1, 1, 141]
  neg_mc:       int32[1, 1, 141]
  neg_notes:    int32[1, 1, 141]
  state:        165 exported state arrays
        |
        v
Depthformer temporal step
  previous_frame -> embed -> mean
  int64/int32[1, 1, 12] -> float[1, 1, 12, D] -> float[1, 1, 1024]
  temporal state -> updated temporal state
        |
        v
Depthformer depth sampling, 12 serial RVQ levels
  depth input/logits -> valid range per RVQ codebook -> sample
  float[1, 1, 1024] / float[1, 1, 768] -> logits[1, 1, 12295] -> int[1, 1, 12]
        |
        v
SpectroStream decode boundary
  unique RVQ codes -> raw codes -> embeddings -> decoder -> inverse STFT
  int[1, 1, 12] -> int[1, 1, 12] -> float[1, 1, 256] -> PCM[1, 2, 1920]
        |
        v
iOS audio handoff contract
  48 kHz stereo PCM, 1920 frames/channel per 40 ms generated chunk
```

`D` is the codebook embedding dimension used by the Depthformer target
embedder. The temporal body uses 1024 model dimensions; the depth body uses a
1024-to-768 adapter before its two-layer depth transformer.

### Exported Runtime Inputs

| Tensor | Shape | Dtype | Owner | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| `cond` | `[1, 1, 141]` | `int32` | CPU/controller | `magenta_rt/mlx/system.py:639-645`; MLX reference implementation | 12 MusicCoCa tokens + 128 notes + 1 drum. Offset is `NUM_RESERVED_TOKENS + 1` in the Python export wrapper, with equivalent reserved-token / mask-id handling in the reference engine. |
| `temperature` | `[1]` | float | CPU/controller | `magenta_rt/mlx/system.py:662-665` | Sampling scalar. Should stay CPU-owned unless sampling remains inside Core ML. |
| `top_k` | `[1]` | `int32` | CPU/controller | `magenta_rt/mlx/system.py:662-665` | Dynamic sampling control. |
| `cfg_musiccoca`, `cfg_notes`, `cfg_drums` | `[1]` each | float | CPU/controller | `magenta_rt/mlx/system.py:666-668` | Exported as scalars, not as conditioning tokens. |
| `neg_musiccoca` | `[1, 1, 141]` | `int32` | CPU/controller | `magenta_rt/mlx/system.py:646-652` | Same condition block with style masked. |
| `neg_notes` | `[1, 1, 141]` | `int32` | CPU/controller | `magenta_rt/mlx/system.py:654-660` | Same condition block with notes masked. |
| `forced_tokens` | `[1, 0, 12]` normal path | `int32` | CPU/controller | `magenta_rt/mlx/system.py:671` | Non-empty forced-token path is for prefill/context seeding, not normal generation. |
| `state_0...state_164` | See state ledger below | mixed | Model/runtime | `magenta_rt/mlx/system.py:542-550` | Real state shapes come from `_state.safetensors`; classified below. |

### Depthformer Step

| Stage | Shape contract | Source evidence | Conversion note |
| --- | --- | --- | --- |
| Initial decoder state | `(rng, sos_frame, temporal_state, step)` | `magenta_rt/mlx/depthformer.py:535-571`; `magenta_rt/jax/depthformer.py:526-562` | `sos_frame` has shape `[B, 1, num_codebooks]`; exported state later fixes concrete shapes. |
| Previous-frame embedding | `previous_frame [B, 1, 12] -> embedded [B, 1, 12, D] -> mean [B, 1, D]` | `magenta_rt/mlx/depthformer.py:630-639` | Token embedding/gather is an ANE risk if kept in-graph. |
| Temporal body | Input/output sequence `[B, 1, 1024]` for `mrt2_small` | `magenta_rt/config.py:64-71`; `magenta_rt/jax/model.py:468-476`; `magenta_rt/mlx/depthformer.py:640-645` | Primary stateful transformer candidate. |
| Depth body init | Fresh depth state per generated frame | `magenta_rt/mlx/depthformer.py:647-655` | This is not the long-lived KV cache; temporal state is. |
| CFG expansion | Batch is interleaved with CFG negatives, then packed back to ordinary output | `magenta_rt/mlx/depthformer.py:1017-1056`; `magenta_rt/mlx/depthformer.py:1141-1148` | Core ML proof should avoid hiding this as dynamic control. |
| 12-level sampling loop | 12 serial calls into the depth body, one valid range per RVQ codebook | `magenta_rt/mlx/depthformer.py:691-724`; `magenta_rt/jax/depthformer.py:727-759` | The loop contains top-k/top-p/random sampling and valid-range masking. CPU-owned sampling is the simpler first proof if logits export is viable. |
| Depth samples | `int[1, 1, 12]` unique-code frame | `magenta_rt/mlx/depthformer.py:736-740` | This becomes `previous_frame` state and the SpectroStream input after unique-code cleanup. |

Small model architecture constants:

| Submodel | Layers | Model dim | Hidden dim | Heads | Head dim | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Temporal decoder | 12 | 1024 | 4096 | 8 | 128 | `magenta_rt/config.py:64-71`; `magenta_rt/jax/model.py:468-476` |
| Depth decoder | 2 | 768 | 3072 | 6 | 128 | `magenta_rt/config.py:55-62`; `magenta_rt/jax/model.py:468-476` |

### SpectroStream Decode

| Stage | Shape / constant | Evidence | Conversion note |
| --- | --- | --- | --- |
| RVQ code cleanup | 12 generated unique-code levels become raw 0-1023 codebook values. | `magenta_rt/mlx/system.py:134-145`; MLX reference implementation | The reference `tokens_out` path subtracts `k * 1024 + reserved_offset` from `previous_frame`. |
| RVQ detokenization | Codes `[B, T, 12] -> embeddings [B, T, 256]` | `magenta_rt/mlx/spectrostream/modeling.py:901-945`; `magenta_rt/mlx/spectrostream/modeling.py:1135-1160` | Default layer uses `mx.take` gather. The JAX direct decode helper can use one-hot/einsum, but the exported path should be assumed gather-risk until proven otherwise. |
| Decoder stack | Embeddings -> 2D conv / transposed-conv decoder features | `magenta_rt/mlx/spectrostream/modeling.py:1051-1069`; `magenta_rt/mlx/spectrostream/modeling.py:1097-1100` | Separate conversion project from Depthformer; conv-only Core ML is plausible after RVQ lookup is isolated. |
| iSTFT | Decoder features -> waveform via causal inverse STFT | `magenta_rt/mlx/spectrostream/modeling.py:1014-1025`; `magenta_rt/mlx/spectrostream/modeling.py:1097-1100` | Host/Accelerate iSTFT is the simpler baseline if Core ML conversion introduces bitcast/complex/overlap issues. |
| Frame ratio | `prod(time ratios) * stft_frame_step = 4 * 480 = 1920` waveform samples per code frame | `magenta_rt/mlx/spectrostream/modeling.py:1071-1076`; `magenta_rt/mlx/spectrostream/modeling.py:1135-1142` | Matches the 25 Hz product clock. |
| Audio output | `[1, 2, 1920]` non-interleaved frame | MLX reference implementation | An iOS audio layer must not assume the render callback asks for exactly 1920 frames. It needs a ring/lookahead bridge. |

### PCM Handoff Boundary

The teardown stops at generated PCM. The downstream iOS audio layer owns:

- 48 kHz stereo PCM.
- 1920 frames per channel per generated 40 ms chunk.
- A pull-side audio callback that may request smaller or larger render quanta
  than 1920 frames.
- A lock-free bridge and lookahead sized from measured p99 inference, not mean
  inference time.

## Streaming State and KV-Cache Contract

The real exported `mrt2_small` state file has 165 arrays. The exhaustive
per-key table was generated by a small probe script that walks
`mrt2_small_state.safetensors` and prints every key, dtype, and shape; the
summary table below condenses its output.

The safetensors keys are flatten-order artifacts. `magenta_rt/mlx/export.py:31-68`
flattens a nested state pytree, `magenta_rt/mlx/export.py:437-466` reconstructs
that state for each exported call, and `magenta_rt/mlx/export.py:529-530` saves
the leaves as `state_0...state_N`. Do not treat those key names as semantic
layer names without the flatten structure from the same export run.

The JAX implementation mirrors the same conceptual decoder state in
`magenta_rt/jax/depthformer.py:526-740`; the MLX implementation is the runtime
source here because the exported assets and the reference engine use it.

### Real State Shape Summary

| Count | Dtype | Shape | Inferred owner | Core ML representation | Risk |
| ---: | --- | --- | --- | --- | --- |
| 48 | BF16 | `[1, 41, 8, 128]` | Temporal transformer bounded attention windows: 12 layers x self/cross attention x K/V. | Primary `ct.StateType` candidate after casting/storage decision to FP16-compatible Core ML tensors. | High. This is the make-or-break state for ANE residency. Exact 41-frame horizon may need padding to 64 only if profiling shows alignment cost. |
| 24 | BOOL | `[1, 41]` | Temporal cache masks paired with 41-frame windows. | Prefer delete/derive from fixed step if possible. Otherwise explicit input/output carry or state only if converter supports the bool path cleanly. | Medium. Bool masks can create control/fallback noise. |
| 24 | I32 | `[1]` | Temporal cache counters/positions paired with attention windows. | Prefer CPU-owned scalar or explicit input/output. | Medium. Scalar state should not force the heavy matmul graph off ANE. |
| 2 | previous-frame candidates | `[1, 1, 12]` | Outer `sampler_previous_output` and inner decoder `previous_frame`. | For the first proof, make previous frame an explicit input/output if CPU owns sampling. Use state only if exporting the current full sampler. | Medium. The reference engine already has to identify the correct slot dynamically. |
| 1 | U32 | `[1, 2]` | RNG key. | CPU-owned. Do not put random sampling state into the first Core ML graph. | High if kept in graph; low if deleted from graph. |
| 1 | U32 | `[1, 1, 12]` | Inner previous-frame candidate in current asset; the reference engine chooses the last rank-3/rank-4 match. | Explicit input/output for CPU sampling path. | Medium. Dtype mismatch with token tensors needs normalization in conversion code. |
| 28 | BOOL | `[1, 2]` | SpectroStream decoder carry masks. | Separate decoder model state or host-owned carry, not Depthformer state. | Medium. Coupling this to the transformer model creates needless ping-pong. |
| 2 | BOOL | `[1, 6]` | SpectroStream decoder carry masks for `[1, 6, 480, 64]` buffers. | Same as decoder state. | Medium. |
| 29 | F32 decoder carry | Multiple `[1, 2, *, *]`, `[1, 6, 480, 64]`, and `[1, 480, 2]` shapes | SpectroStream conv/iSTFT lookback and overlap carry. | Separate decoder carry. If iSTFT stays host-side, only conv lookback belongs near Core ML. | High for audio quality; frame-boundary clicks are more important than elementwise drift here. |
| 3 | I32 | `[1]` | Export wrapper / delay / decoder scalar state outside the 24 temporal-cache counters. | CPU-owned or explicit carry after semantic identification. | Low individually; medium if blindly put into stateful Core ML. |

### Previous-Frame Identification

The MLX reference implementation searches all state shapes and selects the
last rank-3 `[1, 1, 12]` or rank-4 CFG-shaped previous-frame candidate. In the
real `mrt2_small_state.safetensors` inventory:

- `state_0` is `I64 [1, 1, 12]`.
- `state_3` is `U32 [1, 1, 12]`.
- No rank-4 CFG previous-frame variant appears in this asset.

The reference engine's choice therefore resolves to `state_3`. This matches
the source comment that the outer sampler slot is unused for `tokens_out`,
while the inner decoder `previous_frame` carries the just-sampled frame.

### Core ML State Verdict

Current Core ML Tools docs (checked for `coremltools` on 2026-06-05) show
stateful conversion through:

```python
ct.convert(
    traced_model,
    inputs=[...],
    outputs=[...],
    states=[
        ct.StateType(
            wrapped_type=ct.TensorType(shape=fixed_shape, dtype=np.float16),
            name="buffer_name",
        ),
    ],
    minimum_deployment_target=ct.target.iOS18,
)
```

The `StateType` name must match the registered PyTorch buffer name, and the
wrapped `TensorType` must not provide its own name or default value. The docs
show flexible `RangeDim` for normal inputs, but not as the right answer for hot
state. This port's stateful-state guidance is stricter (see
[docs/stateful-kv-coreml.md](stateful-kv-coreml.md)): keep hot-path state
fixed-shape, avoid `RangeDim`, and treat `EnumeratedShapes` as a later
bucketing tool only after the default shape's residency is proven.

Practical contract for the conversion proof:

- Make a minimal PyTorch/traceable wrapper with semantic buffer names, not
  `state_5`, `state_6`, etc.
- Use `ct.StateType` first for the 48 temporal K/V cache tensors only.
- Keep RNG, top-k/top-p, valid-range masking, and token sampling on CPU for the
  first proof. The graph should return logits or pre-sampling values if that is
  the shortest path to removing random/dynamic ops.
- Carry `previous_frame` explicitly when CPU owns sampling. If the full current
  sampler is exported as one graph, `previous_frame` becomes state but the graph
  inherits the random/top-k risk.
- Keep SpectroStream state in a separate decoder proof. Conv lookback can be
  Core ML state; iSTFT overlap is simpler host-owned state until proven
  otherwise.

## Op Inventory and ANE Risk Ranking

Core ML residency must be proven, not inferred. `MLComputeUnits.all` is only a
scheduling request. The first conversion proof should compare `.cpuOnly`,
`.cpuAndGPU`, `.cpuAndNeuralEngine`, and `.all` where a subgraph is plausible,
then use Xcode's Core ML performance report, Instruments Core ML traces, and
`powermetrics` to confirm where work actually ran.

### Temporal Transformer Ops

| Op group | Source evidence | ANE posture | Notes |
| --- | --- | --- | --- |
| Multi-channel token embedding | `magenta_rt/mlx/transformer.py:65-120`; `magenta_rt/mlx/depthformer.py:630-639` | Risky if implemented as gather; acceptable if rewritten as stable embedding/matmul path or moved to CPU input prep. | Previous-frame and sampled-token embedding happen every frame. |
| Q/K/V projections | `magenta_rt/mlx/transformer.py:430-447`; `magenta_rt/mlx/transformer.py:645-655` | Likely ANE-clean as dense/einsum once shapes are fixed. | Favor static `[B, T, D]` and fixed K/V state. |
| Local/windowed self attention | `magenta_rt/mlx/transformer.py:455-479`; state inventory above | Plausible but high-risk because the 41-frame cache/mask update must remain static and resident. | 48 BF16 `[1, 41, 8, 128]` state arrays are the critical proof. |
| Streaming cross attention | `magenta_rt/mlx/transformer.py:657-685`; `magenta_rt/mlx/transformer.py:920-953` | Plausible if source state is fixed and masks are not dynamic-control poison. | Treat as a separate attention-state class in the wrapper, not generic `state_i`. |
| RMSNorm / reductions | `magenta_rt/mlx/transformer.py:241`; `magenta_rt/mlx/transformer.py:779-785` | Usually acceptable, but precision/reduction placement must be checked in Core ML report. | Avoid silent FP32 CPU fallback for reductions. |
| FFN dense + GELU approximation | `magenta_rt/mlx/transformer.py:753-811`; `magenta_rt/mlx/transformer.py:987-990` | Likely ANE-clean. | `mrt2_small` uses ungated FFN from config, so keep it simple. |
| Output projection / logits | `magenta_rt/jax/model.py:450-457`; `magenta_rt/mlx/depthformer.py:691-705` | Dense/logits math likely clean; downstream sampling is not. | Consider exporting logits and sampling on CPU. |

### Depth Sampling Ops

| Op | Source evidence | ANE posture | Decision |
| --- | --- | --- | --- |
| 12 serial depth-body calls | `magenta_rt/mlx/depthformer.py:691-724` | Model math is plausible; loop control is not the right first graph shape. | Trace a fixed 12-step path only if logits-export path fails. |
| Logit soft cap `tanh` | `magenta_rt/mlx/depthformer.py:698-702` | Likely okay. | Keep in math graph if logits remain in graph. |
| CFG arithmetic | `magenta_rt/mlx/depthformer.py:205-257`; `magenta_rt/mlx/depthformer.py:1017-1056` | Medium risk. | Prefer explicit fixed branches or CPU pre/post logic over dynamic batch shuffling. |
| Valid-range masking | `magenta_rt/mlx/depthformer.py:282-292` | Medium/high risk. | CPU sampling path deletes this from Core ML. |
| Top-k/top-p sorting | `magenta_rt/mlx/depthformer.py:293-325` | High fallback risk. | Keep CPU-owned. |
| Gumbel random + RNG update | `magenta_rt/mlx/depthformer.py:276-280`; `magenta_rt/mlx/depthformer.py:721` | High fallback risk and bad state fit. | Delete from graph for first proof. |
| Argmax sample | `magenta_rt/mlx/depthformer.py:326-329` | Possible but not worth coupling to random/top-k. | CPU-owned sampling. |

### SpectroStream Decode Ops

| Op group | Source evidence | Baseline route | Audio-quality risk |
| --- | --- | --- | --- |
| RVQ detokenization | `magenta_rt/mlx/spectrostream/modeling.py:901-945` | CPU/host lookup first. | Wrong unique-code cleanup or FP16 embedding accumulation changes timbre. |
| Conv / transposed-conv synthesis | `magenta_rt/mlx/spectrostream/modeling.py:1051-1069`; `magenta_rt/mlx/spectrostream/modeling.py:1097-1113` | `.cpuAndGPU` baseline; `.cpuAndNeuralEngine` conv-only experiment second. | Missing conv lookback creates frame-boundary clicks. |
| ELU/residual blocks | `magenta_rt/mlx/spectrostream/modeling.py:1135-1158` | Likely GPU/ANE-clean inside conv graph. | FP16 quiet-tail loss and clipping need listening plus metrics. |
| iSTFT / overlap-add | `magenta_rt/mlx/spectrostream/modeling.py:1014-1025`; `magenta_rt/mlx/spectrostream/modeling.py:1097-1100` | CPU/Accelerate first. | Overlap state bugs are click generators; do not hide them inside a black-box graph first. |
| Layout / bitcast / reshape | `magenta_rt/mlx/export.py:464-465`; MLX reference implementation | Host-side explicit layout bridge. | Channel swap or interleave mistakes are easy and audible. |

First decoder quality proof must compare one-shot decode against frame-by-frame
streaming decode for the same token sequence. Minimum metric set: SNR plus
log-spectral distance, with a click detector or manual listening pass if LSD is
not yet wired.

### Ranked Fallback Risks

| Rank | Risk | Source | Likely fallback unit | Mitigation | First proof |
| ---: | --- | --- | --- | --- | --- |
| 1 | Sampling inside Core ML: random Gumbel, top-k/top-p sort, valid masking, argmax. | `magenta_rt/mlx/depthformer.py:259-329`; `magenta_rt/mlx/depthformer.py:691-724` | CPU fallback or unsupported graph control. | Export logits/pre-sampling math and sample on CPU for first proof. | Logits-only converters (shipped as `exporters/convert_temporal_body.py` and `exporters/convert_depth_body.py`) plus Python parity: logits from Core ML vs MLX before sampling. |
| 2 | 48 temporal K/V states `[1, 41, 8, 128]` fail to stay resident or update cheaply. | State inventory above; `magenta_rt/mlx/transformer.py:455-479`; `magenta_rt/mlx/transformer.py:657-685` | CPU/GPU fallback or ANE memory stall. | Fixed-shape `ct.StateType`, semantic buffer names, FP16, profile exact state layout before padding. | Xcode Core ML report + Instruments trace for a one-frame Depthformer state update. |
| 3 | RVQ embedding/gather in SpectroStream or token embedding forces CPU. | `magenta_rt/mlx/spectrostream/modeling.py:925-943`; `magenta_rt/mlx/transformer.py:65-120` | CPU fallback. | Move RVQ lookup to CPU or rewrite as one-hot/einsum only if measured faster and resident. | Compare CPU lookup + decoder graph against one-hot/einsum Core ML candidate. |
| 4 | SpectroStream iSTFT/overlap state creates graph ping-pong or clicks. | `magenta_rt/mlx/spectrostream/modeling.py:1014-1025`; `magenta_rt/mlx/spectrostream/modeling.py:1097-1113` | CPU/GPU ping-pong; audio artifact risk. | Keep iSTFT host-side first; carry overlap explicitly. | One-shot vs frame-by-frame decode SNR/LSD/click check. |
| 5 | CFG batch interleaving and negative branches make a dynamic graph. | `magenta_rt/mlx/depthformer.py:1017-1056`; `magenta_rt/mlx/export.py:428-459` | CPU fallback or shape explosion. | Trace fixed CFG arity, or make CFG branch arithmetic CPU/precomputed for first proof. | Convert a fixed-arity logits subgraph, then compare no-CFG and CFG outputs against MLX. |
| 6 | Bool masks and I32 counters in state pollute the ANE graph. | State inventory above; `magenta_rt/mlx/transformer.py:397-402`; `magenta_rt/mlx/transformer.py:599-607` | CPU control islands. | Derive masks from fixed step or keep scalar/mask carry CPU-owned if possible. | Profile with and without explicit mask state in the minimal wrapper. |
| 7 | FP16 codec drift causes quiet-tail loss, clipping, or timbre change. | `magenta_rt/mlx/spectrostream/modeling.py:990-991`; [docs/rvq-decoder.md](rvq-decoder.md) | Not necessarily fallback; product-quality failure. | Validate audio perceptually and with SNR/LSD before chasing speed. | Frame decode parity script with WAV fixtures. |

Default compute-unit policy for the conversion proof:

- Depthformer math: start with `.cpuAndNeuralEngine`, compare `.all`, and use
  `.cpuOnly` only for parity/debug.
- SpectroStream decoder: start with `.cpuAndGPU` because GPU conv/iSTFT is the
  pragmatic baseline; test `.cpuAndNeuralEngine` only for conv synthesis after
  RVQ lookup and iSTFT are deliberate CPU islands.
- Treat any `.all` win as a hypothesis until Instruments or Core ML reports show
  the ANE/GPU/CPU split.

## Conversion Source Verdict

Recommended route: use the MLX reference path as behavioral truth, then build a
minimal PyTorch conversion wrapper for the Depthformer logits/state subgraph.
Do not reverse-engineer `.mlxfn` as the conversion source. Do not start by
converting all of MRT2 as one graph.

One-paragraph executive verdict: convert two subgraphs, not one product blob.
First, convert a Depthformer logits/state `mlprogram` from a PyTorch wrapper
with semantic buffers and fixed `ct.StateType` temporal KV state; keep
sampling, RNG, top-k/top-p, valid-range masking, CFG control, and
`previous_frame` update CPU-owned until the logits proof is resident and fast.
Second, run a separate SpectroStream decoder proof where CPU owns RVQ
detokenization and iSTFT at baseline, `.cpuAndGPU` handles the pragmatic conv
decoder, and `.cpuAndNeuralEngine` is only a conv-synthesis optimization
experiment. Top risks are sampling-in-graph, 41-frame K/V state residency, and
codec gather/iSTFT ping-pong or audio artifacts.

### Route Comparison

| Route | Readability | Weight loading | Shape control | Statefulness | Core ML converter support | Parity risk | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| JAX/Flax to Core ML through an intermediate | Medium: source is clean but SequenceLayers abstractions hide lowering details. | Medium: native checkpoint lineage exists. | Medium: can express fixed shapes, but export path is not Core ML-native. | Medium: conceptual state is clear, converter path is indirect. | Weak as a direct route. | Medium/high because intermediate rewrites can alter sampling/state. | Reject as first route; keep as reference source. |
| MLX `.mlxfn` reverse engineering | Low: flat state and compiled function obscure semantics. | High for existing assets, low for editable graph. | Low: exported signatures exist but not a source graph for Core ML. | Low: `state_0...state_164` are flatten-order artifacts. | Not a Core ML source. | High. | Reject. Use only as behavior/evidence and parity target. |
| MLX reference engine as source | High for runtime behavior and audio contract. | Low: it consumes `.mlxfn`, not train/export weights. | High for boundary tensors. | Medium: identifies previous-frame/RNG slots dynamically. | Not directly convertible. | Low as a behavioral spec, high as converter source. | Use as runtime spec and parity harness, not as model export source. |
| Minimal PyTorch wrapper/port from safetensors | Medium initially; best once scoped to logits/state subgraph. | Medium: requires explicit mapping from MRT2 weights. | High: wrapper can expose fixed tensors and semantic buffers. | High: `register_buffer` names can map to `ct.StateType`. | Strongest current route for `torch.jit.trace` / `torch.export` to `mlprogram`. | Medium; must validate logits and listening output. | Choose this route. This is the `exporters/mrt2_coreml/` wrapper package in this repo. |

### Subgraph Split

| Subgraph | Convert? | Core ML API | Host islands | Shipped artifact |
| --- | --- | --- | --- | --- |
| Depthformer logits/state | Yes, first. | `mlprogram`, `ct.StateType` for fixed temporal K/V buffers, `minimum_deployment_target=ct.target.iOS18`, FP16 compute. | Sampling, RNG, top-k/top-p, valid-range masking, CFG setup, previous-frame update. | `exporters/convert_temporal_body.py` (1-frame stateful FP16 temporal body) and `exporters/convert_depth_body.py` (FLOAT32 depth logits). |
| SpectroStream RVQ detokenization | No, not first. | None at baseline. | CPU lookup / embedding accumulation. | `exporters/export_rvq_codebooks.py` exports host-owned codebook binaries (`SpectroStreamRVQCodebooks.f32.bin`). |
| SpectroStream conv synthesis | Yes, second. | Separate fixed-shape `mlprogram`; `.cpuAndGPU` baseline, `.cpuAndNeuralEngine` experiment. | RVQ lookup and iSTFT initially. | `exporters/convert_spectrostream_decoder.py`. |
| SpectroStream iSTFT / overlap-add | No, host baseline. | Consider only after conv proof. | CPU/Accelerate overlap state. | Host-side; streaming-vs-one-shot decode verification lives with the decoder validation tooling. |
| MusicCoCa | No for the first conversion proof. | Existing TFLite/CPU path is acceptable. | Prompt encode/tokenize. | None. |

SpectroStream routing threshold:

- Start with CPU RVQ lookup + `.cpuAndGPU` decoder baseline + CPU/Accelerate
  iSTFT.
- Try `.cpuAndNeuralEngine` only for conv synthesis after RVQ and iSTFT are
  deliberately outside the graph.
- Abandon ANE decoder work if it fails to beat the GPU baseline on p99, power,
  or GPU-contention relief after the host islands are isolated.

### iOS Audio Plan Seed

Any downstream audio layer should start from this generated-PCM contract:

- Producer emits 48 kHz stereo PCM in 1920-frame/channel chunks.
- Consumer is a C/C++ render callback that must never block or allocate.
- Bridge is a lock-free SPSC PCM ring.
- Initial lookahead should start at 120-200 ms and then be sized from measured
  p99 inference, not from mean frame time.
- Runtime must expose underrun counters and complete a 10-minute soak with no
  dropouts before product claims are made.

## Research Brief Coverage and Handoff

| Brief section | Status | Where answered | Remaining unknown |
| --- | --- | --- | --- |
| A. Component topology | Answered | [Component and Clock Inventory](#component-and-clock-inventory), [Per-Frame Dataflow and Tensor Shape Ledger](#per-frame-dataflow-and-tensor-shape-ledger) | Exact live upstream drift was not checked against GitHub because the local checkout and assets were sufficient for this recon. If upstream becomes the source of truth, compare commits before conversion. |
| B. Transformer hot path | Answered | [Depthformer Step](#depthformer-step), [Streaming State and KV-Cache Contract](#streaming-state-and-kv-cache-contract), [Temporal Transformer Ops](#temporal-transformer-ops) | Actual ANE residency and p99 latency remain unknown until `ct.convert` plus device profiling. |
| C. SpectroStream decoder | Answered | [SpectroStream Decode](#spectrostream-decode), [SpectroStream Decode Ops](#spectrostream-decode-ops), [Conversion Source Verdict](#conversion-source-verdict) | Conv-only ANE value versus GPU baseline is unknown until a split decoder proof runs. |
| D. MusicCoCa | Answered | [Component and Clock Inventory](#component-and-clock-inventory), [Clock Cut](#clock-cut) | Prompt-change latency is out of scope for the 25 Hz proof. |
| E. Conversion-readiness verdict | Answered | [Op Inventory and ANE Risk Ranking](#op-inventory-and-ane-risk-ranking), [Ranked Fallback Risks](#ranked-fallback-risks), [Core ML State Verdict](#core-ml-state-verdict) | Core ML op support must be verified by conversion reports and Instruments; this document ranks risk, not residency proof. |
| F. Reference-implementation pragmatics | Answered | [Route Comparison](#route-comparison), [Subgraph Split](#subgraph-split) | Weight mapping from MRT2 safetensors into the minimal PyTorch wrapper was the first implementation task (now `exporters/mrt2_coreml/`). |

### Open Unknowns at Teardown Time

Each unknown had one concrete next probe:

| Unknown | Probe |
| --- | --- |
| Do the 48 temporal K/V `ct.StateType` tensors stay ANE-resident with p99 under 40 ms? | Build the temporal-body converter (now `exporters/convert_temporal_body.py`), run a one-frame fixed-state conversion, inspect Xcode Core ML report and Instruments Core ML trace. Use [`FluidInference/mobius` `coreml-cli`](https://github.com/FluidInference/mobius/tree/main/tools/coreml-cli) as an optional terminal-first fallback scan, not as the only proof. |
| Is CPU sampling after Core ML logits fast enough inside the 40 ms frame budget? | Add a Python parity/profiling harness that runs Core ML logits + CPU top-k/Gumbel sampling for 25 frames and reports p50/p99 (now `validation/validate_temporal_body.py`). |
| Does the SpectroStream conv decoder on ANE beat `.cpuAndGPU` after CPU RVQ lookup and CPU iSTFT are isolated? | Build the decoder converter (now `exporters/convert_spectrostream_decoder.py`), compare `.cpuAndGPU`, `.cpuAndNeuralEngine`, and `.all` with p50/p99 and power. |
| Does frame-by-frame SpectroStream decode match one-shot decode without clicks? | Build a streaming-decode verification harness with SNR, log-spectral distance, and a saved A/B WAV fixture. |
| Can MRT2 safetensors round-trip cleanly into the PyTorch wrapper without silent transposes or dtype drift? | Write a weight-loader unit test that compares selected dense, attention, embedding, and decoder outputs against MLX/JAX on fixed tensors. |

### Do Not Do Next

- Do not build an iOS app, Swift UI, AUv3 surface, or beta-distribution
  artifact from this teardown alone.
- Do not convert `mrt2_base`.
- Do not convert MusicCoCa for the 25 Hz proof.
- Do not start with the full exported `.mlxfn` as a black-box reverse-engineering
  target.
- Do not put RNG, top-k/top-p sorting, and Gumbel sampling in Core ML until a
  CPU-owned sampling proof fails.
- Do not claim ANE residency from `MLComputeUnits.all`.

### Where the Plan Landed

The artifacts seeded by this teardown shipped as:

- `exporters/convert_temporal_body.py` — 1-frame stateful FP16 temporal body
  (`MRT2TemporalBody.mlpackage`).
- `exporters/convert_depth_body.py` — FLOAT32 depth logits body
  (`MRT2DepthBody.mlpackage`).
- `exporters/convert_spectrostream_decoder.py` — FLOAT32 conv decoder
  (`SpectroStreamDecoder.mlpackage`).
- `exporters/export_rvq_codebooks.py` — host-owned RVQ codebook export
  (`SpectroStreamRVQCodebooks.f32.bin`).
- `exporters/export_conditioning.py` — conditioning token assembly assets.
- `validation/validate_temporal_body.py` — logits parity and profiling harness.
- PyTorch wrappers with semantic buffer names in `exporters/mrt2_coreml/`.

Optional profiling command to wire into the proof loop after a `.mlmodelc`
exists:

```bash
uvx --from "coreml-cli @ git+https://github.com/FluidInference/mobius.git#subdirectory=tools/coreml-cli" \
  coreml-cli path/to/model.mlmodelc --fallback --json
```

`coreml-cli` is attractive because it reports per-operation CPU/GPU/ANE
assignment from the terminal, but its detailed fallback mode uses private Core ML
APIs. Use it to shorten the loop; keep Instruments/Xcode/powermetrics as the
device-proof authority.

The first implementation milestone was logits parity, not audio. Once
logits/state were stable, the order was: CPU sampling, then SpectroStream
decode, then iOS audio-ring integration as a separate effort.

## Verification

Teardown-level verification at the time of the source freeze:

- Baseline MLX generation succeeded for one second of `mrt2_small`: 25 frames,
  36.9 steps/s, 27.1 ms/step on an Apple Silicon Mac.
- The state inventory probe regenerated its state ledger byte-for-byte across
  runs.
- `git diff --check` passed for all files touched during the teardown commits.
- Test command used for every teardown commit:

```bash
python3 -m pytest tests
```

Latest observed result: 28 passed, 4 skipped, 4 warnings.
