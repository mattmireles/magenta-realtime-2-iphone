# Magenta RealTime 2 on iPhone

**A 230M-parameter live-music model sustains its compute and audio-delivery
clocks on A17 Pro for ten hot foreground minutes without application GPU
execution. Its long-horizon generative-quality clock still fails. Both results
are measured and published.**

This repository is the public artifact for Google DeepMind's
[Magenta RealTime 2](https://huggingface.co/google/magenta-realtime-2)
(`mrt2_small`) on iPhone: Core ML exporters, frozen fixtures, gate verifiers,
machine-readable device receipts, and the paper source.

> **System paper:** [*The Three Clocks of Live Music Generation: Sustained
> GPU-Free MRT2 Inference on iPhone*](paper/main.pdf)
>
> - A17 Pro: 610.19 s, 1.0308x production, 21.66 ms p99 against a 40 ms
>   deadline, zero underruns/drops, and a growing reservoir from a run that
>   begins in the `serious` thermal state.
> - Placement: 10/10 cold-process temporal ANE admissions; a model-attributed
>   trace records temporal and decoder ANE work and zero app GPU intervals.
> - Audio quality: the corrected 600 s WAV passes delivery, finite, stereo,
>   prompt, and reference-similarity checks but fails the frozen pulse detector
>   and three of five calibrated blind votes. This is not hidden or tuned away.
> - A14: 58.59 ms p99, 0.8967x production, and first underflow at 294.08 s even
>   with a 20.16 s reservoir. It is a bounded-reservoir failure, not a
>   real-time tier.
> - Compression: six 4/6/8-bit post-training variants shrink to 25–50% of
>   baseline but all fail deterministic parity before device timing.

> **Pre-converted models:** [huggingface.co/mattmireles/magenta-realtime-2-iphone](https://huggingface.co/mattmireles/magenta-realtime-2-iphone)
> **Exporters, validation harness, docs (this repo):** [github.com/mattmireles/magenta-realtime-2-iphone](https://github.com/mattmireles/magenta-realtime-2-iphone)

> ### Corrected artifact generation (paper *Surgical Inference*, §6.3–6.5)
>
> This repo now ships the **corrected** exporters that reproduce the paper's
> three headline findings. Three earlier conclusions are superseded — the
> supersession is itself a reported result. **[MODELS.md](MODELS.md)** is the
> authoritative artifact map; **[docs/validation-receipts.md §0](docs/validation-receipts.md)**
> is the corrected evidence.
>
> 1. **State mutation, not attention, is the ANE cliff (§6.3).** The stateless
>    host-owned-cache temporal step (`convert_temporal_body_carry.py`: 48 K/V
>    caches as inputs, one-token updates as outputs, no `ct.StateType`) compiles
>    the full 12-layer stack to one ANE-resident graph (`costWeights=ane:1.000`,
>    p99 14.991 ms on iPhone 12 Pro). Every in-graph `ct.StateType` variant
>    fails `ANECCompile()` with error −14. The system-paper runtime now uses
>    this boundary and passes its in-app plan/state gate in 10/10 cold
>    processes on both tested phones.
> 2. **Layout determines FP16 survival (§6.4).** The decoder FP16 export is
>    finite **and** ANE-resident after a channels-first (NCHW) rewrite plus an
>    exact rescale (`--nchw-parallel-layer 5 --fp16-rescale --compute-precision
>    FLOAT16`). "Do not re-export at fp16" was wrong.
> 3. **Weight bandwidth shapes the graph (§6.5).** Depth samples all 12 RVQ
>    levels in one in-graph FP16 rollout from host Gumbel noise
>    (`convert_depth_body_rollout.py`), not 12 host-side predictions.
>
> Sections below retain earlier artifact-generation history. The system-paper
> box above, `MODELS.md`, and `docs/validation-receipts.md` are authoritative
> where historical measurements differ from the final composed runtime.

## The method: redesign the pipeline, not the model

MRT2's generation step is not one graph — it's a chain with fundamentally
different hardware affinities. Cutting it at the right joints produces three
small graphs that each land on the right silicon:

```
prompt text ──(compiled once, on a Mac)──▶ prompt vector
               │
               ▼
┌──────────────────────────────────┐
│  TEMPORAL  (carry, stateless)    │ ◀── ANE-clean (proven);
│  Predicts the next 40 ms of music│     .cpuAndNeuralEngine; fp16
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  DEPTH  (in-graph rollout)       │ ◀── CPU/GPU/ANE (bandwidth)
│  Samples 12 RVQ levels in-graph  │     fp16, host Gumbel noise
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  RVQ GATHER  (Swift / C++)       │ ◀── CPU
│  Sum 12 codebook rows            │     data-dependent logic
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  DECODER  (SpectroStream, NCHW)  │ ◀── Neural Engine
│  Tokens → spectrogram frames     │     fp16, channels-first
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  iSTFT + OVERLAP-ADD  (C++)      │ ◀── CPU
│  Spectrogram → stereo PCM        │     cheap DSP
└──────────────┬───────────────────┘
               ▼
   48 kHz stereo audio — 25 frames per second
```

Exact tensor shapes and I/O names live in the
[graph teardown](docs/graph-teardown.md) and on the
[model card](https://huggingface.co/mattmireles/magenta-realtime-2-iphone).

The cuts that matter, and why:

1. **The KV cache is a host-owned input, not Core ML state (corrected, §6.3).**
   In-graph state mutation is the ANE admission cliff: every `ct.StateType`
   variant fails `ANECCompile()` with error −14 (25-frame stateful graph, both
   phones, both `.cpuAndNeuralEngine` and `.all`). The escape is the opposite of
   Core ML state — a *stateless* step with the 48 K/V caches as ordinary inputs
   and one-token updates as ordinary outputs, host-owned mutation
   (`convert_temporal_body_carry.py`). That graph compiles the full 12-layer
   stack to one ANE-resident island (`costWeights=ane:1.000`, p99 14.991 ms on
   iPhone 12 Pro). *(The earlier `MRT2TemporalBody.mlpackage` shipped the
   stateful graph and is retained as a negative-result artifact.)*
2. **Sampling entropy stays on the host, but the sampling math can move into the
   graph (corrected, §6.5).** Because per-call cost ≈ weight bytes ÷ DRAM
   bandwidth, 12 depth predictions per frame are bandwidth-doomed; the corrected
   depth body samples all 12 RVQ levels in *one* in-graph rollout fed by
   host-supplied Gumbel noise and inverse temperature. Determinism stays
   host-owned (the seed reproduces this graph's runs); FP32 rollout is
   token-exact (0/900), FP16 ships at 12.7 ms/frame on A14.
3. **RVQ gather stays on the host.** A 12-level codebook lookup is a gather —
  ANE-hostile, trivially fast on CPU. Shipping the table as a 12.6 MB flat
   binary beats embedding it in any graph.
4. **The decoder is FP16 after a channels-first rewrite (corrected, §6.4).** A
   naive channels-last fp16 export is non-finite (finite ratio 0.71) — the
   earlier `SpectroStreamDecoder.mlpackage` shipped fp32/GPU for this reason.
   The finding is that *layout* determines FP16 survival: converting the
   parallel upsampling block to NCHW internally (public channels-last I/O
   preserved) plus an exact rescale (`apply_fp16_safe_rescale`) makes the fp16
   graph finite **and** ANE-resident — on iPhone 12 Pro `.cpuAndNeuralEngine`
   gives finite output (184,320/184,320 at 25-frame, p99 24.77 ms) while
   CPU-only and CPU+GPU are non-finite for the same artifact. Lesson: **validate
   `finite_ratio` per compute unit, on device, not just correlation.** FP32 NCHW
   parity vs MLX is 118.85 dB SNR. ([details](docs/validation-receipts.md))
5. **iSTFT and overlap-add stay on the host.** The decoder's output boundary is
  the pre-iSTFT tensor; streaming overlap state is explicit host code instead
   of hidden graph state. See the [RVQ decoder guide](docs/rvq-decoder.md).
6. **CFG is baked at conditioning-export time.** The on-device graph has no
  runtime classifier-free-guidance machinery; guidance strength is encoded in
   the conditioning tokens when a prompt is compiled. Beware the token-unit
   trap documented in [export_conditioning.py](exporters/export_conditioning.py):
   style/notes CFG tokens use a 0.2-per-token scale, drums 1.0-per-token —
   hardcoding the same integer across slots silently bakes *anti*-guidance.

The long-form version of this analysis is the
[graph teardown](docs/graph-teardown.md). The ANE-specific KV-cache patterns are
in the [stateful KV guide](docs/stateful-kv-coreml.md).

## Why the Neural Engine?

Magenta RealTime 2 already runs on Apple Silicon — Google ships an MLX engine
for Mac **GPUs**. A phone is a different game. Real-time music is not a
3-second benchmark: the model must deliver one 40 ms frame every 40 ms,
indefinitely, on a device with no fan. A fast short run is insufficient: the
selected placement must preserve tail latency through minute ten. The Neural
Engine is specialized for the static-shape FP16 matrix math that dominates
this workload, while leaving the CPU to handle irregular state and audio work.

But the iPhone's NPU has rules. No dynamic shapes. No data-dependent control flow. And
a compiler that fails *silently*: push the whole model through as one graph
and Core ML reports success while quietly scheduling your "Neural Engine
model" on the CPU at ~640 ms per frame — 16× over budget, no error raised. We
hit that cliff, measured it, and published it. Then we cut the pipeline at the
joints.

### Historical ANE-vs-GPU power evidence

The table below belongs to the earlier artifact-generation study, not the
corrected composed pipeline evaluated by the new system paper. The corrected
Power Profiler pair could not be completed after Instruments lost USB
attachment; the system paper therefore makes no energy or impact-score claim.

Routing the temporal transformer to the GPU is not a sidegrade — it costs you
twice. A counterbalanced pair of 60-second live-audio runs on the iPhone 12
Pro, identical except for where the temporal stage executes:


| 60 s live run, iPhone 12 Pro        | temporal on ANE           | temporal on GPU     |
| ----------------------------------- | ------------------------- | ------------------- |
| Process GPU impact (Power Profiler) | **0.000**                 | 2.231               |
| CPU instructions                    | **48.1 billion**          | 110.3 billion       |
| Producer thread busy                | **57% — sleeps the rest** | 93% — nearly pegged |


In that historical configuration, the ANE routing switched the application
GPU work off, halved CPU work, and
the producer thread finished each second of audio early and slept 43% of the
run. That was directional headroom in the earlier runtime, not a
corrected-pipeline energy result. In the Instruments Metal capture, the only
process with GPU time was `backboardd`, iOS's screen compositor. The music used
none.
([receipts §4.4](docs/validation-receipts.md))

## Status — what's proven, what's not

We publish what we've validated. Nothing here is aspirational.


| Claim                                                              | Status        | Evidence                                                                                                                                                                                           |
| ------------------------------------------------------------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Temporal transformer numerically matches the MLX reference         | ✅ Proven      | correlation 0.999985904188, max err 0.118 ([receipts](docs/validation-receipts.md))                                                                                                                |
| Temporal + depth pipeline samples identical tokens (deterministic) | ✅ Proven      | 0/12 mismatches, composed correlation 0.999998250871                                                                                                                                               |
| SpectroStream decoder matches MLX                                  | ✅ Proven      | SNR 118.850 dB, log-spectral distance 0.000722 dB                                                                                                                                                  |
| Stateless temporal stack is ANE-clean on device (§6.3)             | ✅ Proven      | full 12-layer stack `costWeights=ane:1.000`, `preferredCounts=ane:1033,cpu:2`, MLComputePlan iPhone 12 Pro; every `ct.StateType` variant fails `ANECCompile −14`                                    |
| Temporal step fits the budget                                      | ✅ Proven      | p99 ≈ 14.991 ms against a 40 ms frame (stateless temporal only, iPhone 12 Pro)                                                                                                                     |
| Temporal runs on the ANE in the full app                          | ✅ Proven      | 10/10 cold-process plan/state gates on A17 Pro and A14; model-attributed traces show temporal ANE work and zero app GPU intervals ([system-paper receipts](docs/validation-receipts.md))          |
| Decoder FP16 is finite and ANE-resident (§6.4)                     | ✅ Proven      | NCHW `.cpuAndNeuralEngine` finite 184,320/184,320 (25-frame, p99 24.77 ms); CPU-only/CPU+GPU non-finite for the same fp16 artifact                                                                  |
| Depth rollout samples identical tokens (FP32) (§6.5)               | ✅ Proven      | 0/900 mismatches vs the reference sampler; FP16 flips only fp16 near-ties (distribution unchanged)                                                                                                 |
| Sustained playback without dropouts                                | ✅ A17 only    | A17: 610.19 s, 0 underruns/drops. A14 is explicitly rejected: 7,952 underruns and 0.8967x production despite a 20.16 s maximum reservoir ([system-paper receipts](docs/validation-receipts.md))     |
| Survives a thermal soak                                            | ✅ A17 only    | the selected A17 run begins and remains in `serious`, holds p99 at 21.21–23.08 ms per minute, and never underruns; A14 fails the compute clock                                                     |
| The selected A17 hot loop never touches the GPU                    | ✅ Proven      | model-attributed A17 trace shows temporal/decoder ANE work, deliberate CPU depth, and zero application Metal GPU intervals ([system-paper receipts](docs/validation-receipts.md))                    |
| Composed pipeline p99 < 40 ms                                      | ✅ A17 only    | A17 selected policy: 21.66 ms sustained p99. A14: 58.59 ms p99 and a bounded-reservoir failure                                                                                                    |
| Turnkey Swift runtime / demo app                                   | ❌ Not shipped | coming when it meets our bar                                                                                                                                                                       |
| Conditioning preset library                                        | ❌ Not shipped | deliberately — see [Conditioning](#conditioning)                                                                                                                                                   |


## Reproduce our numbers in two minutes

No MLX, no JAX, no checkpoint download — the shipped fixtures contain the MLX
reference tensors:

```bash
pip install coremltools numpy torch
hf download mattmireles/magenta-realtime-2-iphone --local-dir models
PYTHONPATH=exporters python validation/validate_temporal_body.py --skip-pytorch
# Core ML vs MLX max error 0.1178550720   (correlation 0.999985904188)
```

For independent end-to-end verification (recompute the reference yourself), run
the same script without `--skip-pytorch` and without the fixture file, in an
environment with the `magenta_rt` MLX backend and the `mrt2_small.safetensors`
checkpoint.

## Re-export from scratch

The converters need only **PyTorch + coremltools + the checkpoint** — the MLX
stack is not required for conversion:

Corrected generation (paper §6.3–6.5 — the exporters to use):

```bash
# checkpoint: mrt2_small.safetensors from google/magenta-realtime-2
# 1) Stateless temporal step (host-owned K/V caches in, one-token updates out):
PYTHONPATH=exporters python exporters/convert_temporal_body_carry.py
# 2) In-graph FP16 depth rollout (all 12 RVQ levels in one prediction):
PYTHONPATH=exporters python exporters/convert_depth_body_rollout.py
# 3) NCHW FP16 decoder — finite and ANE-resident:
PYTHONPATH=exporters python exporters/convert_spectrostream_decoder.py \
    --nchw-parallel-layer 5 --fp16-rescale --compute-precision FLOAT16
# 4) Flat f32 RVQ codebook table (host gather):
PYTHONPATH=exporters python exporters/export_rvq_codebooks.py
```

Superseded generation (retained as negative-result artifacts, mirrored under
`superseded/` on HF): `convert_temporal_body.py` (stateful fp16) and
`convert_depth_body.py` (fp32 logits + host sampling). Both superseded
exporters are headed as superseded with pointers to the corrected ones and the
paper findings. (`convert_spectrostream_decoder.py` with its FP32 defaults
produces the NCHW FP32 *reference* decoder, which remains current.)

The PyTorch wrappers in [`exporters/mrt2_coreml/`](exporters/mrt2_coreml/)
re-express each subgraph in trace-friendly form and load weights directly from
the safetensors checkpoint.

**The exports are deterministic.** Each exporter writes a `*_export_metadata.json`
provenance sidecar next to its `.mlpackage`. The corrected binaries are hosted
on [Hugging Face](https://huggingface.co/mattmireles/magenta-realtime-2-iphone)
with sha256 checksums in [MODELS.md](MODELS.md); before upload, each package
was regenerated by the commands above and its weight payload matched the
certified original byte-for-byte. The superseded binaries remain mirrored
under the HF repo's `superseded/` directory.

## Conditioning

The temporal body cross-attends to a 256-dim `source_encoded` vector — a
compiled prompt. [exporters/export_conditioning.py](exporters/export_conditioning.py)
compiles any text prompt through MusicCoCa and the MRT2 conditioning encoder
(deterministic; requires the MLX stack on a Mac), baking CFG at reference
strength (3.0, 1.0, 1.0).

We ship exactly **one** conditioning vector — a certified test vector for the
prompt "smooth electronic" — so you can verify the pipeline end to end. We do
**not** ship a preset library: we haven't run the listening validation that
would justify one, and unvalidated presets are how on-device music generation
ends up sounding broken. Compile your own prompts; the exporter is the product.

## What's deliberately not here (yet)

- **A Swift runtime package.** Our internal one drives these exact models in a
one-frame stateful loop behind an `AVAudioSourceNode` and a lock-free ring
buffer, but it doesn't yet meet the bar we'd ask you to build on. The model
card's usage sketch shows the loop structure.
- **On-device text→conditioning.** MusicCoCa runs on the Mac today.
- **Preset/style libraries.** See above.

## License

- **Code** (exporters, wrappers, validation, docs): [Apache-2.0](LICENSE).
- **Converted weights** (HF repo): **CC-BY-4.0**, as derivatives of
[google/magenta-realtime-2](https://huggingface.co/google/magenta-realtime-2).
The conversion is content-preserving — precision and memory-layout transforms
only. See [NOTICE](NOTICE).

## Credits

- **Google DeepMind — the Magenta team**, for Magenta RealTime 2, SpectroStream,
and MusicCoCa, and for shipping real on-device weights under a license that
allows work like this. ([repo](https://github.com/magenta/magenta-realtime),
[models](https://huggingface.co/google/magenta-realtime-2))
- **Apple's coremltools team** for `ct.StateType`.
- Conversion, validation, and port by [Matt Mireles](https://github.com/mattmireles).
Prior art in the same spirit: [kokoro-coreml](https://github.com/mattmireles/kokoro-coreml).

*Real-time music generation in your pocket, with the receipts to prove it.*
