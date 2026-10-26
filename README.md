# Magenta RealTime 2 on iPhone

**A 230M-parameter live-music model sustains its compute and audio-delivery
clocks on A17 Pro for ten hot foreground minutes without application GPU
execution. A three-seed crossover then localizes an apparent long-horizon
quality failure to missing causal decoder history and verifies the repair. A
separate no-reset experiment fails its frozen 600-second liveness gate in all
three seeds, so the deployed claim remains explicitly bounded.**

This repository is the public artifact for Google DeepMind's
[Magenta RealTime 2](https://huggingface.co/google/magenta-realtime-2)
(`mrt2_small`) on iPhone: Core ML exporters, frozen fixtures, gate verifiers,
machine-readable device receipts, and the paper source.

> **System paper:** [*Throughput Is Not Liveness: Three Clocks for GPU-Free
> Music Generation on iPhone*](paper/main.pdf)
>
> - A17 Pro: 610.31 s, 1.0301x production, 24.81 ms p99
>   iteration-normalized active cost against a 40 ms throughput deadline, zero
>   underruns/drops, and a non-draining reservoir through `serious` thermal
>   state. This p99 is a quantile of 25-token iteration means, not per-token
>   tail latency.
> - Placement: 10/10 cold-process temporal ANE admissions; a model-attributed
>   trace records temporal and decoder ANE work and zero app GPU intervals.
> - Causal audio result: stateless decoder windowing adds 0.01706 median seed
>   mean pulse share in 60/60 paired windows; the Core ML graph adds only
>   0.00018. Retaining 12 token frames reduces the excess by 0.01650 in 60/60
>   windows and raises pre-iSTFT correlation from 0.1083 to 0.999999999988.
>   The corrected phone capture matches stateful MLX for the principal seed.
> - Unrefreshed liveness: no-reset produces float-PCM full-scale overrange in
>   3/3 frozen 600-second seeds, while matched ten-second combined reset
>   produces none. K/V-only versus feedback-only attribution is ambiguous; no
>   new causal mitigation is claimed. One blinded non-event human excerpt finds
>   both scientific arms technically acceptable and correctly rejects the
>   corruption control.
> - Steering: the final corrected five-frame decoder runs 600 seconds and 30
>   post-ring transitions at a 160 ms retained queue with zero delivery/proof
>   faults. Its frozen paired-reference detector misses 22/30 changes; among
>   eight detections, end-to-end p95 is 0.947 s. The verified deployment tier
>   is buffered, not responsive or live.
> - A14: 58.59 ms p99, 0.8967x production, and first underflow at 294.08 s even
>   with a 20.16 s reservoir. It is a bounded-reservoir failure, not a
>   real-time tier.
> - Compression: six 4/6/8-bit post-training variants shrink to 25–50% of
>   baseline but all fail deterministic parity before device timing.

> **Pre-converted models:** [huggingface.co/mattmireles/magenta-realtime-2-iphone](https://huggingface.co/mattmireles/magenta-realtime-2-iphone)
> **Exporters, validation harness, docs (this repo):** [github.com/mattmireles/magenta-realtime-2-iphone](https://github.com/mattmireles/magenta-realtime-2-iphone)

> ### Corrected artifact generation (paper *Surgical Inference*, §5.1–5.3)
>
> This repo now ships the **corrected** exporters that reproduce the paper's
> three headline findings. Three earlier conclusions are superseded — the
> supersession is itself a reported result. **[MODELS.md](MODELS.md)** is the
> authoritative artifact map; **[docs/validation-receipts.md §0](docs/validation-receipts.md)**
> is the corrected evidence.
>
> 1. **State mutation, not attention, is the ANE cliff (§5.1).** The stateless
>    host-owned-cache temporal step (`convert_temporal_body_carry.py`: 48 K/V
>    caches as inputs, one-token updates as outputs, no `ct.StateType`) compiles
>    the full 12-layer stack to one ANE-resident graph (`costWeights=ane:1.000`,
>    p99 14.991 ms on iPhone 12 Pro) for one frame; the exporter's default
>    2-frame export falls back to an all-CPU plan, and the runtime ships the
>    one-frame `--frames 1 --streaming` graph. Every multi-frame or (on the A14) rolling-history
>    `ct.StateType` variant fails `ANECCompile()` with error −14 (one-frame
>    fixed-slot graphs compiled only as partial ANE islands, and the one the
>    app used masked its history). The system-paper runtime now uses
>    this boundary and passes its in-app plan/state gate in 10/10 cold
>    processes on both tested phones.
> 2. **FP16 range, not layout, gates the decoder (§5.2).** Without the exact
>    mid-network rescale, FP16 is wrong in either layout: non-finite on CPU and
>    GPU, and ANE-resident but finite at ~0 dB SNR on the ANE. With it
>    (`--nchw-parallel-layer 5 --fp16-rescale --compute-precision FLOAT16`) the
>    FP16 decoder is usable on the A17 Pro (38.8 dB held-out on the ANE)
>    except on windows still beyond FP16 range, and wrong on the A14 ANE, so the
>    A14 runs the FP32 decoder on the CPU. "Do not re-export at fp16" was wrong for
>    the A17 Pro only.
> 3. **Weight bandwidth shapes the graph (§5.3).** Depth samples all 12 RVQ
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
│  DEPTH  (in-graph rollout)       │ ◀── CPU (bandwidth-bound)
│  Samples 12 RVQ levels in-graph  │     fp16, host Gumbel noise
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  RVQ GATHER  (Swift / C++)       │ ◀── CPU
│  Sum 12 codebook rows            │     data-dependent logic
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  DECODER  (SpectroStream, NCHW)  │ ◀── ANE fp16 + rescale (A17 Pro)
│  Tokens → spectrogram frames     │     CPU fp32 (A14)
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

1. **The KV cache is a host-owned input, not Core ML state (corrected, §5.1).**
   In-graph state mutation is the ANE admission cliff: every multi-frame or
   (on the A14) rolling-history `ct.StateType` variant fails `ANECCompile()` with error −14
   (25-frame stateful graph, both phones, both `.cpuAndNeuralEngine` and
   `.all`); the one-frame fixed-slot graphs compiled only as partial ANE islands, and the one the app used masked its history. The escape is the opposite of
   Core ML state — a *stateless* step with the 48 K/V caches as ordinary inputs
   and one-token updates as ordinary outputs, host-owned mutation
   (`convert_temporal_body_carry.py`). For one frame that graph compiles the full
   12-layer stack to one ANE-resident island (`costWeights=ane:1.000`, p99
   14.991 ms on iPhone 12 Pro); the exporter's default 2-frame export falls back
   to an all-CPU plan, so the runtime ships `--frames 1 --streaming`. *(The earlier `MRT2TemporalBody.mlpackage` shipped the
   stateful graph and is retained as a negative-result artifact.)*
2. **Sampling entropy stays on the host, but the sampling math can move into the
   graph (corrected, §5.3).** Because per-call cost ≈ weight bytes ÷ DRAM
   bandwidth, 12 host predictions per frame are boundary-heavy; the corrected
   depth body samples all 12 dependent RVQ levels in *one* in-graph rollout fed
   by host-supplied Gumbel noise and inverse temperature. The graph still
   traverses transformer weights at each dependent level; the large measured
   gain comes from FLOAT16, not a single weight traversal. Determinism stays
   host-owned (the seed reproduces this graph's runs); FP32 rollout is
   token-exact (0/900), FP16 ships at 12.7 ms/frame on A14.
3. **RVQ gather stays on the host.** A 12-level codebook lookup is a gather —
  ANE-hostile, trivially fast on CPU. Shipping the table as a 12.6 MB flat
   binary beats embedding it in any graph.
4. **The decoder is FP16 on the A17 Pro only after an exact rescale
   (corrected, §5.2).** The decoder's activations exceed the FP16 range, and
   without the rescale FP16 is wrong in either layout (channels-last or NCHW):
   non-finite on CPU and GPU, and finite but ~0 dB SNR on the fully resident
   ANE (October 2026 device matrix, both phones). The NCHW rewrite plus an
   exact-in-FP32 rescale (`apply_fp16_safe_rescale`) makes the fp16 graph
   usable (approximate) on the A17 Pro (41–46 dB calibration window, 35–48 dB held-out,
   38.8 dB on the ANE), but it still fails on windows beyond FP16 range after
   the rescale (1 of 600 by host scan; 8.5 dB on the A17 Pro ANE) and is wrong
   on the A14 ANE (−12 to −21 dB), so the A14 runs the FP32 decoder on the
   CPU. The June "finite only on the ANE" probe (184,320/184,320, iPhone 12
   Pro) was the unrescaled graph and its output was wrong. Lesson: **a finite
   ANE output is not a correct one — check SNR against FP32 per compute unit,
   on device.** FP32 NCHW parity vs MLX is 118.85 dB SNR.
   ([details](docs/validation-receipts.md))
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
a compiler that fails *silently*: ask for the Neural Engine and Core ML can
report success while quietly scheduling your "Neural Engine model" on the CPU
— our 2-frame host-cache graph printed `ANECCompile() FAILED`, then completed
on an all-CPU plan, no error returned to the caller. We hit that cliff, measured it, and
published it. Then we cut the pipeline at the
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
| Stateless temporal stack is ANE-clean on device (§5.1)             | ✅ Proven      | full 12-layer stack `costWeights=ane:1.000`, `preferredCounts=ane:1033,cpu:2`, MLComputePlan iPhone 12 Pro; every multi-frame or (on the A14) rolling-history `ct.StateType` variant fails `ANECCompile −14`                                    |
| Temporal step fits the budget                                      | ✅ Proven      | p99 ≈ 14.991 ms against a 40 ms frame (stateless temporal only, iPhone 12 Pro)                                                                                                                     |
| Temporal runs on the ANE in the full app                          | ✅ Proven      | 10/10 cold-process plan/state gates on A17 Pro and A14; model-attributed traces show temporal ANE work and zero app GPU intervals ([system-paper receipts](docs/validation-receipts.md))          |
| Decoder FP16 is usable on the A17 Pro ANE (§5.2)                   | ✅ A17 only    | rescaled FP16 on the A17 Pro ANE: 45.9 dB calibration / 38.8 dB held-out vs on-device FP32, 8.5 dB on the 1/600 window still beyond FP16 range; wrong on the A14 ANE (−12 to −21 dB), so A14 runs the FP32 decoder on the CPU |
| Depth rollout samples identical tokens (FP32) (§5.3)               | ✅ Proven      | 0/900 mismatches vs the reference sampler; FP16 flips only fp16 near-ties (distribution unchanged)                                                                                                 |
| Sustained playback without dropouts                                | ✅ A17 only    | Corrected A17: 610.31 s foreground run, 600.00 s PCM, 0 underruns/drops, and 1.0301x production. A14 is explicitly rejected: 7,952 underruns and 0.8967x production despite a 20.16 s maximum reservoir ([system-paper receipts](docs/validation-receipts.md))     |
| Survives a thermal soak                                            | ✅ A17 only    | the corrected A17 run reaches `serious`, holds p99 iteration-normalized active cost at 24.81 ms across the run, and never underruns; A14 fails the compute clock                                  |
| The selected A17 hot loop never touches the GPU                    | ✅ Proven      | model-attributed A17 trace shows temporal/decoder ANE work, deliberate CPU depth, and zero application Metal GPU intervals ([system-paper receipts](docs/validation-receipts.md))                    |
| Composed pipeline p99 < 40 ms                                      | ✅ A17 only    | Corrected A17: 24.81 ms sustained p99 of 25-token iteration means. A14: 58.59 ms p99 before the context repair and a bounded-reservoir failure                                                     |
| Turnkey Swift runtime / demo app                                   | ❌ Not shipped | coming when it meets our bar                                                                                                                                                                       |
| Conditioning preset library                                        | ❌ Not shipped | deliberately — see [Conditioning](#conditioning)                                                                                                                                                   |


## Reproduce our numbers

The validator needs no MLX or JAX; the shipped fixtures contain the MLX
reference tensors. The temporal package must first be exported from the
checkpoint:

```bash
pip install coremltools numpy torch
# Export the temporal package from the checkpoint (see Re-export from scratch):
PYTHONPATH=exporters python exporters/convert_temporal_body.py
PYTHONPATH=exporters python validation/validate_temporal_body.py --skip-pytorch \
  --model-template build/models/mrt2_temporal_body_unrolled_01.mlpackage \
  --metadata-template build/models/mrt2_temporal_body_unrolled_01_export_metadata.json
# Core ML vs MLX max error 0.1178550720   (correlation 0.999985904188)
```

The mirror no longer hosts `MRT2TemporalBody.mlpackage`, so the validator runs
against your own export.

For independent end-to-end verification (recompute the reference yourself), run
the same script without `--skip-pytorch` and without the fixture file, in an
environment with the `magenta_rt` MLX backend and the `mrt2_small.safetensors`
checkpoint.

## Re-export from scratch

The converters need only **PyTorch + coremltools + the checkpoint** — the MLX
stack is not required for conversion:

Corrected generation (paper §5.1–5.3 — the exporters to use):

```bash
# checkpoint: mrt2_small.safetensors from google/magenta-realtime-2
# 1) Stateless one-frame temporal step (host-owned K/V caches in, one-token
#    updates out). Without these flags the exporter builds the 2-frame bucket,
#    which falls back to the CPU:
PYTHONPATH=exporters python exporters/convert_temporal_body_carry.py --frames 1 --streaming
# 2) In-graph FP16 depth rollout (all 12 RVQ levels in one prediction):
PYTHONPATH=exporters python exporters/convert_depth_body_rollout.py
# 3) NCHW FP16 decoder with exact rescale (A17 Pro; the A14 uses the FP32 default):
PYTHONPATH=exporters python exporters/convert_spectrostream_decoder.py \
    --nchw-parallel-layer 5 --fp16-rescale --compute-precision FLOAT16
# 4) Flat f32 RVQ codebook table (host gather):
PYTHONPATH=exporters python exporters/export_rvq_codebooks.py
```

Superseded generation (retained as negative-result artifacts; their export
metadata, and receipts for the temporal graphs, are under `superseded/` on HF): `convert_temporal_body.py` (stateful fp16) and
`convert_depth_body.py` (fp32 logits + host sampling). Both superseded
exporters are headed as superseded with pointers to the corrected ones and the
paper findings. (`convert_spectrostream_decoder.py` with its FP32 defaults
produces the NCHW FP32 *reference* decoder, which remains current.)

The PyTorch wrappers in [`exporters/mrt2_coreml/`](exporters/mrt2_coreml/)
re-express each subgraph in trace-friendly form and load weights directly from
the safetensors checkpoint.

**The exports are deterministic.** Each exporter writes a `*_export_metadata.json`
provenance sidecar next to its `.mlpackage`. sha256 checksums for the
corrected binaries are in [MODELS.md](MODELS.md); each package was regenerated
by the commands above and its weight payload matched the certified original
byte-for-byte. Of the Core ML packages, the
[Hugging Face mirror](https://huggingface.co/mattmireles/magenta-realtime-2-iphone)
hosts all five corrected packages; the superseded ones must be exported
locally.

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
one-frame host-owned-cache loop behind an `AVAudioSourceNode` and a lock-free ring
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
