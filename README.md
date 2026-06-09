# Magenta RealTime 2 Small on the iPhone Neural Engine

**Live music generation at 25 frames per second, on the phone in your pocket.
Ten unbroken minutes of 48 kHz stereo. Zero dropouts. Even on a 2020 iPhone 12
Pro.**

This is Google DeepMind's [Magenta RealTime 2](https://huggingface.co/google/magenta-realtime-2)
(`mrt2_small`, 230M parameters), surgically cut into three Core ML graphs and
placed on the silicon each one wants. The temporal transformer holds **p99
≈ 14 ms** on the Neural Engine against a 40 ms frame budget. Output correlation
vs Google's MLX reference: **0.999985904188** — we publish all twelve digits
because we measured all twelve. Decoder SNR: **118.85 dB**. Sampled tokens:
identical, 0 of 12 mismatched.

> **Pre-converted models:** [huggingface.co/mattmireles/magenta-realtime-2-iphone](https://huggingface.co/mattmireles/magenta-realtime-2-iphone)
> **Exporters, validation harness, docs (this repo):** [github.com/mattmireles/magenta-realtime-2-iphone](https://github.com/mattmireles/magenta-realtime-2-iphone)

## Why the Neural Engine?

Magenta RealTime 2 already runs on Apple Silicon — Google ships an MLX engine
for Mac **GPUs**. A phone is a different game. Real-time music is not a
3-second benchmark: the model must deliver one 40 ms frame every 40 ms,
indefinitely, on a device with no fan. The GPU can hit the latency; it can't
hold the power budget through minute ten. The Neural Engine — the same silicon
that runs Face ID and on-device Siri — devours static-shape fp16 matrix math
at a fraction of the GPU's draw. It is the only compute unit on the phone
built for this job.

But the ANE has rules. No dynamic shapes. No data-dependent control flow. And
a compiler that fails *silently*: push the whole model through as one graph
and Core ML reports success while quietly scheduling your "Neural Engine
model" on the CPU at ~640 ms per frame — 16× over budget, no error raised. We
hit that cliff, measured it, and published it. Then we cut the pipeline at the
joints.

## The method: redesign the pipeline, not the model

MRT2's generation step is not one graph — it's a chain with fundamentally
different hardware affinities. Cutting it at the right joints produces three
small graphs that each land on the right silicon:

```
prompt text ──(offline, Mac)──► source_encoded [1,1,256]          exporters/export_conditioning.py
                                      │
                ┌─────────────────────▼──────────────────────┐
 25 Hz loop:    │  MRT2TemporalBody (ANE, fp16, stateful)     │   1 frame / call
                │  48 × ct.StateType KV buffers [1,41,8,128]  │
                └─────────────────────┬──────────────────────┘
                                      ▼ temporal_outputs [1,1,1024]
                ┌─────────────────────▼──────────────────────┐
                │  MRT2DepthBody (fp32) → logits [1,12,12294] │   sampling on CPU:
                └─────────────────────┬──────────────────────┘   Gumbel + top-k, host RNG
                                      ▼ 12 RVQ tokens
                       CPU: codebook gather (12 × 1024 × 256 table)
                                      ▼ embeddings [1,T,256]
                ┌─────────────────────▼──────────────────────┐
                │  SpectroStreamDecoder (fp32 conv)           │
                └─────────────────────┬──────────────────────┘
                                      ▼ pre-iSTFT tensor [1,96,480,4]
                       CPU: iSTFT + overlap-add → 48 kHz stereo PCM
```

The cuts that matter, and why:

1. **The KV cache is Core ML state, not an input.** The temporal body exports
   as a 1-frame stateful graph with 48 fp16 `ct.StateType` buffers. Multi-frame
   unrolled and host-carried-cache variants were exported, measured, and
   **rejected**: the ANE compiler cliff sits at exactly two frames. Past it,
   compilation fails (BNNS error -14) and Core ML silently falls back to CPU at
   ~640 ms/frame — 16× over budget with no error surfaced. The negative result
   is documented in the [receipts](docs/validation-receipts.md).
2. **Sampling stays on the host.** Depth logits come out of Core ML; Gumbel
   noise, top-k, and the RNG live in ordinary code. Deterministic parity
   becomes provable (0/12 token mismatches vs MLX) and seeds are reproducible.
3. **RVQ gather stays on the host.** A 12-level codebook lookup is a gather —
   ANE-hostile, trivially fast on CPU. Shipping the table as a 12.6 MB flat
   binary beats embedding it in any graph.
4. **The decoder is FLOAT32 on purpose.** The fp16 export of the conv decoder
   overflowed — 15.7% of its output came back NaN or Inf, audibly corrupt on
   every prompt — *while passing a naive correlation check*. The fp32 export
   measures 118.85 dB SNR vs the reference. Lesson: **validate `finite_ratio`,
   not just correlation.** ([details](docs/validation-receipts.md))
5. **iSTFT and overlap-add stay on the host.** The decoder's output boundary is
   the pre-iSTFT tensor; streaming overlap state is explicit host code instead
   of hidden graph state. See the [RVQ decoder guide](docs/rvq-decoder.md).
6. **CFG is baked at conditioning-export time.** The on-device graph has no
   runtime classifier-free-guidance machinery; guidance strength is encoded in
   the conditioning tokens when a prompt is compiled. Beware the token-unit
   trap documented in [`export_conditioning.py`](exporters/export_conditioning.py):
   style/notes CFG tokens use a 0.2-per-token scale, drums 1.0-per-token —
   hardcoding the same integer across slots silently bakes *anti*-guidance.

The long-form version of this analysis is the
[graph teardown](docs/graph-teardown.md). The ANE-specific KV-cache patterns are
in the [stateful KV guide](docs/stateful-kv-coreml.md).

## Status — what's proven, what's not

We publish what we've validated. Nothing here is aspirational.

| Claim | Status | Evidence |
| --- | --- | --- |
| Temporal transformer numerically matches the MLX reference | ✅ Proven | correlation 0.999985904188, max err 0.118 ([receipts](docs/validation-receipts.md)) |
| Temporal + depth pipeline samples identical tokens (deterministic) | ✅ Proven | 0/12 mismatches, composed correlation 0.999998250871 |
| SpectroStream decoder matches MLX | ✅ Proven | SNR 118.850 dB, log-spectral distance 0.000722 dB |
| Stateful temporal model is ANE-resident on device | ✅ Proven | ~70% of the compute plan on ANE; MLComputePlan + Instruments, iPhone 15 Pro Max |
| Temporal step fits the budget | ✅ Proven | p99 ≈ 14 ms against a 40 ms frame (temporal only, on device) |
| Sustained playback without dropouts | ✅ Proven | 10-minute runs: 0 underruns, 0 dropped frames — iPhone 15 Pro Max **and** iPhone 12 Pro (A14, 2020, with a 15 s startup reservoir) |
| Survives a thermal soak | ✅ Proven | the 10-minute soak pushed iOS thermal state to "serious" — and never dropped a frame. On the A14, the only failure mode was latency headroom at *nominal* thermal; heat was never the limiter |
| Composed pipeline p99 < 40 ms in all configs | ⚠️ Not yet | measured 24–62 ms/frame composed; lookahead absorbs the tail |
| Turnkey Swift runtime / demo app | ❌ Not shipped | coming when it meets our bar |
| Conditioning preset library | ❌ Not shipped | deliberately — see [Conditioning](#conditioning) |

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

```bash
# checkpoint: mrt2_small.safetensors from google/magenta-realtime-2
PYTHONPATH=exporters python exporters/convert_temporal_body.py      # → temporal body, stateful fp16
PYTHONPATH=exporters python exporters/convert_depth_body.py         # → depth logits, fp32
PYTHONPATH=exporters python exporters/convert_spectrostream_decoder.py
PYTHONPATH=exporters python exporters/export_rvq_codebooks.py       # → flat f32 codebook table
```

The PyTorch wrappers in [`exporters/mrt2_coreml/`](exporters/mrt2_coreml/)
re-express each subgraph in trace-friendly form and load weights directly from
the safetensors checkpoint.

**The exports are deterministic.** Running each exporter above with default
arguments reproduces the published artifacts **byte-for-byte** — every
`weight.bin`, the codebook table, and the conditioning test vector match the
sha256 checksums in [MODELS.md](MODELS.md). What we published is exactly what
this code produces from Google's checkpoint; verify it yourself.

## Conditioning

The temporal body cross-attends to a 256-dim `source_encoded` vector — a
compiled prompt. [`exporters/export_conditioning.py`](exporters/export_conditioning.py)
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
