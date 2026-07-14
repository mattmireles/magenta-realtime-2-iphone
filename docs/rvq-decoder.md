# The RVQ Codec Decoder on Apple Silicon & the ANE: An Advanced Developer Field Guide

June 4, 2026

*Deploying a SpectroStream-style neural audio codec decoder (Magenta RealTime 2 lineage) to Core ML / the Apple Neural Engine for real-time, streaming, 48 kHz stereo music generation on iPhone & Apple Silicon.*

> **Correction — FP16 is achievable and ANE-resident (paper §6.4).** This guide
> and the earlier `SpectroStreamDecoder.mlpackage` treated the decoder as an
> FP32/GPU stage ("keep the final conv + iSTFT in fp32 if fp16 SNR is
> marginal"). The shipped finding is stronger: **layout determines FP16
> numerical survival.** The naive channels-last FP16 export is non-finite
> (finite ratio 0.71), but converting the parallel upsampling block to
> channels-first (NCHW) internally — public channels-last I/O preserved — plus
> an exact-in-FP32 mid-network rescale (`apply_fp16_safe_rescale`) makes the FP16
> graph finite **and** ANE-resident. On iPhone 12 Pro, `.cpuAndNeuralEngine`
> (ANE 1.000) produces finite output (184,320/184,320 at 25-frame, p99 24.77 ms;
> 30,720/30,720 at 5-frame, p99 6.65 ms) while CPU-only and CPU+GPU are
> non-finite for the same artifact. Build it with
> `exporters/convert_spectrostream_decoder.py --nchw-parallel-layer 5
> --fp16-rescale --compute-precision FLOAT16`. See `MODELS.md` and
> `docs/validation-receipts.md` §0.

## Related Documentation

- **[Graph teardown](graph-teardown.md)**: SpectroStream decode op map, RVQ
  detokenization, and STFT boundaries before export.
- **[Stateful KV caches guide](stateful-kv-coreml.md)**: `ct.StateType` /
  causal-conv lookback patterns (same primitive as streaming decoder state).
- **[Validation receipts](validation-receipts.md)**: Parity numbers (118.85 dB
  SNR vs the MLX reference) for the published `SpectroStreamDecoder.mlpackage`.
- **[Core ML Tools — Stateful Models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)**:
  Official `StateType` / `make_state()` reference (iOS 18+ / macOS 15+ floor).

Port context: this project bets that `mrt2_small` can hold **25 Hz / 40 ms per
frame** on iPhone. Use this guide for **SpectroStream decode** (RVQ
detokenization, 2D conv synthesis, iSTFT) after the graph teardown confirms
shapes and state contracts. Pair with the [Stateful KV caches guide](stateful-kv-coreml.md)
for the transformer's bounded cache — they are separate conversion projects.

> **Scope & honesty note.** SpectroStream is new (arXiv 2508.05207, Aug 2025), gated, and has **no known public Core ML / ANE port** as of June 2026. Magenta RealTime 2 (MRT2, June 2026) ships its own on-device engine on **Apple Silicon GPUs via MLX + a C++ runtime — NOT the ANE.** This guide therefore separates (a) facts verified from SpectroStream/MRT/MRT2 primary sources, (b) ANE behavior verified from Apple and strong practitioner sources, (c) patterns extrapolated from analogous codecs (EnCodec, DAC, SoundStream, Mimi), and (d) explicitly unverified code patterns. Read the **Source-Quality & Uncertainty Notes** section before you ship anything.

---

## TL;DR — Key Takeaways

- **The decoder is feasible under the 40 ms/frame budget, but the ANE is the wrong default first target, and the #1 risk is the RVQ detokenization gather.** Codebook lookup (`gather`/`nn.Embedding`/`index_select`) reliably falls to CPU on the ANE. **Do the 64-codebook lookup-and-sum on CPU (it's a trivially cheap memory op), and reserve the ANE/GPU for the convolutional synthesis network.** Express the lookup-sum as `one_hot @ codebook` (a matmul / 1×1 conv) only if you must keep it in-graph.
- **SpectroStream's decoder is a 2D-convolutional, spectrogram-domain network that ends in an inverse STFT — it is NOT a time-domain transposed-conv vocoder like HiFi-GAN/DAC.** This is verified from the paper and changes everything: there is no giant 1→1920 transposed-conv stack to upsample; the heavy lifting is 2D transposed convs over a time–frequency grid plus a final iSTFT (which you will almost certainly run on CPU/Accelerate, not the ANE). Most "huge upsampling ratio" anxiety from the DAC/EnCodec literature does not apply directly.
- **Match reality: MRT2 ships on the GPU via MLX, not the ANE.** Google's own page states they "built a C++ inference engine powered by MLX that allows MRT2 to run natively on Apple Silicon… [and] uses the MLX runtime to efficiently execute it on Apple Silicon GPUs" — the strongest possible signal that GPU (`.cpuAndGPU`) is the pragmatic target. Treat an ANE port as an optimization experiment to chase lower power, not the baseline. Prove residency with Xcode's per-op report and `powermetrics`; verify FP16 audio quality against an FP32 reference (SNR, log-spectral distance); and use Core ML **stateful** models (iOS 18+) to carry causal-conv lookback buffers across the 40 ms frames so you don't get boundary clicks.

---

## Orientation: What You Are Actually Building

### The SpectroStream / MRT2 facts you can rely on (verified from primary sources)

From the SpectroStream paper (arXiv 2508.05207, Li et al., Google DeepMind), the Live Music Models paper (arXiv 2508.04651, Lyria Team/Google DeepMind), and the MRT2 materials (magenta.withgoogle.com/magenta-realtime-2, June 4 2026):

| Property | Value | Source |
|---|---|---|
| Sample rate | 48 kHz, stereo (full-band, multi-channel) | SpectroStream §1, abstract |
| Frame (token) rate `f_k` | 25 Hz (→ 1920 samples/frame/channel) | SpectroStream §2; MRT2 |
| RVQ depth `d_c` (full) | 64 quantizers (codebooks) | SpectroStream §2; arXiv 2508.04651 ("deeper residual quantizers (dc = 64)") |
| Codebook size | 1024 (10-bit codes), **not shared** across quantizers | SpectroStream §2 |
| Full bitrate | 16 kbps | SpectroStream §2 |
| Embedding dim (bottleneck) | 256 | SpectroStream §2 |
| Decoder conv depth `C_d` | 64 | SpectroStream Fig. 1 |
| Decoder params | ~36M (of 61M total: 9M enc, 36M dec, 16M quantizer) | SpectroStream §2 |
| **Domain** | **Time–frequency (STFT); decoder ends in inverse STFT** | SpectroStream §2 |
| Architecture | **2D conv** encoder/decoder; decoder uses **transposed convolutions** to upsample; **delayed-fusion / early-splitting** for stereo | SpectroStream §1–2 |
| Convolutions | **Causal**, weight-normalized; **ELU** activations, pre-activation (except first layer) | SpectroStream §2 |
| Look-ahead | 1-embedding decoder look-ahead → ~80 ms total architectural latency | SpectroStream §2 |
| STFT config | window 960, hop 480 → 100 STFT frames/s; Hann window; Nyquist bin omitted, DC kept | SpectroStream §2 |
| Authors' deployment claim | "real-time streaming inference … can be readily achieved on a single desktop CPU, without needing specialized accelerators" | SpectroStream §1 (verbatim) |

On stereo, the paper is explicit: SpectroStream "employs a delayed fusion and early splitting strategy in the encoder and the decoder, respectively, allowing audio channels to be processed independently in some parts of the model but jointly in other parts" (SpectroStream §1).

**MRT2-specific (June 2026):** frame size 40 ms (25 Hz), control latency ~200 ms, models the first **12 RVQ levels** at 3 kbps for the generative path (`d_c = 12`, `|V_c| = 2^10`) per the MRT2 appendix; MRT (v1) used the first 16 levels at 4 kbps. The **codec decoder still reconstructs from whatever RVQ depth is provided** — the LM generating fewer levels does not change the decoder's lookup-and-synthesize structure, only how many codebook vectors are summed. MRT2 ships a **C++ inference engine using MLX that runs on Apple Silicon GPUs**; per Google's apps page, "the Base model requires an M3 Pro / M2 Max or higher [for real-time streaming]. The Small model will run on any Apple Silicon MacBook, including MacBook Air." Notably, that engine "handles other necessary infrastructure (model state, audio buffering / resampling, MIDI input)" — i.e., **streaming state is managed in the C++ host layer, outside the compiled model graph.**

> **Critical architectural correction (verified).** Many on-device "neural vocoder" guides assume a SoundStream/DAC/EnCodec-style **time-domain** decoder: a stack of `ConvTranspose1d` layers that upsamples a ~25–75 Hz latent directly to 24–48 kHz samples with huge cumulative stride. **SpectroStream does not do this.** Per the paper, the decoder operates on **2D time-frequency spectrograms** and applies **inverse STFT at the very end** to recover the waveform. The transposed convolutions upsample a 2D (time × frequency) grid, not a 1D sample axis. Consequences:
> - The "1920× upsampling cliff" is distributed across (a) modest 2D transposed-conv upsampling and (b) a deterministic, weight-free **iSTFT** (overlap-add of 960-sample windows at hop 480).
> - The iSTFT is a fixed linear operation. It is **not** a natural ANE op; plan to run it on CPU via vDSP/Accelerate or as a fixed `conv_transpose` "synthesis filterbank." This is itself a likely CPU fallback point and a graph-split boundary.
> - This is the single biggest reason to be skeptical of blindly porting DAC/HiFi-GAN ANE recipes to SpectroStream.

### What the hot path looks like, per 40 ms frame

```
[LM produces RVQ token indices]  →  (per frame: d_c indices, each in [0,1024))
        │
        ▼  (1) DETOKENIZE: look up d_c codebook vectors (dim 256) by index, sum them
[continuous embedding, dim 256]
        │
        ▼  (2) reshape to 2D time-frequency latent; carry causal-conv state
[2D conv synthesis net: transposed convs, ELU, weight-normed, causal]
        │
        ▼  (3) inverse STFT (overlap-add) + early-splitting to 2 channels
[1920 stereo samples = 40 ms of 48 kHz audio]
```

Stage (1) is the classic ANE gather hazard. Stage (2) is ANE-friendly **if** expressed correctly. Stage (3) is almost certainly CPU.

### The hard latency math

- 25 Hz → **40 ms per frame** is your hard wall. Decode must complete (p99) well under 40 ms to leave headroom for the LM (temporal + depth transformers), audio buffering, and resampling. MRT2's published end-to-end control latency is ~200 ms, of which codec decode is only one component.
- The decoder competes for compute with the LM every frame. On a unified-memory SoC, ANE/GPU/CPU contention is real: if the LM is on the GPU (as in MRT2's MLX engine), running the codec on the ANE can be attractive precisely because it uses a *different* engine — but only if the gather and iSTFT don't force constant ANE↔CPU ping-ponging that eats the savings.

---

## Implementation Reference

### 1. RVQ Detokenization / Codebook Lookup — the #1 ANE risk

**The problem.** RVQ detokenization is: for each of `d_c` quantizers, take the integer index, fetch row `index` from that quantizer's `[1024, 256]` codebook, and sum the `d_c` resulting `[256]` vectors into one `[256]` embedding. In PyTorch this is `nn.Embedding` or `torch.gather` or fancy indexing. **All of these lower to `gather`/`embedding` ops that are well-documented to fall back to CPU on the ANE.** This is the same failure that plagues token-embedding and LM-head/classifier layers in Core ML LLM deployments: the Orion ANE characterization paper (arXiv 2603.06728) lists "Embedding lookup → CPU (Table indexing)" and "Classifier backward → CPU (32K channels rejected)" in its CPU/ANE division-of-labor table, and Apple's own `gather` lives in `coremltools` MIL but is not an ANE-friendly op.

**Why it (almost) doesn't matter here.** The detokenization is *cheap*: `d_c` lookups of a 256-vector plus a sum. For `d_c = 64`, that's 64 × 256 = 16,384 fp16 reads and a reduction — microseconds on CPU. The expensive part is the conv synthesis net. **So the dominant, recommended pattern is to split the graph: detokenize on CPU, hand the `[256]`-dim (reshaped to 2D) embedding to the Core ML conv model.**

#### Candidate implementations

**(A) `nn.Embedding` sum — clean, but a CPU fallback if left in the Core ML graph**

```python
import torch, torch.nn as nn

class RVQDetokenizer(nn.Module):
    """d_c codebooks, each [vocab, dim]. Input: indices [B, d_c]. Output: [B, dim]."""
    def __init__(self, d_c=64, vocab=1024, dim=256):
        super().__init__()
        # One embedding table per quantizer (codebooks are NOT shared in SpectroStream).
        self.codebooks = nn.ModuleList([nn.Embedding(vocab, dim) for _ in range(d_c)])

    def forward(self, idx):                      # idx: [B, d_c] int32
        out = 0
        for q, emb in enumerate(self.codebooks):
            out = out + emb(idx[:, q])           # gather -> CPU on ANE
        return out                               # [B, dim]
```

**(B) one-hot @ codebook — expresses the gather as a matmul (ANE-friendly), at a cost**

```python
class RVQDetokenizerMatmul(nn.Module):
    """Gather-as-matmul: one_hot(idx) @ codebook. Stays on ANE/GPU as a matmul/conv."""
    def __init__(self, d_c=64, vocab=1024, dim=256):
        super().__init__()
        # Stack all codebooks: [d_c, vocab, dim]
        self.register_buffer("W", torch.randn(d_c, vocab, dim))

    def forward(self, idx):                      # idx: [B, d_c] int
        B, d_c = idx.shape
        oh = torch.nn.functional.one_hot(idx, num_classes=self.W.shape[1]).float()  # [B, d_c, vocab]
        # einsum: sum over vocab AND over d_c quantizers -> [B, dim]
        return torch.einsum("bqv,qvd->bd", oh, self.W)
```

The matmul form trades a memory-bound gather for a compute-bound matmul: `d_c × vocab × dim = 64 × 1024 × 256 ≈ 16.8M` MACs per frame just for detokenization. That's still tiny (≪ the conv net), and it keeps the op on the ANE's fast datapath. **But** the `one_hot` itself can lower to scatter/gather-like ops; verify it doesn't reintroduce a fallback. A safer variant precomputes `one_hot` on CPU (it's a trivial index→sparse op) and feeds dense one-hots in. (Note: Apple's ANE article shows the 1×1-conv-as-matmul idiom is exactly how Apple maps `nn.Linear` to the ANE.)

**(C) Split-to-CPU (RECOMMENDED) — detokenize outside Core ML entirely**

```python
# Swift/C++ side (host): codebooks held as a flat [d_c, vocab, dim] fp16 buffer.
# For each frame, sum the d_c selected rows into a [dim] vector, then hand to Core ML.
# This is a pure memory gather + add — a few microseconds — and never touches the ANE.
# This mirrors what MRT2's own C++ engine does: it keeps "model state, audio buffering /
# resampling" in the host layer and feeds the compiled graph clean tensors.
```

#### Comparison

| Approach | Stays on ANE? | Cost/frame | Risk | Verdict |
|---|---|---|---|---|
| `nn.Embedding` sum (in-graph) | **No** (gather→CPU) | ~free | Forces ANE→CPU split mid-graph; ping-pong overhead | Avoid in-graph |
| `one_hot @ codebook` (matmul) | **Yes** (matmul/conv) | ~16.8M MACs | `one_hot` may re-trigger fallback; verify | OK if you must keep it in-graph |
| **Split to CPU** | N/A (off-graph) | ~free | None; one clean ANE/GPU entry point for the conv net | **Recommended** |

> **Extrapolated, not verified for SpectroStream:** the exact codebook tensor shapes and whether MRT2 fuses the per-quantizer sum. The lookup-and-sum semantics are verified (RVQ; 64 non-shared codebooks of 1024×256); the Core ML lowering behavior is extrapolated from general ANE/coremltools behavior and LLM-deployment experience.

### 2. The Transposed-Convolution / Upsampling Synthesis Network on the ANE

Because SpectroStream is **spectrogram-domain**, the synthesis net is a stack of **2D** transposed convolutions over (time × frequency), mirroring the strided 2D encoder, followed by iSTFT. Apply ANE conv principles:

#### 2.1 Layout: 1D-as-2D, channels-first `(B, C, 1, W)`

Apple's "Deploying Transformers on the Apple Neural Engine" article is explicit: the ANE's preferred data format is **4D and channels-first**, `(B, C, 1, S)`, "because the most conducive data format for the ANE (hardware and software stack) is 4D and channels-first." Critically, "the last axis of an ANE buffer is not packed; it must be contiguous and aligned to 64 bytes." For SpectroStream's genuinely-2D decoder you already have `(B, C, F, T)` 2D feature maps — keep channels first and make the **time axis the last axis** so streaming concatenation along time is cheap.

#### 2.2 ConvTranspose vs resize+conv vs pixel-shuffle (depth-to-space)

`ConvTranspose` (deconvolution) is the canonical neural-vocoder upsampler but has two ANE-relevant problems: (1) **checkerboard artifacts** when kernel size isn't divisible by stride (Odena et al., distill.pub) — audible as periodic tonal noise in audio (the Dolby "neural upsampling artifacts in audio" study confirms transposed and sub-pixel convs introduce tonal artifacts); (2) variable/uneven ANE support historically. The three alternatives:

| Upsampler | ANE friendliness | Artifacts | Notes |
|---|---|---|---|
| `ConvTranspose` (deconv) | Mixed; historically uneven ANE support | Checkerboard if k % stride ≠ 0 | Make kernel a multiple of stride to suppress artifacts |
| **Resize (NN upsample) + Conv** | NN-upsample is documented to run on ANE; conv is a first-class ANE op | None by default | Apple's `Upsample` (integer NN/bilinear) op; practitioner reports of ANE support |
| **Pixel-shuffle (depth-to-space)** | Reshape+transpose; *may* trigger memory copies / fallback | None after ICNR init | Not a built-in Core ML op — composed from reshape/permute; risky on ANE |

**Recommendation:** prefer **nearest-neighbor resize + conv** as the ANE-safe upsampler (Apple's own `Upsample` layer is reported to run on the ANE, and it sidesteps checkerboarding entirely), or keep `ConvTranspose` with **kernel size a multiple of stride**. Avoid pixel-shuffle on the ANE unless you've profiled the reshape/permute and confirmed no memory-copy fallback — Apple's ANE guidance warns that "reshape and transpose operations are likely to trigger memory copies unless specifically handled." **Caveat:** changing the upsampler means **retraining or fine-tuning** SpectroStream, which you cannot do without the (gated) training pipeline — so in practice you are stuck with whatever SpectroStream actually uses (ConvTranspose, per the paper), and your real lever is graph-level: fuse, keep channels-first, and split the iSTFT out.

#### 2.3 Depthwise/separable convs

Depthwise-separable convs run well on the ANE and cut MACs, but again: you cannot swap conv types without retraining SpectroStream. Relevant only if you train your own codec.

### 3. Stereo Output

SpectroStream uses **delayed fusion (encoder) / early splitting (decoder)**: channels are processed jointly in the middle of the network and split into per-channel (L/R) streams near the output, to balance per-channel fidelity against cross-channel phase consistency. For Core ML:

- Model the 2 output channels as **conv channels** through the joint section, then split. Keep everything channels-first `(B, C, F, T)`.
- The two per-channel iSTFT operations are independent and can both run on CPU/Accelerate.
- Don't naively treat stereo as `batch=2`: the **joint** layers genuinely mix L/R (that's the whole point of delayed fusion), so a batch split would be wrong for those layers. Only the early-split tail is per-channel.

### 4. Streaming / Stateful Decode — avoiding boundary clicks

**Is SpectroStream decode stateless per frame?** Verified facts: SpectroStream uses **causal convolutions** with a small (1-embedding) look-ahead, explicitly to "perform real-time streaming inference." Causal conv streaming inherently requires **carrying the convolution receptive-field state (left-context lookback) across frames** — otherwise each 40 ms frame is decoded with zero-padded history and you get discontinuities (clicks/pops) at every frame boundary. MRT2 confirms a frame-by-frame streaming decode at 40 ms, with streaming state managed in its C++ host layer.

There are two ways to handle conv state across frames:

**(A) Stateful causal convs with a padding/lookback cache (RECOMMENDED).** This is exactly what EnCodec / `cached_conv` (ACIDS-IRCAM) do: cache the last `(kernel-1)*dilation` input samples of each conv layer and prepend them next frame. Core ML supports this directly via **stateful models (iOS 18+ / macOS 15+)**: register the lookback buffers as PyTorch `register_buffer` state, and coremltools converts them to in-place `MLState` tensors — the same mechanism used for LLM KV-caches. See the [Stateful KV caches guide](Stateful-KV-caches-CoreML-guide.md) for ANE vs GPU state patterns and verification; conv lookback is the same primitive with a smaller state tensor.

```python
import torch, torch.nn as nn

class StreamingCausalConv1d(nn.Module):
    """Causal conv that carries (k-1)*dilation samples of lookback as Core ML state."""
    def __init__(self, cin, cout, k, dilation=1):
        super().__init__()
        self.k, self.dil = k, dilation
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)  # NO internal padding
        # State buffer: last `pad` input frames. Shape [1, cin, pad].
        self.register_buffer("lookback", torch.zeros(1, cin, self.pad))

    def forward(self, x):                       # x: [1, cin, T_frame]
        x = torch.cat([self.lookback, x], dim=-1)        # prepend history
        if self.pad > 0:
            self.lookback = x[..., -self.pad:]            # retain tail for next frame
        return self.conv(x)                              # valid conv -> [1, cout, T_frame]
```

> **Unverified code pattern.** The exact slice bookkeeping for the lookback update (which samples to retain, and ordering relative to the conv) must be validated numerically against a non-streaming reference (decode a long clip in one shot vs. frame-by-frame; assert sample-level equality). The `cat`-of-state pattern and `register_buffer`→`MLState` conversion are verified; the specific indices here are illustrative.

Conversion sketch (stateful, iOS 18):

```python
import coremltools as ct, numpy as np
traced = torch.jit.trace(decoder.eval(), example_frame)   # fixed-shape frame input
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="emb", shape=(1, 256, 1, T), dtype=np.float16)],
    outputs=[ct.TensorType(name="wave", dtype=np.float16)],
    states=[ct.StateType(  # one per cached conv; name must match named_buffers()
        wrapped_type=ct.TensorType(shape=(1, C, PAD), dtype=np.float16),
        name="lookback_0")],
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_NE,   # see §5 / debugging on why not .ALL
)
```

**(B) Crossfade / overlap-add between independently-decoded frames (fallback).** If you can't get stateful conv state working, decode slightly-overlapping windows and crossfade. This is simpler but (1) wastes compute on the overlap, (2) adds latency, and (3) only *masks* boundary discontinuities rather than eliminating them. For a causal conv decoder, **(A) is strictly more correct.** The iSTFT itself is naturally an overlap-add operation, so you must in any case carry the iSTFT's overlap buffer (the trailing half-window) across frames — this is a second, separate piece of streaming state from the conv lookback.

> **Connection to SpectroStream specifically (extrapolated):** the paper says decode is streaming with causal convs, which *implies* a lookback-state implementation in their reference (JAX/SeqIO `sequence_layers`), but the exact cache layout is not public. MRT2's MLX engine explicitly "handles model state, audio buffering / resampling" in its C++ layer — i.e., they keep streaming state outside the compiled graph. That's a strong hint that **managing conv/iSTFT state in your host code (Swift/C++) rather than as Core ML MLState is a viable, possibly simpler, design** — and it's the approach Google actually shipped.

### 5. ANE Constraints That Specifically Bite Audio Codecs

#### 5.1 FP16-only precision and audio quality

The ANE computes in **fp16 for everything** (Hollemans; corroborated by the Orion paper: "The ANE operates natively in fp16 (±65,504 dynamic range)"). Two audio-specific consequences:

- **Dynamic range / underflow.** fp16's smallest normal ~6e-5; values `< ~1e-4` lose precision or flush to zero (Hollemans: "very small numbers become 0"). Audio waveforms in `[-1, 1]` and quiet passages / reverb tails live exactly in this danger zone. Expect raised noise floor and loss of low-level detail vs. fp32.
- **Overflow / clipping.** fp16 max ±65504. Intermediate conv activations in a GAN vocoder can spike; Orion's fix was to "clamp activations to [−65504, +65504]" before softmax/layernorm. For audio, an analogous guard is a **final `tanh` or clamp** on the output; verify SpectroStream's final activation and preserve it.

**Verification (do this):** decode a test set on (a) fp32 reference (PyTorch/CPU) and (b) the fp16 Core ML model; compute per-frame **SNR**, **log-spectral distance**, and listen for the boundary/quiet-passage artifacts above. If fp16 SNR is unacceptable, keep the most sensitive layers (final conv + iSTFT) in fp32 on CPU/GPU.

#### 5.2 The ~32 MB on-chip SRAM working-set cliff

The ANE has a limited on-chip working set; exceeding it spills to DRAM and tanks throughput. The best-sourced figure: the Orion paper states a **32 MB on-chip SRAM** budget for the M4 Max ANE, with "Performance drops ∼30% when working sets exceed the 32 MB SRAM budget, forcing spills to DRAM." **Important caveat:** this 32 MB number is **not an Apple-published spec** — it originates from one reverse-engineering benchmark (maderix/manjeet singh, "Inside the M4 Apple Neural Engine," Feb 2026), inferred from where a matmul throughput cliff appears between a 24 MB (fast) and 96 MB (−30%) working set, and is explicitly a **soft, cache-like cliff, M4/M4-Max-specific** ("isn't a hard wall… a cache-like hierarchy rather than a hard scratchpad"), not a hard scratchpad wall. Treat it as an order-of-magnitude design constraint, not gospel; earlier chips (A-series, M1–M3) may differ.

Audio relevance: a single frame's activations are small (1920 samples × 2 ch × conv-channel-width is on the order of a few MB at `C_d = 64`), so you are unlikely to blow 32 MB on activations alone for one 40 ms frame. The danger is (a) wide intermediate 2D feature maps if frequency × time × channels balloons, and (b) batching multiple frames. **Keep per-frame working set comfortably under ~32 MB; don't batch frames on the ANE.**

#### 5.3 Layout alignment / minimize concat-transpose-reshape

- Last-axis 64-byte alignment: per Apple, if the last axis "is used as a singleton one by the model implementation's data format, it will be padded to 64 bytes, which results in 32 times the memory cost in 16-bit and 64 times the memory cost in 8-bit precision." Never put a size-1 axis last; put the **time axis** last.
- Orion notes a **~49 KB minimum IOSurface size** — tiny single-frame tensors get padded up (e.g., a `[1, 768, 1, 1]` token tensor must be padded to `[1, 768, 1, 16]`); don't be surprised by allocation overhead on small per-frame I/O.
- Apple: "reshape and transpose operations are likely to trigger memory copies unless specifically handled." The detokenize→2D-reshape boundary and any `(B,C,F,T)`↔`(B,C,1,W)` juggling are prime offenders. **Minimize concat/transpose/reshape on the hot path; fold reshapes into the CPU detokenization step where they're free.**

#### 5.4 `MLComputeUnits`: why `.all` can hurt

Swift uses `MLComputeUnits`; Python/coremltools uses `ct.ComputeUnit` with the same semantics (`ALL`, `CPU_AND_NE`, `CPU_AND_GPU`, `CPU_ONLY`). Default is **all units** (`ALL` / `.all`).

| Swift / Python | Meaning | When to use for the decoder |
|---|---|---|
| `.all` / `ALL` | CPU + GPU + ANE; Core ML auto-partitions | **Often harmful here:** can ping-pong gather→CPU, conv→ANE, iSTFT→CPU, paying inter-engine transfer overhead each frame |
| `.cpuAndNeuralEngine` / `CPU_AND_NE` | CPU + ANE only | Good for an ANE port: keeps the conv net on ANE, gather/iSTFT on CPU, no GPU contention with the LM |
| `.cpuAndGPU` / `CPU_AND_GPU` | CPU + GPU only | **Pragmatic default** — matches MRT2's MLX-on-GPU reality; predictable, no ANE fallback surprises |
| `.cpuOnly` / `CPU_ONLY` | CPU only | Baseline for output-parity testing & SpectroStream authors' own "single desktop CPU" claim |

`.all` lets Core ML make per-op placement decisions you can't see, and for a graph with known CPU islands (gather, iSTFT) it tends to over-fragment. **Pin the compute units explicitly and measure.** Note a known footgun: numerically `.all`/ANE can differ from `.cpuOnly` due to fp16 rounding — always parity-test.

### 6. Conversion Specifics (PyTorch → Core ML)

#### 6.1 Fuse/remove weight normalization BEFORE tracing

SpectroStream's convs are **weight-normalized** (verified). coremltools historically chokes on `_weight_norm` (GitHub coremltools #1347: "PyTorch convert function for op '_weight_norm' not implemented"). **Always call `torch.nn.utils.remove_weight_norm` (or `parametrize.remove_parametrizations`) on every conv before tracing** — this folds `g` and `v` into a single dense weight, which is both convertible and faster.

```python
import torch
from torch.nn.utils import remove_weight_norm  # or torch.nn.utils.parametrize

def defuse_weight_norm(module):
    for m in module.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d,
                          torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)):
            try:
                remove_weight_norm(m)      # newer PyTorch: parametrize.remove_parametrizations(m, "weight")
            except (ValueError, RuntimeError):
                pass                        # not weight-normed; skip
    return module

decoder = defuse_weight_norm(decoder).eval()
```

#### 6.2 `torch.jit.trace` vs `torch.export`

- **`torch.jit.trace`** remains the mature, highest-coverage path for coremltools (the `torch.export` front-end was at ~56% op parity in recent coremltools releases). For a fixed-shape, per-frame decoder, **trace with a fixed example frame** — this also dodges dynamic-shape ANE issues (see G9).
- Trace with the **exact per-frame input shape** you'll use at runtime. Fixed shapes are an ANE prerequisite; dynamic/flexible shapes frequently force CPU/GPU fallback (whisper.cpp lesson — its Core ML decoder couldn't go to ANE due to dynamic shapes; ONNX Runtime CoreML EP exposes `RequireStaticInputShapes` for exactly this reason).

#### 6.3 Activations

SpectroStream uses **ELU** (verified). ELU is supported by coremltools. Other common audio activations and ANE notes:

| Activation | Used by | ANE/coremltools note |
|---|---|---|
| **ELU** | **SpectroStream** | Supported; verify it stays on ANE (elementwise, should) |
| LeakyReLU | SpectroStream discriminator (not needed at inference), EnCodec | Supported |
| Snake (`x + sin²(ax)/a`) | DAC, BigVGAN | **Not a native op** — composes from sin/mul/add; verify no fallback; relevant only if you deploy DAC, not SpectroStream |
| `tanh` (final) | many vocoders | Supported; good fp16 output guard (clamps to [-1,1]) |

Since SpectroStream uses ELU and ends in iSTFT (not `tanh`), confirm the actual final op from the weights; add an explicit output `clamp(-1, 1)` if needed for fp16 safety.

#### 6.4 Pass-pipeline gotchas

A known macOS 26.x bug: `common::fuse_transpose_matmul` can produce **NaN on GPU** for some sliced-tensor matmuls; workaround is `pipeline.remove_passes(['common::fuse_transpose_matmul'])`. Keep this in your back pocket if you see GPU-only NaNs.

---

## Debugging & Verification Reference

### D1. Prove the decoder is actually on the ANE (not CPU)

See also the **`coreml-profile` skill** (`.claude/skills/coreml-profile/`) and
[Latency benchmark](../../docs/benchmark.md) for the full verify/profile workflow.

1. **Xcode Core ML Performance Report** (Xcode 14+): add the `.mlpackage`, open the **Performance** tab, generate a report on a connected device. It shows **per-layer dispatch** to ANE / GPU / CPU. This is the authoritative, first-stop tool. (Apple demonstrated exactly this for the distilbert case study.)
2. **Symbolic breakpoint** `-[_ANEModel program]` — if it's hit, *some* of the model is on the ANE (Hollemans). For full-model residency you must check that no Espresso CPU/GPU engine calls appear for your layers.
3. **Pause-and-inspect**: pause in the debugger; an `H11ANEServicesThread` indicates ANE use (Hollemans).
4. **`.all` vs `.cpuAndGPU` vs `.cpuOnly` A/B**: if `.all` isn't much faster than `.cpuOnly`, the ANE is barely being used (Hollemans).

### D2. Measure ANE power / utilization

```bash
# Estimated per-subsystem power incl. ANE (root required). macOS.
sudo powermetrics --samplers ane_power -i 200
# Broader view (CPU+GPU+ANE):
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 200
```

`powermetrics` reports **estimated** ANE power (mW). Non-zero ANE power during decode ⇒ ANE is doing work; flat-zero ⇒ it isn't (Eclectic Light's ANE experiments show ANE power sits at 0 mW unless genuinely invoked — e.g., 30→22→49 mW across sampling periods during a real ANE workload). Caveat: values are **estimated, uncalibrated**, not for cross-device comparison. `asitop`/`powermetrics-tui` wrap this in a live dashboard.

### D3. Per-frame latency under the 40 ms budget

- Benchmark the compiled model with `MLModel.prediction` in a tight loop (≥1000 frames); record **p50 and p99**, not just mean. The 40 ms wall is a **p99** constraint for glitch-free audio.
- Use **Instruments → Time Profiler** / the Core ML & Neural Engine templates. Focus the call tree on `-[MLNeuralNetworkEngine predictionFromFeatures:]`; ANE work shows `-[_ANEClient evaluateWithModel...]` (Hollemans).
- **Warm-up matters:** first run on a device triggers ANE model compilation (can be seconds–minutes for big models; whisper.cpp notes "first run on a device may take a while"). Pre-warm at app launch with dummy predictions. Also beware **ANE power-state wake jitter**: practitioners report live/streaming inference being 10×+ slower than offline synthetic benchmarks (one Apple-forum report: ~1.3 ms synthetic vs ~16 ms live) due to the ANE dropping to a low-power state between sparse calls — for a 25 Hz steady cadence this should stay warm, but verify under real streaming load.

### D4. Detect the gather falling to CPU

In the Xcode performance report, look for the embedding/gather op assigned to **CPU** while the convs are on ANE/GPU — that's your detokenization fallback and a graph-split boundary. (This is expected and fine if you've split detokenization to CPU deliberately; it's a problem only if it sits *between* two ANE sections and causes ping-pong.)

### D5. Audio-quality verification (FP16 vs reference)

```python
import numpy as np
def snr_db(ref, test):
    ref, test = ref.astype(np.float64), test.astype(np.float64)
    noise = ref - test
    return 10*np.log10((ref**2).sum() / max((noise**2).sum(), 1e-20))
# Also compute log-spectral distance (per-frame STFT magnitude, dB L2).
```

- Compare Core ML fp16 output vs PyTorch fp32 reference on identical token inputs.
- Targets are application-dependent; for music, watch the **quiet passages** (fp16 underflow) and **transients** (clipping).
- **Boundary/click detection:** decode a long clip (a) one-shot and (b) frame-by-frame-with-state; the difference should be ~numerical-noise. A spike at frame boundaries (every 1920 samples) = broken conv/iSTFT state. Also inspect the waveform's sample-to-sample difference at frame edges for discontinuities, and listen for a 25 Hz buzz (periodic boundary clicks become a 25 Hz tone).

---

## Failure Modes / Gotchas

- **G1 — Gather/embedding falls to CPU.** Detokenization (`nn.Embedding`/`gather`/`index_select`) won't run on ANE. *Fix:* split detokenization to CPU (cheap), or express as `one_hot @ codebook` matmul. Don't let it sit between two ANE sections.
- **G2 — ConvTranspose unsupported/slow or checkerboarding.** *Fix:* ensure kernel size is a multiple of stride; consider NN-resize+conv (ANE-supported, no checkerboard). But you can't change SpectroStream's upsampler without retraining — so this mostly applies if you train your own codec.
- **G3 — FP16 audio degradation.** Raised noise floor (underflow of quiet detail), clipping (overflow). *Fix:* final `tanh`/clamp; keep final conv + iSTFT in fp32 on CPU; SNR/LSD-test.
- **G4 — Boundary clicks/pops from per-frame statelessness.** Zero-padded conv history each frame → discontinuities → a 25 Hz buzz. *Fix:* carry causal-conv lookback **and** iSTFT overlap buffers across frames (stateful model or host-managed state).
- **G5 — iSTFT mis-handled.** Treating the final inverse-STFT as just another conv on the ANE, or forgetting its overlap-add state. *Fix:* run iSTFT on CPU/Accelerate; carry the trailing half-window overlap as separate streaming state.
- **G6 — Receptive-field/padding state bug.** Off-by-one in the lookback slice → subtle artifacts that one-shot decoding hides. *Fix:* numerical parity test (one-shot vs streaming) to sample equality.
- **G7 — `weight_norm` not fused.** coremltools errors on `_weight_norm`. *Fix:* `remove_weight_norm` on all convs before tracing.
- **G8 — Wrong layout / singleton last axis.** Size-1 last axis → up to 32× fp16 memory blowup; time axis not last → expensive streaming concat. *Fix:* channels-first `(B,C,1,W)` with **time last**.
- **G9 — Dynamic shapes force fallback.** Flexible input shapes frequently kick the model off the ANE. *Fix:* fixed per-frame shape; trace with that exact shape.
- **G10 — `.all` over-fragments the graph.** Inter-engine ping-pong (CPU gather ↔ ANE conv ↔ CPU iSTFT) eats the savings. *Fix:* pin `.cpuAndNeuralEngine` or `.cpuAndGPU`; measure.
- **G11 — SRAM working-set spill.** Wide 2D feature maps or batched frames exceed ~32 MB → ~30% throughput drop. *Fix:* keep per-frame working set small; never batch frames on ANE.
- **G12 — Multi-stage pipeline contention.** The codec competes with the LM (temporal+depth transformers) every 40 ms. *Fix:* put codec and LM on different engines (e.g., LM on GPU à la MRT2, codec on ANE) — but only if the gather/iSTFT splits don't reintroduce GPU/CPU contention.
- **G13 — First-run compile + power-state jitter.** Cold-start ANE compile (seconds+) and wake latency from idle. *Fix:* pre-warm at launch; keep the 25 Hz cadence steady to hold the ANE warm.
- **G14 — Assuming SpectroStream is a time-domain vocoder.** Porting DAC/HiFi-GAN 1D-ConvTranspose recipes wholesale. *Fix:* remember it's spectrogram-domain + iSTFT (verified); the upsampling is 2D and modest, the iSTFT is the real "upsampler."

---

## Best Practices (consolidated)

1. **Split the graph:** detokenize (gather+sum) and iSTFT on CPU; conv synthesis net on ANE/GPU. One clean engine entry/exit, no mid-graph ping-pong.
2. **Match MRT2's reality first:** start on the **GPU** (`.cpuAndGPU`) via the architecture Google actually ships; treat ANE as an optimization experiment.
3. **Fuse/remove weight_norm** (and any BN) before tracing.
4. **Fixed per-frame shapes**, traced with `torch.jit.trace`.
5. **Channels-first `(B,C,1,W)`/`(B,C,F,T)` with time as the last axis;** minimize reshape/transpose/concat on the hot path.
6. **Carry streaming state** (conv lookback + iSTFT overlap) via Core ML `MLState` (iOS 18+) or host code; numerically verify against one-shot decode.
7. **Pin compute units explicitly** and A/B them; never trust `.all` blindly.
8. **Verify FP16 audio quality** (SNR, LSD, listening) vs fp32; guard the output with `tanh`/clamp; keep sensitive tail in fp32 if needed.
9. **Pre-warm the model**; measure **p99** per-frame latency, not mean.
10. **Keep per-frame working set ≪ ~32 MB;** don't batch frames on the ANE.
11. **Prove residency** with Xcode's per-op report + `powermetrics --samplers ane_power`.

## Worst Practices (consolidated)

1. Leaving `nn.Embedding`/`gather` in the Core ML graph between two ANE sections.
2. Defaulting to `.all` and assuming Core ML will "do the right thing."
3. Tracing with weight_norm still attached (conversion fails) or with dynamic shapes (ANE fallback).
4. Treating SpectroStream as a 1D time-domain ConvTranspose vocoder; copying DAC/HiFi-GAN ANE recipes wholesale.
5. Running the iSTFT on the ANE / ignoring its overlap-add state.
6. Decoding each 40 ms frame statelessly (zero-padded history) → 25 Hz boundary buzz.
7. Shipping fp16 without an audio-quality regression test.
8. Benchmarking mean latency on warmed-up synthetic buffers and ignoring p99 + cold-start + power-state jitter under real streaming.
9. Putting a singleton axis last in the tensor layout.
10. Batching frames on the ANE and blowing the SRAM working set.

---

## Quick-Reference Checklists

### Implementation checklist
- [ ] Detokenization split to CPU (or `one_hot @ codebook` matmul), not in-graph gather
- [ ] `remove_weight_norm` on every conv/convtranspose before trace
- [ ] Fixed per-frame input shape; `torch.jit.trace`
- [ ] Channels-first layout, **time axis last**; reshapes folded into CPU step
- [ ] Conv lookback state + iSTFT overlap state carried across frames (MLState or host)
- [ ] iSTFT on CPU/Accelerate, not ANE
- [ ] Final `tanh`/clamp for fp16 output safety
- [ ] Compute units pinned (`.cpuAndGPU` default; `.cpuAndNeuralEngine` for ANE port)
- [ ] `minimum_deployment_target = iOS18` if using stateful models
- [ ] Per-frame working set ≪ ~32 MB; no frame batching on ANE

### Debugging checklist
- [ ] Xcode Core ML Performance Report: per-op ANE/GPU/CPU dispatch checked
- [ ] `powermetrics --samplers ane_power` shows non-zero during decode (if targeting ANE)
- [ ] p50 **and p99** per-frame latency measured under real streaming cadence; both < 40 ms
- [ ] Model pre-warmed; cold-start compile time noted
- [ ] Gather confirmed as a deliberate CPU island (not mid-graph ping-pong)
- [ ] FP16 vs fp32 SNR / log-spectral distance computed; quiet passages & transients inspected
- [ ] One-shot vs frame-by-frame decode parity (no boundary spike at 1920-sample marks)
- [ ] `.all` vs `.cpuAndGPU` vs `.cpuOnly` A/B'd; placement understood
- [ ] Output parity vs `.cpuOnly` (fp16 rounding sanity)

---

## Recommendations — staged, with decision thresholds

**Stage 0 — Reproduce the reference on GPU first (days).** Get SpectroStream decode working in PyTorch (or via MRT2's MLX path) and capture an **fp32 CPU reference** for a fixed token→audio test set. Build your SNR / log-spectral-distance / boundary-click harness now; it gates every later decision. *Threshold to proceed:* you can decode a long clip frame-by-frame and match one-shot decode to ~numerical noise.

**Stage 1 — Ship on the GPU via `.cpuAndGPU` (the pragmatic baseline).** This mirrors what Google actually shipped (MLX-on-GPU). Detokenize on CPU/host, conv net + (ideally) iSTFT-as-conv on GPU, manage streaming state in host code. *Threshold to proceed to Stage 2:* GPU decode meets p99 < 40 ms **and** you have a concrete reason to want lower power or to free the GPU for the LM (e.g., the LM is GPU-bound and you're missing the frame budget under contention).

**Stage 2 — ANE port as an optimization experiment.** Convert the **conv synthesis net only** (gather + iSTFT stay on CPU) with `.cpuAndNeuralEngine`, fixed shapes, weight_norm fused, channels-first/time-last layout, stateful conv buffers. Prove residency (Xcode per-op report; non-zero `ane_power`). *Threshold to keep the ANE version:* it must (a) hold p99 < 40 ms, (b) pass the fp16 audio-quality bar from Stage 0, and (c) deliver a measurable power or GPU-contention win over Stage 1. If the gather/iSTFT splits cause ANE↔CPU ping-pong that erases the win, **abandon the ANE port and stay on GPU** — that is a legitimate, defensible outcome here.

**Stage 3 — Harden.** Pre-warm at launch; lock the 25 Hz cadence to keep the engine warm; add the output `tanh`/clamp; keep the final conv + iSTFT in fp32 if fp16 SNR is marginal; wire the audio-quality and latency checks into CI as regression gates.

**Signals that should change the plan:** if fp16 SNR is unacceptable even with an fp32 tail → the codec may need int8/fp16 mixed precision or stays on GPU/CPU. If per-frame working set approaches ~32 MB → you're mis-shaping 2D feature maps; fix layout before blaming the hardware. If MRT2's open C++/MLX engine proves fast enough on your minimum target device (M3 Pro/M2 Max for Base, any Apple Silicon for Small) → **don't build a Core ML port at all; embed their engine.**

---

## Caveats

- **No public SpectroStream Core ML/ANE port exists.** Every "decoder-on-ANE" recommendation here is engineering inference from the verified SpectroStream architecture combined with verified ANE behavior — not a reproduced, benchmarked result. The first person to actually convert it will discover specifics this guide cannot predict.
- **Google ships this model on the GPU, not the ANE.** That is the single most important real-world data point: the people who built MRT2 chose MLX-on-GPU for Apple Silicon. An ANE port may still win on power, but you are deliberately departing from the proven path.
- **The ~32 MB SRAM cliff is a single-source, estimated, M4-specific figure** (Orion paper relaying a maderix reverse-engineering benchmark), not an Apple spec, and is a soft cache-like cliff. Don't architect around it as a hard limit.
- **Code snippets are illustrative and unverified.** The streaming-conv lookback bookkeeping and the `ct.StateType` conversion sketch must be validated numerically (one-shot vs streaming parity) against your actual modules before shipping. The `one_hot @ codebook` detokenizer may reintroduce a fallback via `one_hot`; profile it.
- **MRT2 RVQ depth differs from full SpectroStream.** The generative path uses the first 12 (MRT2) / 16 (MRT v1) RVQ levels, not all 64. Your decoder must handle the depth your LM actually emits; the lookup-and-sum structure is unchanged, only the number of summed vectors differs.
- **All concrete ANE numbers (TOPS, SRAM, layout penalties) are M4/M4-Max-era.** Older A-series/M1–M3 ANEs may have different SRAM sizes, throughput, and op support. Re-verify on your minimum target device.

---

## Source-Quality & Uncertainty Notes

**Verified from primary sources (SpectroStream / MRT / MRT2):**
- SpectroStream is 48 kHz stereo, 25 Hz frames, 64 RVQ codebooks × 1024 entries (10-bit), 16 kbps, embedding dim 256, decoder conv depth 64, ~36M decoder params. (arXiv 2508.05207; 64-quantizer / 25 Hz figures re-confirmed in arXiv 2508.04651)
- SpectroStream decoder is **2D-convolutional, spectrogram-domain, with transposed-conv upsampling and a final inverse STFT**; causal convs + 1-embedding look-ahead (~80 ms latency); **weight-normalized** convs; **ELU** activations; delayed-fusion/early-splitting for stereo. (arXiv 2508.05207 §1–2)
- SpectroStream authors claim real-time streaming "on a single desktop CPU, without needing specialized accelerators." (§1, verbatim)
- MRT2: 40 ms frames, ~200 ms control latency, generative path uses first 12 RVQ levels (3 kbps), ships a **C++/MLX engine on Apple Silicon GPU** (not ANE), with host-managed model state/buffering; RT streaming needs M3 Pro/M2 Max+ (Base) or any Apple Silicon (Small). (magenta.withgoogle.com/magenta-realtime-2 and /mrt2, June 2026; arXiv 2508.04651)

**Verified ANE behavior (Apple + strong practitioner sources):**
- ANE prefers 4D channels-first `(B,C,1,S)`; last axis must be contiguous & 64-byte aligned; singleton last axis → 32× (fp16) / 64× (8-bit) memory blowup; reshape/transpose trigger copies. (Apple ML Research, "Deploying Transformers on the Apple Neural Engine," verbatim)
- ANE computes in fp16 (±65504); small values flush toward 0. (Hollemans neural-engine; Orion arXiv 2603.06728)
- Core ML stateful models (`MLState`, iOS 18+) for in-place caches; `register_buffer`→state. (coremltools docs; Apple WWDC24 "Bring your ML and AI models to Apple silicon")
- coremltools fails on `_weight_norm`; remove before tracing. (coremltools GitHub #1347)
- Detection tooling: Xcode per-op perf report; `-[_ANEModel program]` breakpoint; `H11ANEServicesThread`; `powermetrics --samplers ane_power`. (Hollemans; ss64/Apple man pages; Eclectic Light)
- Dynamic shapes degrade/forbid ANE placement; fixed shapes preferred. (whisper.cpp #548/#566; ONNX Runtime CoreML EP docs)
- Upsample (NN/bilinear) op reported to run on ANE; ResizeBilinear historically did not. (Hollemans; machinethink "Upsampling in Core ML")

**Extrapolated from analogous codecs (EnCodec / DAC / SoundStream / Mimi) — NOT verified for SpectroStream:**
- Streaming causal-conv state via lookback cache (EnCodec `trim_right_ratio`; ACIDS-IRCAM `cached_conv`); EnCodec/Mimi are time-domain SEANet decoders with 1D transposed convs (SpectroStream is spectrogram-domain — so the *mechanism* transfers, the *topology* differs). Mimi/Moshi demonstrate fully-causal 80 ms-frame streaming codecs with on-device (MLX) inference — proof the streaming-codec pattern works on Apple Silicon.
- Checkerboard-artifact mitigation (kernel divisible by stride; resize+conv; pixel-shuffle ICNR init) — Odena et al. (distill.pub); Dolby neural-upsampling-artifacts study (confirms tonal artifacts from transposed/sub-pixel convs in audio). General to transposed convs; applicability depends on SpectroStream's exact kernels.
- Snake activation considerations apply to DAC/BigVGAN, not SpectroStream (which uses ELU).
- Gather-falls-to-CPU and large-vocab-head-falls-to-CPU patterns drawn from Core ML LLM deployments (whisper.cpp, CoreML-LLM, Orion Table 5) and applied by analogy to RVQ detokenization.

**The ~32 MB SRAM cliff — single-source, estimated, M4-specific:**
- Stated in the Orion paper (arXiv 2603.06728) and originating from one reverse-engineering benchmark (maderix/manjeet singh, Feb 2026). **Not an Apple-published spec.** It is a soft, cache-like throughput cliff (~30% drop past the budget), inferred from a matmul scaling experiment, validated on M4/M4 Max only. Use as an order-of-magnitude design guide, not a hard limit.

**Unverified code patterns (validate before shipping):**
- All Swift/Python snippets are illustrative. The streaming-conv lookback slice bookkeeping in §4 **must** be numerically validated against a one-shot reference. The stateful-conversion `ct.StateType` sketch assumes one state per cached conv with names matching `named_buffers()`; confirm against your actual module. The `one_hot @ codebook` detokenizer may reintroduce a gather/scatter fallback via `one_hot` — profile it.
- No public Core ML/ANE port of SpectroStream exists to validate against; all decoder-on-ANE guidance here is engineering inference from the verified architecture + verified ANE behavior, not a reproduced result.
