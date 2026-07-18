# Validation receipts

This document is the evidence ledger for the Core ML port of Magenta RealTime 2
(`mrt2_small`) shipped in this repo and its companion Hugging Face model repo.
Every number below was measured, not estimated; each table names the receipt
file (mirrored in the HF repo's `validation/` folder) or the harness that
produced it. Where a result is partial or negative, it is labeled as such.

The working rule throughout: **do not infer Core ML or ANE behavior from
hope.** Every claim points to a command, a receipt file, or an explicit
failure.

## System paper — corrected composed runtime (2026-07-18)

The current end-to-end result is the evidence package under
`validation/results/system-paper/`. It supersedes the older composed-runtime
status in Sections 0 and 4 while retaining those sections as experiment
history. The paper is
`paper/main.pdf`; its claim contract is
`docs/plans/mrt2-system-paper-claims.md`.

### Placement and state (G1)

The selected temporal boundary is one fixed-shape pure Core ML function with
48 ordinary K/V inputs and 48 one-token K/V updates. Swift owns a preallocated
41-frame ring. A 64-step fixture crosses the ring boundary by 23 predictions.

- A17 Pro and A14 each pass 10/10 cold app processes with the same artifact
  hashes, temporal ANE cost weight at least 0.95, zero temporal GPU plan cost,
  fresh/warmed divergence, and the 64-step state proof.
- The A17 process trace records 653 temporal and 24 decoder ANE predictions;
  the app's Metal GPU interval table is empty. Depth is deliberate CPU work.
- Public receipts:
  `validation/results/system-paper/{a17pro,a14}/placement/g1-{manifest,report}.json`.

The supported language is **GPU-free**, not **all-ANE**.

### Sustained runtime and matched policy control (G2/G4)

| Device / policy | Window | p50 / p90 / p99 (ms/frame) | Production | Underruns | Reservoir outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| A17 Pro, selected ANE policy | 610.19 s | 20.26 / 20.81 / 21.66 | 1.0308x | 0 | grows 2.88 → 21.47 s |
| A17 Pro, matched `.cpuAndGPU` temporal policy | 610.03 s | 23.56 / 43.55 / 49.23 | 1.0081x | 0 | banks near 21 s, then drains to 7.65 s |
| A14, selected ANE policy + FLOAT32 decoder | 610.04 s | 43.88 / 47.25 / 58.59 | 0.8967x | 7,952 | maximum 20.16 s reservoir first underflows at 294.08 s |
| A14, matched `.cpuAndGPU` temporal policy | 610.03 s | 50.85 / 53.46 / 64.14 | 0.7721x | 25,655 | 2.88 s reservoir first underflows after 13.18 s |

The selected A17 run begins in `serious` thermal state and keeps per-minute
p99 within 21.21–23.08 ms. The matched CPU/GPU-policy control begins `fair`,
reaches `serious` after 40.12 s, crosses the 40 ms p99 deadline after minute
six, and drains its banked reservoir. A final zero-underrun counter therefore
does not substitute for the compute clock.

The A14 result is a **bounded-reservoir failure**, not a real-time tier. Its
FP16 decoder also has two retained device-specific numerical failures; the
final run uses the certified 136 MB FLOAT32 decoder (weight SHA-256 prefix
`38cbdf5c`).

Receipts:

- `validation/results/system-paper/a17pro/soak/g2-{manifest,report}.json`
- `validation/results/system-paper/a14/soak/g4-{manifest,report}.json`
- `validation/results/system-paper/evaluation/{a17pro,a14}-cpugpu-control-soak.json`
- `validation/results/system-paper/evaluation/evaluation-manifest.json`
  (five independent processes for each of four device/policy cells, including
  startup and per-stage median/IQR)

### Long-horizon audio quality (G3): failed, retained

The corrected 600 s A17 WAV is finite 48 kHz stereo and has zero associated
runtime underruns/drops. It passes clipped-sample ratio (`4.05e-6`), normalized
chunk-boundary jump (`0.0061`), L/R correlation (`0.9863`), prompt adherence
(`0.3119`), reference similarity (`0.8541`), and rejection of all six frozen
known-bad controls. It fails two independent predeclared requirements:

- last-window 4–16 Hz envelope-pulse share is `0.1364` against `≤0.070`;
- only 2/5 calibrated blind votes accept the candidate, although all five
  votes correctly accept the reference and reject/rank the click control.

A temperature-0.5 full-run intervention lowers pulse share to `0.0417` but
fails clipping, stereo, prompt, and reference-similarity bands. The gate is not
retuned. The supported conclusion is sustained inference and delivery, not
arbitrary-horizon musical validity. Receipt:
`validation/results/system-paper/audio/g3-{manifest,report}.json`.

### Compression ladder: negative result

Int8 linear quantization and 6-/4-bit palettization reduce temporal and depth
package bytes to roughly 50%, 38%, and 25% of baseline. All six artifacts stay
finite and all six fail deterministic parity; early stopping prevents device
timing or listening. Receipt:
`validation/results/MRT2WeightCompressionLadder.{json,md}`.

### Corrected-pipeline power comparison: unsupported, omitted

The first corrected-bundle Power Profiler pair is invalid because the signed
bundle omitted `warm.bin` and the app exited during preflight. The bundle was
repaired and regression-tested, but subsequent Instruments attempts could not
attach to either phone over USB. The invalid traces are retained privately and
excluded. This evidence package therefore makes no corrected-pipeline energy,
impact-score, battery-life, or producer-duty-cycle claim. The matched 610 s
temporal-policy control above is the supported comparative evidence.

## 0. Correction — the corrected artifact generation (paper §6.3–6.5)

**The ledger in sections 1+ below documents the earlier, superseded artifact
generation.** The shipped findings in *Surgical Inference* (§6.3–6.5) correct
three of its conclusions. Read this first; see `MODELS.md` for the corrected
artifact table and receipts.

- **Temporal / stateful KV (contradicts §1, §5.2 below).** In-graph *state
  mutation* — not attention math — is the ANE admission cliff. The stateless
  host-owned-cache boundary (`exporters/convert_temporal_body_carry.py`,
  `TemporalBodyCoreMLCarryWrapper`: 48 K/V caches as inputs, 48 one-token
  updates as outputs, no `ct.StateType`) compiles the full 12-layer stack to a
  single ANE-resident graph — on iPhone 12 Pro, `MLComputePlan`
  `preferredCounts=ane:1033,cpu:2`, `costWeights=ane:1.000`, p99 **14.991 ms**,
  beating CPU-only and GPU. Every `ct.StateType` variant fails ANE compilation:
  the 25-frame stateful graph reproduces `MILCompilerForANE … ANECCompile()
  FAILED`, Core ML **error −14**, on both phones and under both
  `.cpuAndNeuralEngine` and `.all`. Core ML vs MLX temporal correlation
  0.999975 (25-frame) / 0.999984 (carry). **Honesty note (§6.7):** admission is
  *instance-fragile* — an artifact that compiled to the ANE in a test harness
  later fell back to CPU in the shipping app — so the shipped temporal placement
  is `.cpuAndGPU` today, with this stateless graph as the documented escape.

- **Decoder FP16 (contradicts §3 below: "do not re-export at fp16").** The
  overflow conclusion is superseded. A channels-first (NCHW) internal rewrite
  plus an exact-in-FP32 mid-network rescale (`apply_fp16_safe_rescale`) makes the
  FP16 decoder finite **and** ANE-resident. On iPhone 12 Pro,
  `.cpuAndNeuralEngine` (ANE cost 1.000) yields finite output — 30,720/30,720 at
  5-frame (p99 **6.65 ms**), 184,320/184,320 at 25-frame (p99 **24.77 ms**) —
  while **CPU-only and CPU+GPU produce non-finite output from the same FP16
  artifact.** The plain channels-last FP16 export is non-finite everywhere
  (finite ratio 0.71). The ANE was the only compute unit that produced finite
  FP16 output. FP32 NCHW parity vs MLX: SNR 118.85 dB (§3's receipt is the NCHW
  FP32 wrapper parity, which — because the rescale is exact in FP32 — also
  validates the FP16 transform).

- **Depth FP32 / host sampling (contradicts §depth below).** Superseded by the
  in-graph FP16 rollout (`exporters/convert_depth_body_rollout.py`): all 12 RVQ
  levels sampled in one prediction from host-supplied Gumbel noise, because
  per-call cost ≈ weight bytes ÷ DRAM bandwidth on every compute unit (§6.5).
  FLOAT32 rollout is token-for-token exact (**0/900**;
  `validation/results/MRT2DepthBodyRollout_f32_validation.*`); FP16 flips fp16
  near-tie tokens without changing the distribution
  (`validation/results/MRT2DepthBodyRollout_f16_validation.*`) and ships at
  12.7 ms/frame (A14) / 8.4 ms (A17 Pro).

**End-to-end status.** This historical paragraph is superseded by the system-
paper section above. The corrected A17 compute and delivery clocks now pass;
the separately frozen 600 s generative-quality clock fails and remains an
explicit result.

---

## 1. Headline results

| Claim | Number | Evidence |
| --- | --- | --- |
| Temporal body numerical parity (Core ML vs MLX) | correlation `0.999985904188`, max abs error `0.1178550720` | `validation/MRT2TemporalBody_validation.{json,md}` |
| Temporal → depth composed parity (Core ML vs MLX) | correlation `0.999998250871`, `0 / 12` deterministic sampled-token mismatches | `validation/MRT2TemporalBody_validation.{json,md}` |
| Decoder numerical parity (Core ML vs MLX) | SNR `118.850 dB`, log-spectral distance `0.000722 dB` | `validation/SpectroStreamDecoder_validation.{json,md}` |
| Decoder conversion-source baseline (PyTorch vs MLX) | SNR `119.701 dB` | `validation/SpectroStreamDecoder_validation.{json,md}` |
| Temporal ANE residency (iPhone 15 Pro Max) | ~70% of estimated cost preferred on ANE (`MLComputePlan` cost weights `ane:0.704, cpu:0.296`) | device probe, Section 4 |
| Temporal-only device latency (iPhone 15 Pro Max) | p99 `13.943 ms`/frame over 100 stateful predictions | device probe, Section 4 |
| Composed pipeline device latency (iPhone 15 Pro Max) | `24–62 ms`/frame p99 depending on configuration and run length — **does not yet hold p99 < 40 ms in all configurations** | device probe, Section 4 |
| Sustained audio (iPhone 15 Pro Max) | 10-minute foreground run, `0` render underruns, `0` dropped frames — achieved **with lookahead buffering**, not per-frame real-time margin | device probe, Section 4 |
| FP16 decoder export | **rejected**: ~15.7% non-finite outputs (overflow) | Section 3 |
| Multi-frame temporal unrolls and host-cache K/V variants | **rejected on device**: ANE compiler cliff, Core ML error `-14`, silent CPU fallback | Section 5 |

The 40 ms number used throughout is the real-time budget: MRT2 streams at
25 Hz, one frame per 40 ms.

## 2. Numerical parity

### 2.1 Methodology

All parity claims share one harness design:

- **Teacher-forced fixtures.** A reference run of the exported MLX function
  (`mrt2_small.mlxfn`, generated by `mrt mlx export` from the public
  `google/magenta-realtime-2` checkpoint) records 25 frames of conditioning
  tensors, per-frame RVQ tokens, and reference PCM. Each Core ML stage is then
  fed the *reference* inputs for each frame, so errors measure the stage under
  test rather than accumulated autoregressive divergence.
- **Deterministic argmax sampling.** Token-level comparisons apply per-level
  valid-range masking followed by deterministic argmax to both pipelines, so a
  "mismatch" means the logits actually crossed, not that two RNGs disagreed.
- **Zero-cache negative control.** State continuity is easy to fake; a
  fresh-state prediction must therefore *fail* to match. On a two-frame
  unrolled sibling of the same temporal export, frame 1 evaluated with an
  empty K/V cache scored max abs error `10.8432359695` and correlation
  `0.8180290774` vs MLX, while the same frame with carried state scored
  correlation `0.9999834761318928`. The match depends on prior-slot temporal
  state, not on the current input alone.

### 2.2 Temporal body — `MRT2TemporalBody.mlpackage`

FP16 `mlprogram`, 12-layer temporal transformer, one 40 ms frame per
prediction, 48 FP16 `ct.StateType` K/V buffers (`[1, 41, 8, 128]`).

Receipt: `validation/MRT2TemporalBody_validation.{json,md}`.

| Boundary | Max abs error | Mean abs error | Correlation |
| --- | ---: | ---: | ---: |
| Core ML vs MLX, temporal output | 0.1178550720 | 0.0197259826 | 0.999985904188 |
| Core ML temporal → depth body (FLOAT32), full logits | 0.0652890205 | 0.0112627821 | 0.999998250871 |

Deterministic sampled-token mismatches through the composed temporal → depth →
argmax path: **0 / 12**.

The max error of ~0.118 on the raw hidden state is FP16 state/math drift; the
composed check is the one that matters, because tokens — not hidden states —
are what the decoder consumes. Two corroborating results from the same
harness family:

- A 25-frame teacher-forced loop with the FLOAT32 depth body sampled
  **0 / 300** tokens differently from MLX.
- Depth body at FLOAT16 instead mismatched 16 / 300 tokens (5.33%),
  concentrated in the finer RVQ levels (4, 6, 7, 8, 9, 10, 11) where stereo
  detail lives. This is why `MRT2DepthBody.mlpackage` ships FLOAT32: at
  FLOAT32 the mismatch count is 0 / 300 with max error ~1e-4.

### 2.3 Decoder — `SpectroStreamDecoder.mlpackage`

FLOAT32 `mlprogram`, NCHW-parallel conv layout, host RVQ embeddings
`[1, 25, 256]` → pre-iSTFT tensor `[1, 96, 480, 4]`. RVQ codebook gather
(`SpectroStreamRVQCodebooks.f32.bin`) and iSTFT/overlap-add stay on the host
CPU by design.

Receipt: `validation/SpectroStreamDecoder_validation.{json,md}`.

| Boundary | Max abs error | Mean abs error | SNR | Log-spectral distance |
| --- | ---: | ---: | ---: | ---: |
| PyTorch vs MLX (conversion source) | 0.0032958984 | 0.0000067176 | 119.701 dB | 0.001008 dB |
| Core ML vs MLX | 0.0030517578 | 0.0000076265 | 118.850 dB | 0.000722 dB |

The host boundaries around the Core ML graph were validated separately:

- **RVQ lookup:** host CPU codebook gather matches the MLX
  `codes_to_embeddings` path with max abs error `0.0000000000` and
  correlation `1.0`.
- **iSTFT/overlap-add:** the host C++ iSTFT against the MLX iSTFT on the same
  `decoder_stft` tensor measures SNR 109–117 dB, correlation 1.000000, zero
  lag, identical L/R correlation (0.9461 = 0.9461). The full host path
  (FLOAT32 Core ML decoder + host iSTFT) vs the MLX reference on identical
  input measures SNR 104.68 / 107.31 dB (L/R) — perceptually bit-exact.

## 3. The FP16 decoder failure (read this before re-exporting anything)

The SpectroStream decoder conv stack **overflows FP16**. A FLOAT16 export of
the same graph converted and compiled cleanly, passed a latency smoke test,
and produced garbage:

- `finite_ratio = 0.843` — roughly **15.7% of the pre-iSTFT output values
  were NaN/Inf**, with `correlation = NaN` and `snr_db = NaN` in the export's
  own validation report. The PyTorch-vs-MLX source comparison in the same
  report was ~119 dB, so the checkpoint and port were fine; FP16 alone
  destroyed the output.
- Because precision is a property of the graph, the corruption appeared on
  **every prompt**: captured audio measured peak 10.84 vs the reference's
  0.92, crest factor ~84x vs ~15x, energy below 500 Hz 1.4% vs 73%, and L/R
  correlation −0.06 vs +0.92 — structurally broken, not merely distorted.
- Other FP16 decoder variants failed the same way (finite ratios `0.663954`,
  `0.706380`, `0.741875` across layouts and prefix lengths). The overflow
  lives in the large upsampling tail, not the early conv prefix.

Re-exporting at FLOAT32 fixed it completely: `finite_ratio = 1.0`,
correlation `0.99999999994`, SNR `118.85 dB` — the receipt shipped with this
repo. The FLOAT32 decoder does not need the per-frame path: it runs roughly
once per ~1 s chunk (~180 ms on device), off the 40 ms-per-frame critical
path.

**The lesson, stated bluntly:** correlation and latency smoke tests are not
validation. The failing export's receipt already contained
`finite_ratio=0.843` when it was first treated as healthy. Gate every
precision decision on `finite_ratio == 1.0` *and* correlation/SNR. The same
class of bug hit the FP16 depth body more subtly (5.33% corrupted tokens, no
NaNs at all) — which is why both `MRT2DepthBody` and `SpectroStreamDecoder`
ship FLOAT32 while only the temporal body, which was validated faithful at
FP16 (Section 2.2), ships FLOAT16.

## 4. Device evidence — iPhone 15 Pro Max (iPhone16,2, iOS 26.5)

All device numbers come from a minimal on-device probe app that loads compiled
`.mlmodelc` artifacts, runs warmups separately from timed iterations, and
reports p50/p90/p99 per model and compute-unit selection.

### 4.1 ANE residency of the temporal body

Measured two independent ways:

- **`MLComputePlan`** (queried on device for the loaded model): with
  `.cpuAndNeuralEngine`, the temporal body plans `697` ops preferred on ANE
  vs `579` on CPU, with estimated cost weights `ane:0.704, cpu:0.296` —
  about **70% of the model's estimated cost on the ANE**, stable across
  repeated runs. The CPU remainder is concentrated in attention bookkeeping
  (`matmul`, `transpose`, `concat`, `slice_update`, `read_state`); ANE cost
  is dominated by `linear` and `matmul`.
- **Instruments (Core ML template) on a composed run:** the exported
  ANE hardware-interval table contains real per-prediction ANE intervals for
  the temporal, depth, and decoder models, and the exported Metal GPU
  interval table contains no rows for the app process — runtime evidence,
  stronger than the static plan.

### 4.2 Latency

| Configuration | p99 | Notes |
| --- | ---: | --- |
| Temporal body alone, `.cpuAndNeuralEngine`, 1 prediction | 14.148 ms | single stateful frame |
| Temporal body alone, 100 predictions | 13.943 ms | stable ANE island |
| Temporal + depth (25-frame loop) | 25.548 ms/frame | no decoder |
| Temporal + depth + decoder + audio accounting | 28.831–31.181 ms/frame | simulator path |
| Live audio engine (short run) | 24.333 ms/frame | `AVAudioSourceNode`, 0 render underruns |
| Live audio engine (60 s run) | 62.453 ms/frame | 0 render underruns |

Honest framing: the temporal body alone holds **p99 ≈ 14 ms/frame**, well
inside the 40 ms budget. The composed pipeline measures **24–62 ms/frame at
p99 depending on configuration and run length**, and therefore **does not yet
hold p99 < 40 ms in all configurations** — longer live runs drift above the
budget. The system still plays gapless audio because the producer runs ahead
of the renderer through a lock-free SPSC ring with primed lookahead; the p99
budget is recovered by buffer depth, not per-frame margin.

### 4.3 Sustained audio

- **First live render-thread smoke:** `0` underrun events across `6,487`
  render callbacks while the Core ML pipeline ran (drop counters were high by
  design — the unthrottled smoke generates as fast as possible into a finite
  ring).
- **10-minute proof:** a 600.053 s foreground run of the full real-PCM
  pipeline — temporal + depth + true RVQ codebook lookup + Core ML decoder +
  host iSTFT/overlap-add + SPSC render delivery — completed with **0 render
  underruns and 0 dropped frames**, using 50 primed chunks (~2 s) of
  lookahead and persistent cross-chunk decoder lookahead carry. The temporal
  stage in that particular run was the CPU/GPU burst variant (Section 5);
  the published 1-frame stateful temporal has its own zero-underrun live
  proofs at 60 s.
- **Thermal:** the 10-minute run reached and held iOS thermal state
  `serious`. That is not a nominal-thermal claim; it is evidence against
  critical thermal collapse during the run. Shorter (≤60 s) composed runs on
  the ANE-routed configuration stayed `nominal`/`fair`. Sustained thermal
  headroom remains unfinished work.

For context, an older-device control (iPhone 12 Pro, A14) failed a
zero-reservoir 10-minute run with 1,150 underruns at `nominal` thermal —
a latency-headroom failure, not thermal — and then passed 10 minutes with
0 underruns after prefilling a 15 s PCM reservoir. Older devices are viable
as a higher-startup-latency tier.

### 4.4 GPU absence and power

The strongest placement evidence is what is *absent*. Instruments Core ML
traces of composed all-ANE runs — on both iPhone 12 Pro and iPhone 15 Pro
Max — export Metal GPU interval tables whose only rows belong to
`backboardd`, the iOS screen compositor. App-attributed GPU time: **0 ns**.

A counterbalanced Power Profiler pair (same app, same 60 s reservoir-backed
live-audio run, temporal routed to ANE vs routed to GPU as the control) on
iPhone 12 Pro measured:

| Metric (60 s live run, iPhone 12 Pro) | temporal on ANE | temporal on GPU (control) |
| --- | ---: | ---: |
| Process GPU impact (duration-weighted) | 0.000 | 2.231 |
| Process CPU impact (duration-weighted) | 1.380 | 2.625 |
| CPU instructions | 48,123,948,704 | 110,255,108,500 |
| Thermal state | nominal | nominal |

A separate timing analysis of the same ANE-vs-GPU routing pair reported a
producer active-work duty cycle of **0.57 for the ANE routing vs 0.93 for
the GPU routing**: the ANE route finishes each second of audio early and
sleeps ~43% of the wall clock, while the GPU route runs nearly pegged. These
are process-attributed impact scores and duty cycles, not calibrated joules —
strong directional evidence, not a battery-life claim.

Scope note: these traces ran an all-ANE runtime configuration in which every
Core ML stage requested `.cpuAndNeuralEngine`. In the published bundle, the
fp32 decoder is the deliberate exception: fp32 cannot execute on the ANE
(it is fp16 hardware), so the decoder schedules on GPU — one 25-frame call
per second of audio, off the 40 ms critical path, traded for the 118.85 dB
decode (Section 3).

## 5. Negative results: why the published temporal export is one frame

The obvious export — unroll many frames into one prediction, or carry the K/V
cache as ordinary inputs/outputs — failed on device in instructive ways.
These results are the reason `MRT2TemporalBody` is a 1-frame stateful graph.

### 5.1 The ANE compiler cliff at 2 frames

Stateful unrolled exports at frame counts 1, 2, 4, 8, 16, 25 all convert and
compile on the Mac (`coremlcompiler` passes; conversion time grows from
20.5 s at 1 frame to 671.0 s at 25 frames). On the iPhone 15 Pro Max with
`.cpuAndNeuralEngine`:

| Frames | Device result |
| ---: | --- |
| 1 | **pass** — p99 22.292 ms first run, 13.943 ms warmed |
| 2 | fail — `MILCompilerForANE ... ANECCompile() FAILED`, Core ML error `-14` |
| 4, 8, 16, 25 | same failure |

The cliff is identical on iPhone 12 Pro. For the 25-frame artifact, every
escape route was also red: `.all` fails with the same ANE compile error,
`.cpuOnly` fails BNNS execution-plan compilation with
`mmap ... Cannot allocate memory` (also error `-14`), and `.cpuAndGPU`
technically runs but measures **7,621.233 ms for one prediction** after a
163.6 s load — two orders of magnitude over budget.

### 5.2 Host-owned K/V "carry" graphs silently fall back

Replacing Core ML mutable state with host-owned cache tensors (48 inputs +
48 update outputs) avoids the hard compile failure but not the placement
failure. On device, the 2-frame carry graph under `.cpuAndNeuralEngine`
printed `ANECCompile() FAILED`, then **completed anyway on a fallback path**
— `MLComputePlan` showed `cpu:2263` preferred ops and no ANE cost at all. The
smallest realistic history-specific burst behaved the same (CPU-only plan,
p99 55.394 ms). This is the textbook silent-fallback trap: the prediction
succeeds, the latency looks plausible, and nothing tells you the ANE is idle
unless you ask the compute plan or Instruments.

The carry family does run acceptably on CPU/GPU (composed 23–35 ms/frame at
p99 in burst form), and it powered the 10-minute zero-underrun proof in
Section 4.3 — but it is a CPU/GPU result, not an ANE result, and in a full
integration the same fallback measured ~640 ms of temporal compute per
25-frame generation block, which cannot sustain 25 Hz without minutes of
buffering. Paired Power Profiler captures confirmed the direction: routing
temporal work onto the ANE path removed essentially all process-attributed
GPU impact and roughly halved process CPU impact versus the CPU/GPU control.

### 5.3 What the cliff is, and is not

Layer-by-layer falsification on device showed the cliff is **not** attention
math, cache reads, cache-update outputs, or output count — stateless
attention+FFN stacks and host-cache stacks up to all 12 layers each map
cleanly to ANE in isolation (`costWeights ane:1.000`). The failures
correlate with large stateful multi-frame graphs hitting the ANE/BNNS
execution-plan builders. Practical guidance for anyone porting a similar
model: keep the stateful graph to one frame, keep dynamic bookkeeping on the
host, and verify placement with `MLComputePlan` *and* an Instruments Core ML
trace — never trust a successful `.cpuAndNeuralEngine` prediction alone.

## 6. Reproducing the validation

### 6.1 Fixture-only (no MLX, no checkpoint download)

The repo ships the teacher-forced reference fixtures, so the headline
temporal parity numbers can be re-derived against the published
`MRT2TemporalBody.mlpackage` on any Apple-silicon Mac:

```bash
python validation/validate_temporal_body.py \
  --skip-pytorch \
  --reference-npz fixtures/reference_temporal_unrolled.npz
```

This replays the recorded MLX reference tensors
(`fixtures/reference_temporal_unrolled.npz`,
`fixtures/generated_tokens_unique.npy`) through the Core ML package and
reports max/mean error, correlation, and deterministic sampled-token
mismatches. The output should match
`validation/MRT2TemporalBody_validation.{json,md}`.

### 6.2 Full-stack independent verification

To regenerate the reference itself rather than trusting the shipped
fixtures:

1. Download the public checkpoint:
   `huggingface_hub.hf_hub_download(repo_id="google/magenta-realtime-2",
   filename="checkpoints/mrt2_small.safetensors", ...)` (1.128 GB; no token
   required).
2. Export the MLX function: `mrt mlx export --output-name mrt2_small
   --model mrt2_small --checkpoint mrt2_small.safetensors`.
3. Regenerate fixtures with `validation/generate_reference_fixtures.py`,
   then run `validation/validate_temporal_body.py` without `--skip-pytorch`
   to compare PyTorch, MLX, and Core ML three ways.

The decoder receipt (`validation/SpectroStreamDecoder_validation.{json,md}`)
is produced the same way by the decoder exporter's validation pass; check
`finite_ratio` first, then SNR/LSD, per Section 3.
