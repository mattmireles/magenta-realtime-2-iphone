# Throughput Is Not Liveness: Three Clocks for GPU-Free Music Generation on iPhone

Matt Mireles, Independent Researcher

## Abstract

Real-time generative audio is usually reduced to one question: can inference
run faster than playback? That criterion is necessary and insufficient. A
system may average faster than real time while buffering tens of seconds of
stale audio, and it may deliver every sample while a state-boundary error
changes the sound. We present an end-to-end deployment and falsification study
of the 230-million-parameter `mrt2_small` live-music model on iPhone. The system
combines a 12-layer temporal transformer, a 12-level residual-vector-quantized
depth rollout, SpectroStream decoding, inverse STFT, and 48 kHz stereo
delivery. A pure fixed-shape temporal graph executes on the Apple Neural
Engine (ANE), while Swift owns its 41-frame K/V ring. Ten of ten cold processes
admit the graph on both A17 Pro and A14, and a model-attributed trace contains
temporal and decoder ANE work with no application-attributed GPU interval.

On an iPhone 15 Pro Max, the corrected runtime sustains 610 seconds in the
foreground, reaches the `serious` thermal state, completes 628 nominal
one-second producer iterations in 609.65 seconds, and records zero underruns or
drops. The p99 of iteration-normalized active stage cost is 24.81 ms per token
against a 40 ms throughput budget. We emphasize what this number is not: it is
the p99 of 25-token iteration means, not per-token tail latency. Backpressure
bounds the queue near 21 seconds, so the experiment proves continuous
throughput, not low-latency steering. Separate device evidence measures
controller application in 4-15 ms but audible changes only after 6.48-9.02
seconds without queue discard.

The main scientific result comes from a reviewer-motivated 2x2 crossover. For
three seeds, we generate 600-second token trajectories with MLX and the shipped
Core ML port, then decode each through stateful MLX or the stateless phone
boundary. Fixed-token graph and DSP controls separate sampling, numerical,
windowing, and reconstruction hypotheses. The pulse excess follows stateless
decoder windowing in all 60 paired 30-second windows: median seed effect
+0.01706, compared with +0.00018 for the Core ML graph itself. A direct tensor
probe shows that the original 25-token calls discard causal convolution
history; retaining 12 token frames raises correlation with stateful MLX from
0.108 to above 0.999999999. Adding that context reduces pulse share by 0.01650
and changes the original diagnostic-limit count from 35/60 windows to 13/60,
versus 11/60 for stateful MLX. A new 600-second iPhone capture matches the
stateful reference count at 5/20 windows for the principal seed, with no
dropout. Thus a result first framed as long-horizon model degeneration was a
systems boundary bug. The broader lesson is methodological: live generation
has separate compute, delivery, and generative clocks, and a credible study
must localize failures across all three.

## 1. Introduction

Live music generation is an unusually unforgiving systems workload. Its
autoregressive model must keep producing, its audio callback must never wait,
steering should become audible while the intent is still relevant, and the
trajectory must remain musically plausible. A batch generator can finish late.
A live generator that finishes late clicks or falls silent; one that buffers
too aggressively plays an obsolete decision; one with incorrect state can run
perfectly on time while sounding wrong.

Magenta RealTime introduced continuously steerable causal music generation
[Pasini et al., 2025]. Magenta RealTime 2 (MRT2) provides open weights and an
MLX implementation for Apple Silicon Macs [Google DeepMind, 2026]. Its small
configuration emits 12 residual-vector-quantized (RVQ) codes at 25 Hz, and a
SpectroStream decoder reconstructs 48 kHz stereo audio [Google, 2026]. This
paper asks whether the complete loop can execute continuously on an iPhone
without application GPU work, and, more importantly, what evidence is needed
to make that statement mean anything.

Our first answer was wrong in an instructive way. A signed A17 Pro runtime
sustained 610 seconds with zero underruns and a growing reservoir. Its captured
audio nevertheless crossed a predeclared 4-16 Hz envelope-pulse threshold and
failed three of five exploratory automated listening votes. We reported a
failed "generative clock." That framing survived many engineering controls but
not the decisive one: we had never run the reference sampler to the same
horizon and crossed token source with decoder implementation.

We therefore freeze three causal hypotheses. H1 is model-intrinsic or
sampling-induced recurrence: the pulse should follow tokens into either
decoder. H2 is numerical divergence in the temporal/depth port: the Core ML
token source should be worse after decoding with clean MLX. H3 is a decoder or
DSP boundary error: the pulse should follow the stateless phone decoding path
for either fixed token source. We then split H3 into an MLX FLOAT32 pre-iSTFT
graph, the Core ML FP16 graph, the production C++ reconstruction core, and a
measured context intervention. The result strongly supports a refined H3: a
stateless causal decoder was invoked with insufficient left context.

This correction changes the paper's thesis. "Faster than playback" is not a
complete real-time claim, and a failed audio detector is not a causal label.
We define three clocks:

1. **Compute clock.** Sustained production rate must be at least playback rate.
   Batch-normalized stage costs describe headroom but do not become per-token
   tail latency by division.
2. **Delivery clock.** The callback must observe zero underruns and producer
   drops, while queued audio remains inside a band whose upper bound is chosen
   from an audible steering-latency budget.
3. **Generative clock.** Prompt-conditional audio measures and controlled
   comparisons must remain valid across the horizon. A detector can identify a
   symptom; only an intervention or crossover can identify its source.

We make four contributions:

- A complete GPU-free placement and sustained-throughput study of MRT2 on A17
  Pro, with a pure temporal graph, host-owned K/V state, in-process admission
  gates, and model-attributed accelerator evidence.
- A duration-controlled thermal result: an A17 CPU+GPU-policy control has
  38.51 ms p99 iteration-normalized cost in five short processes, but 49.23 ms
  over a matched 610-second run and crosses 40 ms only after minute six.
- A three-seed, long-horizon token-by-decoder crossover that localizes a
  plausible generative failure to missing causal decoder history, plus a
  12-frame overlap intervention verified at tensor, audio, and physical-device
  levels.
- A corrected liveness contract that distinguishes continuous playback from
  interactive steering. The current runtime sustains output, but its ordinary
  6.48-9.02 second audible control latency is not yet a strong live-product
  result.

The weights are unchanged. Previous conversion and accelerator-admission
findings appear in the companion *Surgical Inference* manuscript [Mireles,
2026]. Here the units of contribution are the composed runtime, sustained
physical-device evidence, causal audio localization, and measurement contract.

## 2. Background and related work

### 2.1 Live music and neural audio

MusicGen generates text- or melody-conditioned music from interleaved codec
tokens [Copet et al., 2023]. RAVE demonstrated 48 kHz neural synthesis faster
than real time on a laptop CPU [Caillon and Esling, 2021]. Live Music Models
instead make causality and continuous steering part of the product contract
[Pasini et al., 2025]. MRT2 is the open model used here. Its SpectroStream
codec operates in the time-frequency domain and supports full-band stereo
[Pasini et al., 2025b], extending the streamable neural-codec lineage of
SoundStream and EnCodec [Zeghidour et al., 2021] [Defossez et al., 2022].

Autoregressive generation can amplify model or decoding errors over a long
horizon [Rohatgi et al., 2025]. Repetitive text degeneration motivated dynamic
nucleus sampling [Holtzman et al., 2020]. These results make recurrence a
credible hypothesis for a periodic audio symptom, not a conclusion. In our
case, both MLX and Core ML trajectories retain full distinct-frame ratio, about
6.2-6.4 bits of mean per-level entropy in late complete windows, and no exact
period-1 through period-8 cycles. More decisively, the symptom follows decoder
windowing under fixed tokens.

### 2.2 Mobile heterogeneous inference

MLPerf Mobile emphasizes that on-device performance is a property of the full
software-hardware stack [Reddi et al., 2022]. Apple documents fixed-shape
transformer deployment patterns for the ANE [Apple, 2022; Apple, 2026]. Similar
heterogeneous scheduling appears in mobile NPU work such as `llm.npu` [Xu et
al., 2025]. MRT2 differs from mobile LLM prefill: it repeats a small causal
step indefinitely, feeds an audio callback, and exposes every state or timing
error to a listener. Requested compute units, graph conversion, and a short
latency benchmark are therefore weak proxies. We separately measure admission,
accelerator activity, sustained capacity, callback continuity, queue depth,
and audio behavior.

## 3. System design

### 3.1 Workload and host boundary

At 25 Hz, one MRT2 token frame represents 1,920 output samples and 40 ms of
48 kHz stereo audio. The temporal transformer has 12 layers and consumes the
previous frame's mean token embedding, source conditioning, and a local
attention history. The depth transformer autoregressively selects 12 RVQ
codes. Their 256-dimensional codebook rows are summed and passed to the
SpectroStream decoder. [Fig. system] shows the deployed boundary.

The temporal graph is a pure single-frame tensor function. Inputs contain the
current temporal and source vectors, a fixed-shape validity bias, and 48
ordinary K/V arrays for self- and cross-attention across 12 layers. Outputs
contain one temporal vector and 48 one-token cache updates. Swift owns a
preallocated 41-frame chronological ring. This removes in-graph state mutation,
which had compiled in a probe but failed ANE admission in the application.

The state proof asks two independent questions. First, identical input on fresh
and warmed state must diverge, preventing a write-only-state false pass.
Second, 64 consecutive predictions must match a frozen reference, including 23
after the ring wraps. Correlation is 0.9995 on both phones, and all outputs are
finite. The runtime repeats this proof and queries the temporal compute plan at
session start.

### 3.2 One-call depth rollout

The depth transformer has dependent RVQ levels: level `k+1` consumes the code
selected at level `k`. Twelve separate Core ML predictions on A14 cost 1,004 ms
per 25-frame producer iteration, or 40.2 ms per token frame. A one-prediction
graph unrolls the dependency, takes host-generated Gumbel noise, applies the
fixed top-40 mask, gathers embeddings, and returns all 12 codes plus temporal
feedback.

The measured ablation corrects an easy but false bandwidth story. The in-graph
rollout does not turn 12 dependent transformer applications into one weight
traversal. FLOAT32 still costs about 37 ms per frame. The large gain comes from
the validated FLOAT16 artifact, which measures 12.7 ms on A14 and 8.4 ms on
A17 Pro ([Table depth]). FLOAT32 is token-exact in 900 comparisons; FLOAT16
flips near-tie tokens and was accepted only after distributional and device
audio gates.

### 3.3 Decoder context and reconstruction

The original phone decoder accepted 25 token embeddings and emitted 96 STFT
frames. The host advanced by 24 token frames, preserving only the one-frame
SpectroStream lookahead. This treated the decoder as a finite-window function.
Its convolutional stack is causal but stateful: output near a boundary depends
on more history than the formal lookahead.

The corrected boundary retains 12 token frames. The first 25-token prediction
initializes reconstruction state and emits its full 96 STFT frames. Each later
prediction advances by 12 tokens, drops the first 48 context STFT frames, and
emits the 48 new frames. The Core ML input shape and model artifact remain
unchanged. A preallocated FLOAT32 crop buffer materializes logical output using
the declared `MLMultiArray` shape and strides; allocation and DSP remain off the
audio callback.

The C++ inverse STFT uses the periodic Hann synthesis window used by upstream
SpectroStream and performs overlap-add into a lock-free single-producer,
single-consumer PCM ring. During diagnosis we found a second parity debt: the
old C++ core applied a normalized dual-window convention. Direct sample tests
now match the upstream periodic-Hann reconstruction. However, this change alone
*increases* pulse share by a median 0.00321 across seeds. It is a real parity
fix but not the causal fix for the reported symptom.

### 3.4 Buffering and steering

Inference and reconstruction run on a producer task. The Core Audio callback
only reads a bounded ring and zero-fills an underflow. A startup reservoir
absorbs jitter. Once the ring reaches its high watermark, producer
backpressure prevents unbounded memory growth.

That architecture guarantees neither low latency nor liveness. A high
watermark of roughly 21 seconds allows 21 seconds of already-rendered intent.
In an earlier two-event control test, the controller applied new state after
0.015 and 0.004 seconds, but matched-reference audible changes arrived after
6.479 and 9.023 seconds. A proof-mode queue discard retained 19,200 frames
(0.4 seconds), faded in over 3,840 frames, discarded 647,936 frames, and ended
with zero underruns and drops. The capture point was before the playback ring,
so that receipt cannot prove click-free speaker output. We therefore report
continuous buffered playback as solved and subsecond audible steering as open.

### 3.5 Ten-second temporal refresh

All principal device and crossover trajectories reset temporal K/V and the
previous-frame feedback vector every 10 seconds. Decoder context, overlap-add,
and queued PCM remain continuous. This condition must be stated because the
41-frame ring covers only 1.64 seconds. After 10 seconds, attention slots have
already rolled over several times; the intervention's distinctive long-horizon
effect is resetting the recurrent feedback loop and returning the cache to a
cold start. It is a deployed bounded-trajectory protocol here, not evidence
that an unrefreshed model is stable.

## 4. Experimental method

### 4.1 Devices, artifacts, and placement

The primary device is a 2023 iPhone 15 Pro Max (`iPhone16,2`, A17 Pro). The
boundary device is a 2020 iPhone 12 Pro (`iPhone13,3`, A14). Signed release
builds run in the foreground with the screen on. The fixed condition is `warm
ambient texture`, certified `warm.bin` source conditioning, top-k 40, style
guidance 3.1, temperature 1.0, and seed 20260718 unless a replication seed is
named.

Placement has two receipts. Ten cold application processes query
`MLComputePlan`; all ten assign temporal estimated cost 1.0 to the ANE on each
phone and pass the state proof. A process-targeted Instruments trace then
records temporal and decoder ANE predictions, Core ML activity for every model
family, and an empty application Metal GPU table. The claim is GPU-free, not
all-ANE: depth, Gumbel generation, RVQ lookup, cache mutation, inverse STFT,
ring management, logging, and UI work remain on the CPU.

### 4.2 Throughput and the batch-normalized cost

One producer iteration generates 25 temporal/depth frames and one to three
decoder calls. We preserve the historical diagnostic

`(temporal + depth + sampling + decoder) / 25`.

We rename it **iteration-normalized compute cost**. Its p99 is the p99 of a
mean over each 25-token iteration. It suppresses within-iteration dispersion
and cannot certify a per-token deadline. The actual compute-clock verdict is
sustained nominal generated duration divided by producer wall time, combined
with the delivery counters and ring trajectory. We still report the normalized
quantiles because they decompose thermal changes and preserve comparison with
the earlier campaign.

The final corrected A17 run lasts 610 seconds after audio start, captures 600
seconds of PCM, logs every producer iteration, and records token frames after
warmup. A separate five-process campaign provides run-level medians and IQRs
for temporal ANE and CPU+GPU requested policies. A matched 610-second
CPU+GPU-policy run exposes duration effects.

### 4.3 Crossover design

For seeds 20260718, 271828, and 1618033, both the MLX sampler and an exact
Python implementation of the shipped Core ML graph/host boundary generate
15,001 token frames, enough for 600 seconds after decoder lookahead. Each uses
its native seeded random-number implementation; token-source arms are therefore
independent trajectories, not token-by-token parity trials. Within every
decoder comparison, however, the token file is fixed and hash-identical.

The principal 2x2 crosses token source (MLX or Core ML port) with decoder path
(stateful streaming MLX or stateless Core ML plus C++ DSP). A fifth arm replaces
only the Core ML decoder graph with the MLX FLOAT32 pre-iSTFT graph while
preserving 25-token windows and C++ reconstruction. Four more arms cross both
token sources and decoder graphs with the corrected periodic-Hann DSP. The
causal intervention then repeats MLX and Core ML graphs with 12 frames of left
context and identical corrected DSP.

For each 30-second window we measure the fraction of RMS-envelope spectral
power from 4 to 16 Hz. The original 0.070 limit was frozen for one ambient
prompt before this study. We retain it as a diagnostic count, not a universal
music-quality boundary: legitimate rhythm occupies the same band. Primary
inference uses paired effect sizes over all windows and reports the median and
range of seed-level means. We do not treat 60 temporally adjacent windows as 60
independent samples.

### 4.4 Independent evidence and receipts

The decoder-context tensor probe compares a stateful MLX prefix against
stateless 25-token evaluations at one fixed segment while varying retained
history over 0, 1, 2, 4, 8, and 12 tokens. The production C++ reconstruction
has an independent sample-level parity test against a NumPy periodic-Hann
inverse STFT and overlap-add implementation.

Every public result is generated from a machine-readable receipt. Raw device
WAVs, events, tokens, and traces remain private because the product runtime is
private, but their SHA-256 hashes, normalized summaries, commands, model
identities, and source hashes are public. Precision experiments stop in the
order finite, deterministic parity, device latency, then audio; a wrong
function never earns a favorable speed number.

## 5. Results

### 5.1 Reliable ANE admission and GPU absence

Both devices pass 10/10 cold-process temporal admission and state gates. The
A17 trace contains 653 temporal ANE predictions and 24 decoder ANE predictions,
plus expected CPU depth activity. The application's Metal GPU table contains
zero intervals and zero duration. The A14 cross-device trace reaches the same
placement verdict. Moving recurrence to ordinary tensor I/O therefore makes
admission repeatable in the application without deleting transformer math.

### 5.2 Sustained A17 throughput, with the metric named honestly

The corrected A17 context-12 runtime completes 628 nominal one-second producer
iterations over 609.65 seconds of producer-loop wall time, a 1.030x nominal
rate. It reaches `serious` thermal state and records zero callback underruns,
zero producer drops, and no near-zero dropout longer than one sample. The PCM
capture is exactly 600.0 seconds, finite, unclipped, and stereo. [Table sustain]
summarizes the physical-device results.

Iteration-normalized stage cost is 23.50 ms p50, 24.33 ms p90, and 24.81 ms
p99. Decoder context roughly doubles decoder cadence: decoder work rises to
58.74 ms per 25-token iteration at p50, while temporal and depth remain 280.01
and 246.14 ms. Backpressure occupies 38.9% of producer-loop wall time, direct
evidence that sustained capacity exceeds playback.

The delivery statement is narrower. Queue fill starts at 2.88 seconds, never
falls below 2.73 seconds, and ends at 21.01 seconds, near the configured high
watermark. That proves safe continuous playback for the measured run. It also
quantifies why the default runtime is not a low-latency steering system.

### 5.3 Duration changes the placement verdict

The five-process pre-context campaign reports A17 median run-level p99
iteration-normalized cost of 22.35 ms for temporal ANE and 38.51 ms for the
CPU+GPU requested-policy control ([Table dispersion], [Fig. latency]). In the
matched 610-second control, the latter becomes 49.23 ms. Its per-minute p99
crosses 40 ms only after minute six and reaches 51.46 ms, while a banked
reservoir drains from about 21 seconds to 7.65 seconds ([Fig. soak]). A
five-minute benchmark would have selected a policy that fails the ten-minute
capacity criterion. The selected ANE trajectory begins hotter yet remains
flat.

These pre-context controls do not estimate the exact cost of the repaired
decoder cadence; the final row above does. They isolate a separate temporal
placement and duration effect using identical historical decoder work.

### 5.4 Crossover localizes the audio defect

The crossover result is stable across all three seeds ([Table crossover],
[Fig. crossover]). Replacing MLX tokens with Core ML-port tokens at the
stateful MLX decoder has median seed effect +0.00181, with range -0.00057 to
+0.00627. Replacing the FLOAT32 MLX decoder graph with the FP16 Core ML graph
under identical stateless windows and DSP has effect +0.00018 [0.00016,
0.00088]. Neither is large enough to explain the symptom.

Stateless windowing and the legacy production DSP, by contrast, add 0.01706
[0.01613, 0.01796] to pulse share relative to streaming MLX, and the paired
difference is positive in all 60 windows. The FLOAT32 MLX graph reproduces the
effect, exonerating FP16 as its primary source. Correcting the Hann convention
without context adds another 0.00321 rather than removing it.

The context probe identifies the missing state. With no left history, a
stateless 25-token evaluation has correlation 0.1083 with the corresponding
stateful pre-iSTFT tensor and maximum absolute error 1,071.7. Correlation is
0.9926 with one token, 0.9999955 with eight, and above 0.999999999 with 12;
maximum error falls to 0.000319. Under the same graph and corrected DSP,
adding 12-token context reduces audio pulse share by 0.01650 [-0.01772,
-0.01574], negative in all 60 paired windows.

The original 0.070 diagnostic fires in 35/60 stateless Core ML windows, 13/60
context-corrected windows, and 11/60 stateful MLX windows. On the physical A17
capture, median window share is 0.06091 and 5/20 windows cross the limit -
exactly the stateful MLX count for seed 20260718. The final window is 0.09181,
which reinforces why a single tail threshold is not a universal clock. The
full trajectory contains no short exact token cycle, no sustained dropout, and
no monotone decoder-specific late-onset failure.

The evidence therefore rejects the paper's earlier simple claim of
runtime-independent long-horizon recurrence. It does not prove that MRT2 can
never degenerate, or that every prompt is high quality. It proves that the
measured pulse excess attributed to the phone path is caused primarily by an
incorrect causal decoder boundary and is repaired by restoring measured
history.

### 5.5 A14 is below the sustained boundary

The A14 story remains a negative lower bound. Before the context repair, its
certified FLOAT32 decoder configuration measured 43.88, 47.25, and 58.59 ms
iteration-normalized p50/p90/p99, produced at 0.897x, and exhausted a 20.16
second reservoir after 294.08 seconds. It recorded 7,952 underruns. Context
repair increases decoder cadence, so it cannot rescue this capacity result.

The A17 FP16 decoder artifact is not promoted on A14. A short ANE probe on A14
is finite but has unsafe amplitude; a long retry later becomes non-finite,
while FLOAT32 CPU isolation remains bounded. We did not localize the exact
overflowing layer, so we report a device-artifact numerical incompatibility,
not a layer-level mechanism. A14 needs a different trained precision or
decoder/depth architecture, not a larger hidden buffer.

### 5.6 Compression ladder and early stopping

We also test int8 linear quantization and 6-bit and 4-bit palettization on
temporal and depth packages ([Fig. compression]). Temporal packages shrink to
50.2%, 37.6%, and 25.2% of baseline, but 64-step correlation falls to 0.9973,
0.9770, and 0.8874, below the 0.999 gate. Depth argmax mismatch reaches 46.3%,
68.7%, and 94.0%. All remain finite; none reaches device timing. This is a
negative result about tested post-training methods, not a claim that trained
low-precision MRT2 is impossible.

## 6. Discussion

### 6.1 Why the crossover changes the science

The first manuscript had disciplined receipts and the wrong causal unit. It
compared a 600-second phone capture with short clean MLX excerpts, then named a
failed generative clock. The crossover forces every explanation to predict a
path through fixed tokens. Because the FLOAT32 MLX pre-iSTFT graph reproduces
the phone effect and 12-frame context cancels it under both graphs, the result
is stronger than either "the model degenerates" or "Core ML is numerically
different." It is a specific state-contract error with a minimal intervention.

This pattern generalizes. Causal neural decoders may expose a formal lookahead
that is not their complete left-receptive-field contract. Exporting a static
window without explicit state turns missing history into deterministic boundary
modulation. Output finiteness, tensor shape parity, seam-jump metrics, and even
short perceptual clips can all pass. A useful export test must compare a
stateless window with a stateful prefix at multiple history depths.

### 6.2 Throughput is not frame latency

Dividing a batch time by 25 produces a unit of ms/token, not 25 observations.
Its quantiles describe iteration-level compute density. They average away
within-batch stalls and should not be labeled p99 frame latency. Our corrected
compute clock uses the variable the architecture actually needs: sustained
audio production relative to playback. Per-token tail latency would require
instrumenting individual temporal/depth steps, which the current production
log does not do.

This correction does not weaken the thermal result. The reservoir trajectory,
producer rate, and late CPU+GPU-policy slowdown are batch-appropriate evidence.
It does narrow the headline: we demonstrate sustained generation throughput,
not a frame-synchronous 40 ms scheduler.

### 6.3 Continuous playback is not interactive liveness

A growing or high-watermark-limited reservoir prevents underflow and delays
intent. The old delivery criterion accepted any final queue larger than the
initial queue. For a continuously steerable model, that condition rewards the
wrong behavior. A defensible upper bound is derived from prompt-change-to-
audible-change latency, not memory capacity.

The discard experiment offers an architectural direction: retain 0.4 seconds,
apply a short fade, and count intentional discard separately from producer
drops. But it has not passed a human speaker-level click-free test. The current
system is therefore a robust buffered playback engine and a candidate live
engine. Calling both simply "real time" would erase the most important product
difference.

### 6.4 Evidence paths are systems too

The project exposed two evidence-path bugs. An early recorder flattened a
padded FP16 `MLMultiArray`, manufacturing huge coefficients that live playback
never consumed. The decoder's reconstruction core then used a different Hann
convention from upstream. The first bug invalidated a capture; the second was a
real parity debt but not the observed root cause. Both are now covered by
independent stride-aware and sample-level tests.

The methodological rule is simple: test the measuring instrument with known
good and known bad controls, then cross implementations before assigning a
failure to the model. Hashes and manifests preserve evidence; they cannot make
an underdetermined experiment causal.

## 7. Limitations

The crossover uses three seeds and one ambient prompt. That is enough to
replicate the decoder-boundary effect, not to characterize the prompt
distribution or long-horizon musical quality of MRT2. The 4-16 Hz measure is
prompt-sensitive and overlaps legitimate musical rhythm; its 0.070 value is
reported only as the original diagnostic boundary. We do not use the prior
five automated listening votes as independent statistical evidence: two late
excerpts overlap, and an audio model is not a substitute for counterbalanced
human listening.

The MLX and Core ML token generators use different native random-number
schemes. Token-source arms therefore test distributions and downstream
symptoms, not token-exact port equivalence. Temporal fixtures and FLOAT32 depth
rollout provide separate deterministic parity evidence, but rare FP16
near-tie flips remain expected. The context tensor probe uses one segment;
three-seed full-audio interventions provide the broader replication.

The final corrected runtime has one 610-second physical-device run. The
placement campaign, pre-context latency campaign, and A14 study reuse prior
signed builds with identical model artifacts; the context repair is host-only.
The A14 FP16 failure is not localized to a layer. We report Instruments impact
and placement evidence but no calibrated joules or battery-life claim.

Finally, the default 21-second queue is incompatible with a strong live-
steering claim. The 0.4-second discard path needs counterbalanced human
speaker-level evaluation. The complete Crossfade runtime and raw audio remain
private; public artifacts contain exporters, validators, normalized receipts,
hashes, figure sources, and manuscript source. The runtime is available from
the author for artifact review.

## 8. Conclusion

MRT2 can sustain GPU-free generative-audio throughput on an A17 Pro iPhone
under a ten-minute foreground thermal load. The corrected system produces at
1.030x nominal real time with zero underruns or drops, while a pure temporal
boundary makes ANE admission repeatable. Those are substantial systems
results, but they are not synonymous with per-token tail latency or low-latency
steering.

The strongest result came from trying to falsify our own negative claim. A
three-seed token-by-decoder crossover showed that an apparent long-horizon
generative failure followed stateless decoder windowing, not the model, token
source, FP16 graph, or Hann reconstruction. Twelve frames of measured causal
history recover tensor parity and remove the excess on Mac and iPhone. Live
generation has three clocks. Good science requires measuring each one and,
when a clock fails, crossing the boundary until the cause has nowhere left to
hide.

## Artifact statement

The public repository contains Core ML exporters, frozen fixtures, placement
and sustain verifiers, the three 600-second crossover reports, decoder-context
probe, corrected A17 normalized manifest, compression ladder, figure sources,
paper source, and exact build protocol. Public reports hash private WAVs,
tokens, events, traces, signed executables, and relevant runtime sources.
Preconverted artifacts are mirrored on Hugging Face. The product application
and raw listening media are available from the author for artifact review.

## Acknowledgments

OpenAI Codex assisted with experimental code, data normalization, figure
generation, and manuscript preparation. The author designed the study,
operated the devices, inspected the evidence, selected the claims, and accepts
responsibility for the manuscript.

## Reproducibility

### Public evidence map

The canonical crossover result is `aggregate.json` in the public crossover
receipt directory. It hashes three seed reports and the decoder-context probe.
Each seed report hashes token summaries and every decoded WAV arm. The
corrected physical-device receipt is `context12-soak-manifest.json` in the A17
context-12 directory;
it hashes the 600-second PCM capture, 610-second event trace, 15,790-frame token
capture, normalized summaries, signed executable, runtime source, and render
core source.

The A17 and A14 placement directories contain admission receipts. The
historical A17 placement soak and CPU+GPU duration control are under
`a17pro/soak/` and `evaluation/`. The A14 bounded-reservoir result is
under `a14/soak/`. `MRT2WeightCompressionLadder.json` contains all six
early-stopped compression arms, and `depth-rollout-ablation.json` records the
measured 12-call, one-call FLOAT32, and one-call FLOAT16 depth results.

### Crossover commands

The private Crossfade harness exposes generation, decoding, summarization, and
decoder-context probe subcommands. A complete replication uses
15,001 codebook-local token frames, temperature 1.0, top-k 40, a 10-second
refresh, and a 600-second decode. Decoder arms name the token file explicitly;
the analyzer refuses incomplete fixed-DSP or context pairs.

The public analysis commands are:

- run `analyze_system_paper_crossover.py` once per seed with the
  four 2x2 WAVs, FLOAT32 graph split, four corrected-DSP controls, and two
  context arms;
- run `aggregate_system_paper_crossover.py` on the three reports
  plus `decoder-context-probe.json`; and
- regenerate `fig-crossover.pdf` from the aggregate only.

### Corrected runtime protocol

1. Load the signed precompiled models with explicit compute policies, query the
   temporal compute plan, execute the fresh/warmed state proof, and perform
   three untimed warmups.
2. Prime the PCM ring. For each token frame, present chronological temporal K/V,
   run one temporal prediction, sample 12 depth levels in one prediction, and
   sum selected RVQ rows.
3. Once 25 decoder embeddings are available, predict 96 STFT frames. After the
   first window, retain 12 token embeddings, advance by 12, discard 48 context
   STFT frames, and send only 48 new frames to the periodic-Hann inverse STFT.
4. Push PCM off the callback. The callback performs bounded ring reads; count
   zero-fill underruns. Backpressure at the high watermark. Every 10 seconds,
   clear temporal K/V and previous feedback without clearing decoder context,
   overlap-add state, or queued PCM.
5. Capture logical STFT-derived PCM and codebook-local tokens independently;
   stop after 610 seconds and hash every artifact before normalization.

### Verification suites

The public suites cover crossover audio analysis, seed aggregation, G1-G4
manifests, placement, sustain, and compression contracts. The private runtime
suites cover Swift configuration defaults, padded `MLMultiArray` reads, exact
SplitMix64 Gumbel values, cache updates, token-cycle metrics, C++ periodic-Hann
sample parity, and the signed iOS build. Tectonic rebuilds the paper from the
readable Markdown source, and Poppler-rendered pages are visually audited
before publication.
