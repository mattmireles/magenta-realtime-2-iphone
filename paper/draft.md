# The Three Clocks of Live Music Generation: Sustained GPU-Free MRT2 Inference on iPhone

Matt Mireles, Independent Researcher

## Abstract

A live generative-audio system has to satisfy three clocks at once: the model
must produce tokens before its autoregressive deadline, the renderer must
deliver samples without interruption, and the generated trajectory must remain
musically valid for as long as the system runs. Most inference benchmarks
measure only the first clock. We present an end-to-end deployment of the
230-million-parameter `mrt2_small` live-music model on an iPhone 15 Pro Max. A
single fixed-shape Core ML temporal graph executes all 12 transformer layers on
the Apple Neural Engine (ANE), while the host owns a 41-frame K/V ring; a
one-call 12-level RVQ rollout, SpectroStream decoder, inverse STFT, and
lock-free audio ring complete the 48 kHz stereo pipeline. Ten of ten cold
processes admitted the temporal graph to the ANE. A model-attributed Instruments
trace recorded 653 temporal and 24 decoder ANE predictions and no
application-attributed GPU interval.

The compute and delivery clocks pass under an unusually strict condition: a
610.19-second foreground run that began in the `serious` thermal state produced
audio at 1.031 times real time, with zero underruns or drops and a 21.66 ms
p99 effective cost against the 40 ms deadline. Per-minute p99 remained between
21.21 and 23.08 ms, and the audio reservoir grew rather than drained; a matched
CPU/GPU-policy control reaches 49.23 ms p99 and drains its banked reservoir
after minute six. The
generative clock does not pass. Although the 600-second capture remains finite,
stereo-coherent, prompt-aligned, and embedding-close to a clean reference, its
last-window 4-16 Hz envelope energy is 0.136 against a frozen 0.070 limit, and
only two of five blinded calibrated audio-model votes accept excerpts spanning
the run. A temperature reduction removes the envelope excess but fails four
other quality measures. We retain this as a result rather than tune the gate
away.

The older A14 device exposes the hardware boundary: even with a 136 MB FLOAT32
decoder chosen to avoid an FP16 numerical failure, p99 is 58.59 ms, production
is 0.897 times real time, and the maximum 20.16-second reservoir underflows
after 294.08 seconds. Six post-training weight-compression variants shrink
packages to 25-50% of baseline but all fail deterministic parity before device
testing. These results show both what is solved and what is not: GPU-free,
thermally sustained live-model inference on a phone is practical, but a
throughput pass is not evidence of sustained generative quality. We release the
exporters, fixtures, gate verifier, machine-readable receipts, and paper source.

## 1. Introduction

Live music generation is a systems problem with an audible failure mode. A
batch image model can finish late; a live model that misses a deadline clicks,
stutters, or falls silent. Magenta RealTime established the model class: a
causal generator steered continuously by text or audio rather than a prompt
that returns a fixed song [Pasini et al., 2025]. Magenta RealTime 2 (MRT2)
extends that system with open weights and an Apple-Silicon implementation. Its
small configuration contains 230 million parameters [Google, 2026] and emits 12 residual
vector-quantized codes at 25 Hz; SpectroStream reconstructs 48 kHz stereo
audio. Upstream targets Apple-Silicon Macs through MLX [Hannun et al., 2023]
and a C++ runtime. This
work asks a narrower, harder deployment question: what does it take to run the
complete loop continuously on a phone without using the GPU?

The obvious success criterion is one frame every 40 ms. It is necessary and
incomplete. A system can benchmark below 40 ms while silently executing on the
wrong processor. It can average faster than real time while p99 stalls drain a
finite buffer. It can deliver every sample on time while the autoregressive
trajectory collapses into periodic clicking. We therefore define three clocks:

1. **Compute clock.** The temporal, depth, sampling, and decoder stages must
   produce each 25 Hz frame with p99 below 40 ms, or the result must be labeled
   as a buffering tier.
2. **Delivery clock.** Produced PCM must remain ahead of the audio callback for
   the full run, with zero underruns and drops and a non-draining reservoir.
3. **Generative clock.** The audio itself must remain finite, stereo-coherent,
   prompt-aligned, free of periodic defects, and perceptually acceptable across
   the complete horizon.

This separation is the paper's central methodological claim. It prevented two
false conclusions in one experiment. First, an early 600-second WAV appeared
catastrophically broken even though live playback was correct; the evidence
recorder had flattened a padded `MLMultiArray` instead of following its
strides. Second, after that recorder was fixed, the runtime passed every
throughput and delivery requirement, but blind excerpts and a frozen envelope
metric found a genuine long-horizon audio defect. A single "real-time factor"
would have hidden both failures.

We make four contributions:

- **A complete GPU-free phone deployment of MRT2's compute path.** The
  temporal transformer's full 12-layer math and SpectroStream decoder are
  observed on the ANE; depth rollout, sampling entropy, RVQ lookup, buffering,
  and DSP remain deliberate CPU work. We do not call this "all ANE."
- **A simple streaming boundary that is reliable in the application process.**
  K/V tensors are ordinary model inputs and one-token updates are ordinary
  outputs. Swift owns a preallocated 41-frame chronological ring. The graph is
  pure fixed-shape math and is admitted in 10/10 cold processes on both A17 Pro
  and A14.
- **A sustained evaluation that keeps the clocks separate.** On A17 Pro the
  compute and delivery clocks pass for ten foreground minutes from a hot start;
  the generative-quality clock fails its predeclared gate. On A14 a maximum
  reservoir cannot hide the thermally sustained deficit.
- **Negative results with early stopping.** Three compression methods across
  temporal and depth packages remain finite and much smaller, yet all fail
  deterministic state/token gates. FP16 decoder placements fail differently on
  the two phones. No rejected artifact is promoted by a speed-only result.

Our intended scope is a systems study, not a claim about a new generative
model. The weights are unchanged. Previous conversion and ANE-admission
findings are described in the companion *Surgical Inference* manuscript [Mireles, 2026]; this
paper cites those methods and contributes the composed runtime, reliability
study, sustain envelope, audio-delivery evaluation, and the three-clock result.

## 2. Background and related work

### 2.1 Live and controllable music models

MusicGen [Copet et al., 2023] generates text- or melody-conditioned music from interleaved neural
codec tokens, but it is designed around bounded generation rather than a
perpetual interaction deadline. RAVE [Caillon and Esling, 2021] demonstrated high-quality 48 kHz neural
audio synthesis substantially faster than real time on a laptop CPU. Magenta
RealTime [Pasini et al., 2025] introduced "live music models" whose output is causal and whose style
can be steered during generation. MRT2 [Google DeepMind, 2026] supplies the open model and Mac runtime
used here. Our contribution is not a competing model; it is a physical-phone
system and a sustained evaluation contract.

### 2.2 Neural audio codecs

SoundStream [Zeghidour et al., 2021] established end-to-end residual-vector-quantized neural audio and
reported a streamable smartphone-CPU codec. EnCodec [Defossez et al., 2022] extended neural compression
to high-fidelity 48 kHz stereo. SpectroStream [Pasini et al., 2025b] moves full-band multichannel
reconstruction into the time-frequency domain and combines delayed channel
fusion with an RVQ representation. MRT2 generates the first 12 SpectroStream
codebooks at 25 frames/s. Our host sums the corresponding 256-dimensional
codebook rows, invokes the fixed-shape decoder, and performs inverse STFT and
overlap-add natively. Codec *decoding* in real time is therefore a necessary
subsystem, not evidence that the autoregressive generator meets its deadline.

### 2.3 Mobile heterogeneous inference

MLPerf Mobile [Reddi et al., 2022] showed why on-device performance must be evaluated as a complete
software-hardware stack rather than inferred from accelerator specifications.
Apple has published ANE-oriented transformer layouts and fixed-shape deployment
patterns through Core ML [Apple, 2022; Apple, 2026]. On Qualcomm devices,
`llm.npu` [Xu et al., 2025] similarly reconstructs
model boundaries and schedules work by hardware affinity. Our workload differs
from mobile LLM prefill: the batch is one temporal frame, the deadline repeats
forever, and the user hears tail latency directly. These constraints favor one
fixed graph per repeated stage, host-owned recurrence, and direct measurement
of p99, placement, thermal state, and buffer depth.

## 3. System design

### 3.1 Workload and deadline

`mrt2_small` contains a 12-layer temporal transformer and a 12-level
autoregressive depth transformer. At frame *t*, the temporal stack consumes the
previous frame's mean token embedding, source conditioning, and local attention
history. The depth stack produces one code from each of 12 codebooks, feeding
each selected embedding into the next level. The selected embeddings are
summed for SpectroStream. One generated frame represents 1,920 stereo samples,
so the exact deadline is 40 ms at 48 kHz.

The deployed pipeline is summarized in [Fig. system].

Temporal and decoder models are loaded with `.cpuAndNeuralEngine` on A17 Pro;
depth uses `.cpuOnly`. Sampling-noise generation, tensor bookkeeping, and
audio DSP also use the CPU. No runtime decision depends on an average latency.

### 3.2 Host-owned temporal state

The temporal artifact is a single-frame, pure tensor function. Its ordinary
inputs are one `[1,1,1024]` temporal vector, one `[1,1,256]` source vector, a
static attention bias, and 48 K/V arrays representing self- and cross-attention
state across 12 layers. It returns the temporal output and 48 one-token K/V
updates. The host preallocates all cache arrays and owns the 41-frame ring.

This boundary is intentionally dull. In-graph state mutation had compiled in a
probe and then failed ANE admission in the application. Moving mutation to the
host deletes that compiler variable without deleting attention math. The
runtime writes each update into the ring, presents chronological cache inputs
on the next frame, and advances a valid-history bias. There is one temporal
model, not 41 position-specialized packages.

The state proof has two parts. First, the same input on a fresh and warmed
runtime must diverge; otherwise state may be write-only. Second, 64 predictions
must match a frozen reference, including 23 predictions after the 41-frame ring
wraps. Core ML correlation is 0.9995 on both phones, finite ratio is 1.0, and
maximum absolute errors are 0.8725 on A17 Pro and 0.7635 on A14.

### 3.3 One-call depth rollout

Calling a 75 MB depth model 12 times per temporal frame repeatedly streams its
weights and cannot meet a phone deadline. The deployed depth graph unrolls the
12 dependent levels inside one prediction. The host supplies a `[12,1024]`
Gumbel-noise tensor and inverse temperature; the graph applies the fixed top-40
mask, selects codes, gathers the next embedding, and returns both 12 codes and
the mean temporal-feedback vector. Randomness remains seeded and host-owned,
while the large weight stream occurs once per frame.

### 3.4 Decoder and real-time delivery

The first 12 RVQ codebooks are stored as a 12.6 MB float table. A host lookup
sum forms each decoder embedding. The 68 MB FP16 decoder used on A17 Pro has
channels-first internal geometry and a channels-last public STFT output. The
output is materialized by following `MLMultiArray.shape` and `.strides`.

A producer thread invokes Core ML and pushes decoder STFT frames into a C++
render core. Inverse STFT and overlap-add occur off the audio callback. The
callback performs only bounded ring reads and zero-fills an underflow; it never
waits for inference. A finite startup reservoir absorbs ordinary jitter, but a
native-real-time pass additionally requires the final reservoir not to be
smaller than the initial one. This makes buffering visible rather than a way to
hide a slow producer.

### 3.5 Bounded trajectory refresh

Short corrected captures were clean while unbounded trajectories deteriorated
after roughly 90 seconds. We tested the smallest intervention: every 10 seconds
the host clears only temporal K/V and previous temporal feedback. Decoder
embeddings already queued, decoder overlap, and the audio ring remain intact.
This separates temporal trajectory history from delivery continuity. At 180
seconds the intervention passes all objective bands, including 0.926 embedding
similarity and 0.053 pulse share. At 600 seconds it does not prevent the
late-horizon clicking failure. We therefore report it as a bounded ablation,
not as a solution.

## 4. Experimental method

### 4.1 Devices and software

We evaluate a 2023 iPhone 15 Pro Max (`iPhone16,2`, A17 Pro, iOS 26.5.2) and a
2020 iPhone 12 Pro (`iPhone13,3`, A14, iOS 26.5). Runs use signed release
builds in the foreground with the screen on. The prompt is `warm ambient
texture`, source conditioning is a certified `warm.bin` vector, top-k is 40,
style guidance is 3.1, and the principal capture uses temperature 1.0 and seed
20260718. The runtime logs every generation iteration, compute policy, thermal
state, ring depth, pushed/pulled frames, underruns, drops, and stage time.

### 4.2 Effective frame metric

One generation-loop iteration produces 25 temporal/depth frames and one or
occasionally two decoder batches. Backpressure sleep is not model work. We
define effective frame time as

`(temporal + depth + sampling + decoder) / 25`.

We report run-level p50, p90, and p99. The 600-second gate requires at least
15,000 effective-frame samples. The dispersion campaign uses at least five
independent application processes per device/policy cell and reports the
median and IQR of run-level percentiles. Startup is measured from the
auto-start event to the first audible PCM after load, placement/state proof,
warmup, and priming.

### 4.3 Placement evidence

Requested compute units are metadata, not evidence. At session start,
`MLComputePlan` must assign at least 0.95 estimated temporal cost to the ANE
with zero GPU operations. Ten consecutive cold processes must preserve the
same model hashes, policy, and state proof. A process-targeted Instruments
trace must then contain temporal and decoder ANE intervals, Core ML intervals
for temporal/depth/decoder, and no Metal GPU interval attributed to the
application. This proves admission reliability and actual accelerator activity
as separate properties.

### 4.4 Audio integrity

The 600-second float WAV is produced by an independent render core from the
same logical STFT tensor used by playback. The gate checks 48 kHz stereo,
finite ratio 1.0, clipping ratio at most `1e-5`, maximum normalized decoder
chunk-boundary jump at most 0.07, left/right correlation at least 0.97, text
embedding adherence at least 0.30, embedding similarity to a clean reference
at least 0.80, and 4-16 Hz envelope-pulse power share at most 0.07.

Six frozen controls target known failure families: stride corruption, missing
temporal feedback, write-only state, a click-comb capture, channel collapse,
and injected dropouts. All must be rejected. Five 24-second lineups use unique
frozen order seeds, neutral labels, RMS matching, and shared peak attenuation.
Each contains the candidate, the clean MLX reference, and the known-bad click
control. A multimodal audio model's vote is valid only when it passes the
known-good clip, rejects the known-bad clip, and ranks those controls
correctly. The candidate requires at least four of five valid votes. These
votes are a calibrated secondary instrument, not human-subjective ground
truth.

### 4.5 Receipts and early stopping

Every public verdict is produced by a checked-in verifier from a
machine-readable manifest. Private runtime logs and traces are hashed into
those manifests. Precision experiments run in the order finite -> deterministic
parity -> device latency -> audio. A candidate that fails an earlier gate is
not installed or listened to. This avoids both wasted measurement and the
temptation to select a fast invalid artifact.

## 5. Results

### 5.1 Reliable ANE admission and GPU absence

Both devices pass 10/10 cold-process temporal plan/state gates. The A17 Pro
trace contains 653 temporal ANE predictions totaling 5.007 s and 24 decoder
ANE predictions totaling 0.341 s. It also contains 586 depth Core ML
predictions, as expected for the CPU stage. The application's Metal GPU table
is empty: zero intervals and zero duration. The A14 supporting trace likewise
contains temporal and decoder ANE intervals and an empty application GPU table
for the FP16 artifact set used in that placement experiment.

The claim is therefore **GPU-free**, not **all-ANE**. The CPU remains essential
for sampling-noise construction, the depth placement chosen for this runtime,
RVQ lookup, state mutation, iSTFT, ring management, logging, and UI work.

### 5.2 Cross-process dispersion and the CPU/GPU-policy control

Five independent application processes per device and policy separate the
selected ANE placement from a temporal `.cpuAndGPU` policy control; the prompt, seed, model
hashes, depth CPU policy, decoder ANE policy, warmup count, and buffer settings
are otherwise fixed. [Table dispersion] reports the median and IQR of each
run-level percentile rather than pooling frames across processes.

On A17 Pro, ANE lowers median run-level p50 from 27.32 to 20.77 ms and p99 from
38.51 to 22.35 ms. The temporal stage alone falls from 16.35 to 11.25 ms p50;
cold process startup to audible PCM falls from 6.22 to 4.26 seconds. On A14,
ANE lowers p50 from 49.73 to 40.38 ms and the temporal stage from 24.85 to
13.85 ms ([Fig. latency]). Its 50.72 ms p99 remains over deadline and is statistically
indistinguishable from the GPU control's 50.64 ms at this sample size. Thus the
control supports an A17 tail-latency win and an A14 central-latency win; it does
not turn A14 into a real-time device.

### 5.3 A17 Pro sustain: compute and delivery pass

The sustained device outcomes are compared in [Table sustain]. The final A17
run starts with the phone already in the `serious` thermal state
after preceding experiments. It measures 610.186 seconds after generation
starts and pulls 610.208 seconds of audio. The producer completes 629 one-second
iterations, or 1.0308 times real time. Effective frame time is 20.260 ms p50,
20.813 ms p90, and 21.660 ms p99. There are no underruns and no dropped frames.

The finite reservoir does not create this pass ([Fig. soak]). It begins at 138,240 frames
(2.88 s), ends at 1,030,656 frames (21.47 s), and has a positive slope of
1,462.5 frames/s. After the initial fill, the per-minute p99 stays in a narrow
21.21-23.08 ms range. The first minute includes reservoir accumulation and has
the largest p99; later minutes remain below 21.71 ms despite the sustained hot
state. The compute clock and delivery clock therefore pass independently.

The matched 610.03-second `.cpuAndGPU` temporal-policy control separates that
stability from mere buffered playback. It begins fair, reaches serious thermal
state after 40.12 seconds, and ends with zero underruns, but p99 is 49.23 ms and
production is only 1.008x. Its per-minute p99 crosses 40 ms after minute six
and reaches 51.46 ms; a reservoir banked near 21 seconds drains to 7.65 seconds
by the end. Thus its delivery clock happens to survive the finite experiment,
while its compute clock fails. The selected ANE run begins hotter yet keeps
tail latency flat, a stronger explanation than final underrun count alone.

### 5.4 A14 boundary: buffering cannot manufacture real time

The FP16 decoder configuration that works on A17 Pro is numerically unsafe on
A14. ANE placement first yields finite but unusable amplitude; a long CPU retry
later becomes logically non-finite. The final A14 experiment therefore uses the
certified 136 MB FLOAT32 decoder with `.cpuAndNeuralEngine`, the same temporal
ANE request, and CPU depth.

This configuration begins nominal, reaches `fair` at 51.08 s and `serious` at
106.06 s. Its effective frame time is 43.876 ms p50, 47.248 ms p90, and
58.586 ms p99. Production is 0.8967 times real time. The audio ring permits a
maximum observed start reservoir of 20.16 seconds; it underflows at 294.08
seconds, records 7,952 callback underruns, and produces 568.32 seconds of PCM
during the 610.04-second window. There are zero producer drops.

This is not an A14 real-time tier. It is a measured bounded-reservoir failure.
The distinction matters: a short benchmark or a larger but unbounded preload
could make the same system appear successful.

### 5.5 Generative clock: the long capture fails

The corrected temperature-1.0 A17 capture is 600.0 seconds, finite, 48 kHz
stereo, and free of render failure. It passes clipping (`4.05e-6`), normalized
chunk-boundary jump (0.0061), left/right correlation (0.9863), prompt adherence
(0.3119), and reference embedding similarity (0.8541). All six known-bad
controls are rejected.

It nevertheless fails the predeclared generative-quality gate in two
independent ways. The last 30 seconds have 4-16 Hz envelope-pulse share 0.1364,
nearly twice the 0.070 limit. All five blind votes are valid because every vote
passes the clean reference and rejects the click control, but the candidate
passes only two. The rejected excerpts begin at 151.68, 555.04, and 566.83
seconds; the calibrated judgments flag persistent periodic clicks. Accepted excerpts
begin at 33.44 and 361.96 seconds. The defect is therefore intermittent across
the trajectory, not a global recorder or render failure.

Two ablations prevent a facile explanation. A 10-second temporal-state refresh
passes all bands at 180 seconds but not at 600 seconds. Lowering temperature to
0.5 reduces pulse share to 0.0417, but clipping rises to `3.27e-5`, left/right
correlation falls to 0.9462, prompt adherence to 0.2684, and reference
similarity to 0.7838. We reject the lower-temperature path rather than select
the one metric it improves.

The two principal quality arms are compared in [Table audio]. This is the
principal negative result. ANE placement, p99 headroom, a growing
ring, and semantically plausible embeddings do not guarantee sustained musical
output. The system has solved inference and delivery, not long-horizon
generation quality.

### 5.6 Compression ladder: smaller is not equivalent

The complete compression ladder appears in [Fig. compression]. We test int8
linear quantization and 6- and 4-bit palettization separately on
the 365.7 MB temporal and 74.5 MB depth packages. Temporal candidates shrink to
50.2%, 37.6%, and 25.2% of baseline. Their 64-step correlations are 0.9973,
0.9770, and 0.8874, below the 0.999 gate, with maximum errors 4.59, 13.21, and
23.95. Depth candidates shrink to 50.5%, 38.0%, and 25.6%; deterministic
argmax mismatch rises to 46.3%, 68.7%, and 94.0%.

All candidates remain finite. None reaches a phone. We therefore make no claim
that compression improves latency or preserves audio, and the sustained system
uses uncompressed temporal plus the prior FP16 depth artifact.

## 6. Discussion

### 6.1 The benchmark must follow the failure surface

The three clocks correspond to three different classes of bug. Compute failure
is exposed by stage p99 and thermal trend. Delivery failure is exposed by ring
depth, pushed/pulled accounting, and callback counters. Generative failure is
exposed by full-horizon audio and calibrated controls. Substituting one class's
instrument for another is invalid: an audio embedding cannot prove no
underruns, and a zero-underrun log cannot prove music.

The stride bug demonstrates a fourth requirement: the evidence path must be
tested too. Live playback already read the decoder tensor by declared strides;
the recorder assumed dense storage and manufactured huge STFT coefficients
from padding. Replaying the historically clean rolling model through the same
recorder reproduced the corruption, falsifying the new temporal graph as the
sole cause. One padded-FP16 unit test then closed the recorder defect. Without
that falsification sequence, we might have "fixed" a correct renderer.

### 6.2 Accelerator placement is a runtime property

A convertible graph is not necessarily an admitted graph, and a requested
policy is not observed execution. Our state boundary succeeded because it
removed mutation from the compiled graph, but the contribution is not merely
that this graph can compile. The 10-process matrix establishes repeatability in
the application, while the model-attributed trace establishes physical ANE
activity and GPU absence. Either artifact alone is weaker.

The broader design lesson is conservative: keep the compiled graph pure and
fixed; keep small mutation and irregular lookup on the host; measure the whole
pipeline. This resembles NPU partitioning in mobile LLM systems, but here the
partition is constrained by a continuous audio deadline and by the need to
prove state across predictions.

### 6.3 Negative results change the system boundary

The A14 result rules out a tempting product claim. A 20-second reservoir is
large enough to make short demonstrations smooth, yet the full run exposes the
sustained 0.897x rate. The compression ladder rules out a tempting engineering
response: package bytes can be cut in half, but deterministic behavior changes
before we have earned a speed measurement. The lower-temperature audio arm
rules out another: suppressing one detector while damaging four other measures
is not a fix.

Each failure narrows the next legitimate experiment. A14 needs a materially
different precision/training or depth/decoder architecture, not a larger hidden
buffer. The long-horizon quality defect needs token-level comparison against
the reference sampler across the onset, not more audio-thread tuning. We stop
where the evidence stops.

## 7. Limitations

This study uses one prompt, one principal sampling seed, two phones, and one
model size. It characterizes a system and a failure mechanism, not the prompt
distribution of MRT2. The A17 soak deliberately begins hot, which strengthens
the sustain observation but is not an estimate of ordinary battery behavior.
Instruments power-impact scores are not calibrated joules; we do not make a
battery-life claim. A corrected, counterbalanced Power Profiler pair could not
be completed because Instruments lost USB attachment to both phones after the
signed bundle was repaired. The invalid preflight traces are excluded, and we
report neither an energy comparison nor a process-impact comparison. The blind
audio votes come from a multimodal model rather
than human MUSHRA listeners, so they are treated only as a calibrated detector
with explicit controls. The complete product runtime is private; public
artifacts include exporters, validators, fixtures, pseudocode-level contracts,
model hashes, normalized result manifests, the gate verifier, and this paper.
The runtime is available from the author for artifact review.

Most importantly, the generative-quality gate fails. The title's
"GPU-free MRT2" refers to observed compute placement and sustained inference,
not to a claim that the current output is ready to ship for arbitrary-length
listening. A follow-up should reproduce the click onset with token taps, compare
the Core ML and MLX sampled trajectories under identical noise, and localize
the first divergence before changing another system parameter.

## 8. Conclusion

We demonstrate that the full compute and delivery path of a 230M-parameter
live-music model can run GPU-free on an iPhone 15 Pro Max for ten hot foreground
minutes with 1.031x production, 21.66 ms p99 effective frame cost, zero
underruns, and a growing audio reservoir. A pure one-frame temporal graph plus
host-owned K/V state makes ANE admission reliable without fragmenting the
model. The same experiment draws a hard A14 boundary and rejects six attractive
but invalid compressed artifacts.

It also demonstrates why those results are not the finish line. The
600-second output fails a frozen pulse detector and three of five calibrated
blind votes even though performance, delivery, semantic, and stereo metrics
pass. Live generation therefore has three clocks, not one. The scientifically
correct result is both halves at once: sustained GPU-free inference on a phone
is solved here; sustained generative quality is not.

## Artifact statement

The public repository contains the Core ML exporters, frozen fixtures,
validation scripts, machine-readable G1-G4 manifests and verifier reports,
figure sources, manuscript source, model-weight hashes, and exact protocol.
Preconverted public artifacts are mirrored on Hugging Face. Private raw device
logs, Instruments traces, 600-second WAVs, and the Crossfade application are
hashed by the public manifests and are available from the author for review.

## Acknowledgments

OpenAI Codex assisted with experimental code, data normalization, figure
generation, and manuscript preparation. The author designed the study,
operated the devices, inspected the evidence, selected the claims, and accepts
responsibility for the manuscript.

## Reproducibility

### Public evidence map

The repository's `validation/results/system-paper` tree is the canonical
publication dataset. The A17 and A14 `placement` directories contain G1
manifests and verifier reports; `a17pro/soak` contains G2; `audio` contains G3;
`a14/soak` contains G4; and `evaluation/evaluation-manifest.json` contains the
four five-process latency cells. `MRT2WeightCompressionLadder.json` contains
the six early-stopped precision arms. Each manifest hashes the private raw
event trace, console log, WAV, or Instruments export from which it was
normalized. The verifier reports hash their input manifests, so a changed
number invalidates the chain rather than silently updating a verdict.

The artifact identities that bind the principal runs are the same across the
placement and sustain receipts: temporal weight SHA-256 prefix `02657e24`,
depth `6c69ebae`, A17 FP16 decoder `ceb3ed7d`, and A14 FLOAT32 decoder
`38cbdf5c`. Full hashes are retained in the manifests and model metadata; the
prefixes are printed here only for readability. The 64-step temporal fixture
contains 23 predictions after the 41-slot ring wraps. The public G1 report hash
is itself included in G2, and the G2 report hash is included in G3.

### Runtime protocol

The producer implements the following bounded protocol:

1. Load the signed, precompiled Core ML packages with explicit compute
   policies; query the temporal compute plan; run the fresh/warmed and 64-step
   state proofs; then execute three untimed warmups.
2. Prime a fixed-capacity single-producer/single-consumer PCM ring. For each
   temporal frame, present chronological K/V inputs, predict one temporal
   update, copy the 48 updates into the current ring slot, sample all 12 depth
   levels in one prediction, and sum the selected RVQ rows.
3. At decoder cadence, predict the STFT block, materialize it by declared
   shape and strides, perform inverse STFT and overlap-add, and push PCM. The
   audio callback performs only a bounded ring read; underflow is zero-filled
   and counted.
4. Every ten seconds, clear temporal K/V and previous temporal feedback while
   leaving decoded PCM, decoder overlap, and ring state continuous. Stop only
   after the measured horizon; capture events and rendered PCM independently.

The fixed condition is `warm ambient texture`, `warm.bin`, top-k 40, style
guidance 3.1, temperature 1.0, seed 20260718, 48 kHz stereo, foreground,
screen on. The principal A17 run begins already in the `serious` thermal state.
The A14 run begins nominal and is allowed to transition naturally. The matched
sustain control uses the same condition and differs only in requested temporal
compute policy. A power pair was attempted under that design but produced no
valid measurement, as stated in the limitations.

### Re-running the public verdicts

The four hard verdicts require only Python and the checked-in manifests. Run
the gate verifier with `g1` on the A17 placement manifest, then repeat with
`g2`, `g3`, and `g4` on the corresponding sustain, audio, and A14 manifests.
The relevant unit suites are:

- `test_system_paper_gates`;
- `test_system_paper_soak_manifest`;
- `test_system_paper_audio_manifest`; and
- `test_system_paper_evaluation_manifest`.

They exercise both accepted receipts and binding false positives. Figure
sources read only public JSON. The manuscript builder regenerates LaTeX from
this readable draft; Tectonic builds the PDF without a venue-specific class.

Device recapture additionally requires the private Crossfade host, the signed
model bundle, and physical phones. The public exporter and fixture tests can
still reproduce model boundaries, deterministic parity, compression rejection,
manifest normalization, and every publication verdict without that host. This
division is deliberate and is the limitation stated in the main text, not an
implied open-source claim about the product runtime.
