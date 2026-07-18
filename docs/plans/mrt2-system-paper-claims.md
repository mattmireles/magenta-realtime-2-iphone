# MRT2 System Paper Claims Ledger

- **Frozen:** 2026-07-18
- **Owner:** Matt Mireles
- **Primary publication path:** arXiv technical paper
- **Length target:** 10–12 pages of main text plus references and a compact
  reproducibility appendix

**Paper title:** *Throughput Is Not Liveness: Three Clocks for GPU-Free Music
Generation on iPhone*

This ledger is the paper's scientific contract. A sentence may enter the
abstract, contributions, conclusion, or public launch copy only if its gate is
green and the cited receipt exists. Failed gates remain publishable negative
results, but they do not become qualified versions of a stronger claim.

## Publication and Artifact Decisions

### Publication target

The deliverable is a venue-neutral arXiv paper, not a conference submission.

- Use a clean single-column arXiv layout optimized for reading, not a venue
  template.
- Budget 10–12 main-text pages for the complete system, two-device evaluation,
  thermal behavior, audio integrity, negative results, and limitations.
- Put dense command lines, schemas, and artifact hashes in a short appendix so
  the scientific narrative remains tight.
- Write to the standard of a selective systems/audio venue, but do not spend
  execution time on CFPs, anonymity, page-limit games, or submission portals.
- Matt Mireles is the sole human author and is responsible for every result,
  citation, and sentence. OpenAI Codex assistance may be acknowledged for
  experimental code, analysis, figures, and manuscript preparation.

### Artifact story

The public artifact is this repository plus its Hugging Face mirror:
exporters, validators, frozen fixtures, model contracts, paper source, and
machine-readable result summaries. The shipping Crossfade runtime remains
private for this submission and is identified as available from the author for
artifact review. No proprietary runtime subset will be copied into the public
repository during this execution.

This is the smallest honest artifact story. The paper will include pseudocode
for the producer/ring/reservoir contract, exact command lines, hashes for model
artifacts, summarized Instruments exports, and enough checked-in data to audit
every reported number without publishing the product application. The public
gate verifier is `validation/verify_system_paper_gate.py`; private capture tools
may produce the manifests, but every publication verdict is reproducible here.

These publication and artifact choices were made under the user's 2026-07-18
instruction to execute the plan and take the actions required to produce an
immediately publishable paper. They may be changed before external submission,
but the claims and evidence gates below may not be weakened silently.

## Measurement Contract

- **Model:** `mrt2_small` only.
- **Devices:** `Commas` = iPhone 15 Pro Max (`iPhone16,2`, A17 Pro);
  `Webcam` = iPhone 12 Pro (`iPhone13,3`, A14).
- **Audio contract:** 25 RVQ frames/s; 1,920 samples/frame; 48 kHz stereo.
- **Throughput contract:** producer rate at least 1.0x, p99
  iteration-normalized active stage cost below 40 ms/token, and no draining
  finite reservoir. A mean, p50, short run, or prefilling result cannot be
  labeled as an unqualified throughput pass.
- **Sustain contract:** foreground, screen on, 600 s after priming, with
  thermal state, ring depth, produced/pulled audio, dropped frames, and
  underruns logged throughout.
- **Iteration-normalized compute cost:** for each 25-token producer iteration,
  sum temporal + depth + sampling + decoder work and divide by 25; then report
  quantiles across iterations. This is explicitly not a per-token latency
  distribution. Backpressure sleep is excluded from compute cost and included
  in wall-clock duty cycle.
- **Delivery and steering contract:** zero callback underruns and producer
  drops are necessary, but a queue satisfies interactive delivery only when its
  maximum depth is chosen from an audible steering-latency budget. The current
  21-second soak reservoir proves continuity, not low-latency steering.
- **Precision gate order:** `finite_ratio == 1.0` first; only then parity,
  token agreement, SNR/correlation, latency, and audio judgment.
- **State gate:** fresh-vs-warmed divergence plus streaming agreement beyond
  the 41-frame attention window.
- **Placement gate:** requested compute units are metadata, not evidence. A
  passing receipt requires Instruments ANE prediction intervals for every
  model family claimed as ANE-resident, Core ML prediction intervals for every
  composed model family, and zero GPU interval rows attributed to the app
  process. The current pipeline claims ANE residency for temporal and decoder;
  its FLOAT32 depth rollout is deliberately CPU-resident and is reported as
  such rather than mislabeled.
- **Dispersion:** at least five independent runs per latency cell; report the
  median of run-level p50/p90/p99 values and IQR (or all run values when N=5).
- **Power:** report process-attributed impact scores, CPU instructions, and
  duty cycles. Do not convert these data into joules or battery-life claims.

## Hard Gates

### G1 — Shipping-process ANE placement and GPU absence

**Pass:** On `Commas`, 10/10 consecutive cold app launches produce an
in-process temporal compute-plan and state receipt with:

1. temporal ANE estimated cost weight at least 0.95 and zero temporal GPU
   operations/cost;
2. the same model hashes and compute policy recorded in all 10 runs; and
3. fresh-vs-warmed divergence plus a successful 64-frame frozen-fixture match
   for that exact artifact (23 predictions after the 41-frame ring wraps).

An Instruments trace from the same signed artifact must additionally show:

1. temporal and decoder ANE prediction intervals greater than zero;
2. temporal, depth-rollout, and decoder Core ML prediction intervals greater
   than zero; and
3. app-attributed Metal GPU interval count and duration both equal to zero.

This separates two questions cleanly: the ten-launch matrix establishes
admission reliability for the temporal artifact, while the process trace
establishes actual hot-path accelerator activity and GPU absence. One trace is
not repeated ten times because the model hash, signed app artifact, and
in-process plan are invariant and each repeat would answer no new question.

`Webcam` repeats the 10-launch matrix as cross-device evidence. An A14 failure
is reported rather than allowed to invalidate an A17-only headline.
The A17 manifest declares `deviceRole: primary`; the A14 manifest declares
`deviceRole: cross-device`, and the public verifier rejects a role/device
mismatch.

**Capture and public verifier:**

```bash
python3 scripts/summarize_coreml_xctrace_exports.py RUN_EXPORT_DIR \
  --placement-evidence-json RUN_EXPORT_DIR/placement-evidence.json \
  --required-model ACTUAL_TEMPORAL_MODEL \
  --required-model ACTUAL_DECODER_MODEL \
  --required-runtime-model ACTUAL_TEMPORAL_MODEL \
  --required-runtime-model ACTUAL_DEPTH_MODEL \
  --required-runtime-model ACTUAL_DECODER_MODEL \
  --app-process CrossfadeRuntimeHost

python3 scripts/verify_crossfade_placement_policy.py \
  --event-summary RUN_EXPORT_DIR/event-summary.json \
  --placement-evidence RUN_EXPORT_DIR/placement-evidence.json \
  --require-ane-proof \
  --output-json RUN_EXPORT_DIR/placement-verification.json \
  --output-md RUN_EXPORT_DIR/placement-verification.md

python3 validation/verify_system_paper_gate.py g1 \
  validation/results/system-paper/a17pro/placement/g1-manifest.json \
  --output-json validation/results/system-paper/a17pro/placement/g1-report.json
```

**Public receipt roots:**

- `validation/results/system-paper/a17pro/placement/`
- `validation/results/system-paper/a14/placement/`

### G2 — Sustained A17 Pro throughput and delivery

**Pass:** A foreground, screen-on run on `Commas` lasting at least 600 s after
priming has:

- `maxUnderruns == 0` and `maxDropped == 0`;
- pulled audio duration at least 600 s;
- generation rate at least 1.0x real time over the measured window;
- p99 iteration-normalized active stage cost below 40 ms/token, explicitly
  labeled as a quantile of 25-token iteration means rather than per-token tail
  latency;
- a nondecreasing final safety margin after the initial prime (the run may
  backpressure, but may not survive by draining a finite reservoir); and
- a thermal timeline included even if it reaches `serious`.

**Capture analyzer and public verifier:**

```bash
python3 scripts/analyze_crossfade_runtime_host_log.py RUN.log --json \
  > RUN.summary.json

python3 validation/verify_system_paper_gate.py g2 \
  validation/results/system-paper/a17pro/soak/g2-manifest.json \
  --output-json validation/results/system-paper/a17pro/soak/g2-report.json
```

The G2 manifest must expose the exact measured wall-clock interval, at least
15,000 effective-frame samples, generation rate, and reservoir slope. The
public verifier rejects a zero-underrun run if its finite prime drains.

**Public receipt root:**
`validation/results/system-paper/a17pro/soak/`

### G3 — Audio integrity

**Pass:** A 600 s capture from the corrected G2 runtime, using the fixed prompt
`warm ambient texture`, temperature ≤1.1, and top-k 40:

- passes deterministic PCM integrity checks for finite samples, clipping,
  channel count/order, sample rate, chunk-boundary discontinuities, and
  sustained underrun/dropout counters;
- has finite ratio 1.0, clipped-sample ratio ≤1e-5, maximum normalized
  chunk-boundary jump ≤0.07, explicit `[left, right]` channel order, L/R
  correlation ≥0.97, prompt adherence ≥0.30, similarity to the clean reference
  ≥0.80, and 4–16 Hz envelope-pulse share ≤0.07;
- passes at least 4/5 blinded automated listening votes against the MLX
  reference, with five unique frozen order seeds; a vote counts only if it
  passes the known-good control, rejects the known-bad control, and ranks the
  controls correctly; and
- rejects every frozen known-bad control (stride corruption, missing temporal
  feedback, write-only state, click-comb control, channel collapse, and
  dropout injection).

No listening model's prose is treated as ground truth by itself. The numeric
bands were frozen from the prior clean A17 capture (adherence 0.333, L/R 0.987,
reference similarity 0.87, pulse share 0.056) and separated from the known-bad
click control (0.093 pulse share) before the new run.
The G3 manifest must hash the passing G2 report and carry the same zero-underrun
and zero-drop counters, preventing a clean excerpt from masking a failed run.

**Public verifier:**

```bash
python3 validation/verify_system_paper_gate.py g3 \
  validation/results/system-paper/audio/g3-manifest.json \
  --output-json validation/results/system-paper/audio/g3-report.json
```

**Public receipt root:**
`validation/results/system-paper/audio/`

**Original measured result (2026-07-18): FAIL, retained as the investigation
trigger.** The first corrected 600 s capture
passes finite, clipping, boundary, stereo, prompt-adherence, reference-
similarity, delivery-counter, and all six known-bad-control checks. It fails
the frozen 4–16 Hz pulse-share band (`0.1364 > 0.07`) and receives only 2/5
candidate passes from five otherwise-valid calibrated blind votes. A
temperature-0.5 intervention fixes pulse share but fails clipping, stereo,
prompt, and reference bands. G3 is not weakened or tuned away.

#### G3-R — causal localization and context repair

The original threshold is now treated as a prompt-specific diagnostic, not a
universal music-quality oracle. The revision freezes three causal hypotheses
and requires three independent 600-second seeds crossed by token source and
decoder path, fixed-token graph/DSP controls, a decoder-history tensor probe,
and a paired context intervention. The public verifier requires the stateless
effect to be positive in all 60 windows, the graph-only effect to remain below
0.001, the 12-frame intervention to be negative in all 60 windows, context-12
tensor correlation above 0.999999999, and a 600-second corrected A17 run with
zero underruns/drops.

**Measured result: PASS.** Stateless windowing adds 0.01706 median seed mean
pulse share and is positive in 60/60 windows. The Core ML graph contributes
0.00018. Twelve-frame context reduces the metric by 0.01650 and is negative in
60/60 windows; tensor correlation rises from 0.1083 to 0.999999999988. The
corrected physical-device run is finite, continuous, and matches the stateful
MLX prompt-specific diagnostic count (5/20) for seed 20260718. The old
automated-listening votes used overlapping excerpts from the superseded output
and are excluded from the primary causal claim rather than silently rerun.

**Public verifier:**

```bash
python3 validation/verify_system_paper_revision.py \
  --output-json validation/results/system-paper/revision-report.json
```

### G4 — A14 tier decision

Exactly one outcome must be selected from measured `Webcam` evidence:

- **Native throughput pass:** the pipeline has p99 iteration-normalized active
  stage cost below 40 ms/token and completes the G2-equivalent 600 s
  zero-underrun run
  without relying on a draining reservoir; or
- **Reservoir tier or bounded-reservoir failure:** report startup prime
  duration, achieved continuous-play duration, reservoir slope,
  p50/p90/p99 iteration-normalized active cost, thermal timeline, and underrun/drop
  counters. A tier requires a complete zero-underrun run. If the largest
  fixed reservoir still drains, report the first-underrun time and name the
  result a bounded-reservoir failure. The words "real time on a 2020 phone"
  are forbidden for either non-native outcome.

**Public receipt root:**
`validation/results/system-paper/a14/soak/`

**Public verifier:**

```bash
python3 validation/verify_system_paper_gate.py g4 \
  validation/results/system-paper/a14/soak/g4-manifest.json \
  --output-json validation/results/system-paper/a14/soak/g4-report.json
```

## Paper Claim Ledger

| ID | Candidate paper claim | Required gate / receipt | Current disposition |
| --- | --- | --- | --- |
| C1 | A complete `mrt2_small` pipeline generates and delivers 48 kHz stereo continuously on an iPhone 15 Pro Max with zero app-attributed GPU intervals in the attributed placement trace. | G1 + G2 + G3-R; A17 placement, crossover, and context-12 device roots | **Supported with explicit bounds.** The corrected run proves one 600-second prompt/seed trajectory, not arbitrary-prompt musical validity. The context-12 phone diagnostic matches stateful MLX for that seed. |
| C2 | The system sustains the 40 ms/token throughput contract for 10 foreground minutes with zero underruns. | G2; five-run latency dispersion plus the corrected 600 s soak | **Passed on A17 Pro.** The 610.31 s run produces at 1.0301x, p99 iteration-normalized active cost 24.81 ms/token, zero underruns/drops, and a non-draining reservoir. The statistic is a quantile of iteration means, not per-token tail latency. |
| C3 | A stateless host-owned K/V boundary makes the full temporal math island reliably ANE-resident in the shipping app. | G1 plus fresh/warm and >41-frame state receipts | **Passed on A17 Pro and A14.** Both phones pass 10/10 cold-process plan/state gates; both model-attributed traces show temporal ANE predictions and zero app GPU intervals. |
| C4 | Reducing temporal weight bytes improves frame cost and sustained headroom in the bandwidth-bound regime without changing audible behavior. | Quantization ladder: artifact bytes, finite/parity/audio gates, device latency and soak pair | **Rejected for the tested post-training methods.** Int8-linear, 6-bit palettized, and 4-bit palettized temporal artifacts shrink to 0.502x, 0.376x, and 0.252x of baseline, but all fail the 64-step deterministic state gate. They were stopped before device/audio measurement, so no speedup or audio-equivalence claim is supported. |
| C5 | The delivery architecture converts measured jitter into bounded queued playback without blocking the audio render thread. | Runtime log, ring/reservoir trace, zero-underrun soak, steering receipts, pseudocode | **Continuity passes; low-latency steering does not.** The corrected run captures 600 s with zero callback underruns/drops while the reservoir grows from 2.88 to 21.01 s. Separate receipts show 4-15 ms controller application but 6.48-9.02 s until audible change without queue discard. |
| C6 | GPU-free placement lowers process GPU impact and producer duty cycle relative to a temporal-GPU control. | Corrected-pipeline, counterbalanced Power Profiler pair; placement receipts | **Unsupported and omitted.** The first corrected-bundle trace is invalid because the bundle omitted `warm.bin` and exited at preflight; after repairing and testing the signed bundle, Instruments lost USB attachment to both phones. The invalid trace is retained and excluded. No energy, impact-score, or producer-duty-cycle claim is made. |
| C7 | The same pipeline has a measured, honest deployment tier on an A14 iPhone. | G4 | **Passed as a bounded-reservoir failure, not a tier.** p99 is 58.59 ms, production 0.8967x, the maximum 20.16 s reservoir first underflows at 294.08 s, and the 610 s run records 7,952 underruns. Unqualified A14 real time is forbidden. |
| C8 | Cold start to first audible PCM is measured on both phones. | Five cold launches/device with load, compile, prime, and first-render timestamps | **Measured.** Median (IQR) is 4.26 s (0.04) on A17 Pro and 7.52 s (0.01) on A14 under the selected ANE policy; all five run values remain in the evaluation manifest. |
| C9 | Placement reliability is a deployment property distinct from graph convertibility and one-off harness success. | Ten cold launches/device plus falsification matrix | **Passed.** The first post-install A14 process costs 39.498 s while warmed gate processes cost about 1.7 s, yet all 10/10 on each device preserve the same ANE plan and state result. |
| C10 | Negative results identify where state, precision, compression, placement, thermal behavior, or long-horizon generation break. | Failed variants retained with commands, hashes, and failure text | **Supported and revised.** The public receipts retain six rejected compression variants, the A14 FP16/FLOAT32 boundary and bounded-reservoir failure, recorder-stride falsification, the original failed 600 s diagnostic, and the crossover that rejects its model-degeneration interpretation in favor of missing decoder context. |

## Claims Explicitly Forbidden

- "All computation runs on the ANE." Host control, sampling entropy, buffer
  management, audio DSP, and UI remain deliberate CPU stages.
- "Zero GPU" based only on `.cpuAndNeuralEngine`, `MLComputePlan`, or process
  power impact. The claim requires an empty app row set in Instruments Metal
  GPU intervals from the same run.
- "Real time" based on a p50, a short run, or a finite reservoir that drains.
- "Thermally stable" merely because the process did not terminate.
- Calibrated energy, battery-life, or joule claims from Instruments impact
  scores.
- A14 unqualified real time if its result is the reservoir tier.
- Novelty for conversion, layout, state, or bandwidth findings already
  published in *Surgical Inference*. This paper cites them as prior method and
  contributes the corrected composed system, reliability study, sustain
  envelope, and audio-delivery evaluation.

## Phase 0 Exit Check

- [x] Every candidate headline and contribution claim maps to a named gate.
- [x] Pass criteria separate A17 headline evidence from the A14 tier.
- [x] arXiv-first publication target and a venue-neutral 10–12-page main-text
  budget are frozen; conference submission rules are explicitly out of scope.
- [x] Public/private artifact boundary is explicit.
- [x] Failed gates have predeclared negative-result language.
- [x] Prior *Surgical Inference* findings are marked as cited method, not new
  contributions.
- [x] `python3 -m unittest validation.test_system_paper_gates` proves every gate
  accepts its declared success case and rejects its binding false-positive.
