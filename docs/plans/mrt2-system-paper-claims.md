# MRT2 System Paper Claims Ledger

- **Frozen:** 2026-07-18
- **Owner:** Matt Mireles
- **Primary publication path:** arXiv technical paper
- **Length target:** 10–12 pages of main text plus references and a compact
  reproducibility appendix
**Working title:** *Live: Sustained GPU-Free Generative Music on iPhone*

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
- **Real-time contract:** p99 effective production cost below 40 ms/frame, or
  an explicitly named buffering tier. A mean, p50, or prefilling result cannot
  be labeled as an unqualified real-time pass.
- **Sustain contract:** foreground, screen on, 600 s after priming, with
  thermal state, ring depth, produced/pulled audio, dropped frames, and
  underruns logged throughout.
- **Effective frame cost:** temporal + depth + sampling + decoder cost
  amortized by the exact number of decoded RVQ frames. Backpressure sleep is
  excluded from compute latency and included in wall-clock duty cycle.
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

### G2 — Sustained A17 Pro real time

**Pass:** A foreground, screen-on run on `Commas` lasting at least 600 s after
priming has:

- `maxUnderruns == 0` and `maxDropped == 0`;
- pulled audio duration at least 600 s;
- generation rate at least 1.0x real time over the measured window;
- p99 effective frame production cost below 40 ms;
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

### G4 — A14 tier decision

Exactly one outcome must be selected from measured `Webcam` evidence:

- **Native real-time pass:** the corrected pipeline has p99 effective frame
  cost below 40 ms and completes the G2-equivalent 600 s zero-underrun run
  without relying on a draining reservoir; or
- **Reservoir tier:** report startup prime duration, achieved continuous-play
  duration, reservoir slope, p50/p90/p99 effective frame cost, thermal
  timeline, and zero-underrun result. The words "real time on a 2020 phone"
  are forbidden for this outcome unless always qualified by the measured
  startup reservoir tier.

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
| C1 | A complete `mrt2_small` pipeline generates 48 kHz stereo music continuously at 25 Hz on an iPhone 15 Pro Max with zero app-attributed GPU time. | G1 + G2 + G3; A17 placement, soak, and audio roots | **G1 passed; blocked on G2/G3.** The composed A17 trace is GPU-empty and the observed short-run cost is below budget, but the headline still requires the 600 s and audio gates. |
| C2 | The system sustains the 40 ms/frame contract for 10 foreground minutes with zero underruns. | G2; five-run latency dispersion plus the 600 s soak | **Blocked.** Prior corrected-pipeline soak fails around minutes 5–7 and is negative-result context only. |
| C3 | A stateless host-owned K/V boundary makes the full temporal math island reliably ANE-resident in the shipping app. | G1 plus fresh/warm and >41-frame state receipts | **Passed on A17 Pro and A14.** Both phones pass 10/10 cold-process plan/state gates; both model-attributed traces show temporal ANE predictions and zero app GPU intervals. |
| C4 | Reducing temporal weight bytes improves frame cost and sustained headroom in the bandwidth-bound regime without changing audible behavior. | Quantization ladder: artifact bytes, finite/parity/audio gates, device latency and soak pair | **Rejected for the tested post-training methods.** Int8-linear, 6-bit palettized, and 4-bit palettized temporal artifacts shrink to 0.502x, 0.376x, and 0.252x of baseline, but all fail the 64-step deterministic state gate. They were stopped before device/audio measurement, so no speedup or audio-equivalence claim is supported. |
| C5 | The delivery architecture converts measured jitter into bounded startup latency without blocking the audio render thread. | Runtime log, ring/reservoir trace, zero-underrun soak, pseudocode | **Supported historically, must be re-proven on the corrected pipeline.** |
| C6 | GPU-free placement lowers process GPU impact and producer duty cycle relative to a temporal-GPU control. | Corrected-pipeline, counterbalanced Power Profiler pair; placement receipts | **Blocked pending re-run.** The old 0.57-vs-0.93 duty-cycle pair predates correctness fixes. |
| C7 | The same pipeline has a measured, honest deployment tier on an A14 iPhone. | G4 | **Guaranteed as a reporting result, not guaranteed as a real-time pass.** |
| C8 | Cold start to first audible PCM is measured on both phones. | Five cold launches/device with load, compile, prime, and first-render timestamps | **Unmeasured.** No responsiveness claim until receipts exist. |
| C9 | Placement reliability is a deployment property distinct from graph convertibility and one-off harness success. | Ten cold launches/device plus falsification matrix | **Passed.** The first post-install A14 process costs 39.498 s while warmed gate processes cost about 1.7 s, yet all 10/10 on each device preserve the same ANE plan and state result. |
| C10 | Negative results identify where state, precision, compression, placement, or thermal behavior break. | Failed variants retained with commands, hashes, and failure text | **Supported for state, precision, compression, and placement; thermal pending G2/G4.** The compression receipt retains sizes, hashes, metrics, gate order, and the explicit early-stop disposition for all six rejected variants. |

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
