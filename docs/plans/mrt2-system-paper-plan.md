# MRT2 System Paper Plan — "Live: Real-Time Streaming Music Generation on iPhone, GPU-Free"

**Date:** 2026-07-15
**Status:** Active — Phases 0–2 complete; Phase 3 in progress

## Executive Summary

Ship the sequel to *Surgical Inference*: an end-to-end **system paper** whose
headline claim is that a full generative music pipeline (temporal transformer →
12-level RVQ depth rollout → SpectroStream decode → 48 kHz stereo render) holds
a **25 Hz / 40 ms-per-frame real-time contract on iPhone with zero
app-attributed GPU time**, sustained under thermal soak, with receipts. The
paper is the reward for closing three named engineering gaps; the plan covers
both the engineering and the writing, ending with an arXiv-ready PDF and a
  publication bundle suitable for expert academic review.

## Problem Statement

- **Symptom:** All the artifact-level results exist (ANE-resident stages,
  parity receipts, power pairs), but *Surgical Inference* §6.7 honestly reports
  the composed system as open: the A17 Pro fails the 10-minute foreground soak
  (throttles to ~0.95× real time at minutes 5–7), the A14 misses the frame
  budget (~50 ms/frame), and the shipping temporal placement is `.cpuAndGPU`
  because ANE admission proved instance-fragile.
- **Root Cause:** The first paper's unit of contribution was the *artifact*;
  the composed runtime (private Crossfade repo) was out of scope. The system
  claims were never engineered to a gate.
- **Impact:** Without closing the gaps there is no second paper — republishing
  the §6.3–6.5 findings alone would be salami-slicing a spent result. With
  them closed, the result ("live generative music at 48 kHz stereo on a 2020
  phone, GPU-free") is unpublished anywhere.

## Goals and Non-Goals

### Goals

- [ ] Temporal stack ANE-resident **in the shipping runtime** (not just the
      probe harness), proven by the in-app machine-readable placement gate.
- [ ] 10-minute foreground thermal soak on A17 Pro at ≥1.0× real time, 0
      render underruns, screen on, corrected pipeline.
- [ ] A14 story resolved: either <40 ms/frame via temporal weight-byte
      reduction, or an explicit, measured higher-startup-latency reservoir
      tier reported as such.
- [ ] Audio integrity closed: clicking/stutter/dropout defects root-caused and
      fixed, with blind audio-quality gates passing (known-bad controls
      rejected).
- [ ] Full evaluation matrix with dispersion (repeat runs) and checked-in
      receipts for every number the paper states.
- [ ] arXiv-ready PDF and source bundle built, audited, and submitted.

### Non-Goals

- **Re-deriving the §6.3–6.5 conversion findings.** They are published in
  *Surgical Inference*; this paper cites them as method and contributes the
  composed system. No claim overlap beyond citation.
- **New model variants** (mrt2 base/large). `mrt2_small` only.
- **Battery-life claims.** Power evidence stays process-attributed impact
  scores and duty cycles, per the receipts discipline already in
  `docs/validation-receipts.md` §4.4.
- **Shipping the Aperture product app.** Product work (Crossfade Plans
  005–014) continues independently; this plan touches the runtime only where
  the paper's gates require it.

## Scope and Constraints

- **Scope:** Engineering gates live mostly in the private **Crossfade** repo
  (`/Users/mm/Documents/GitHub/crossfade`, `Sources/CrossfadeRuntime*`);
  exporters, validators, receipts, and the paper itself live **here**
  (`magenta-realtime-2-iphone`), with model binaries mirrored on the HF repo
  (`mattmireles/magenta-realtime-2-iphone`).
- **Constraints:** Two test devices (iPhone 12 Pro / A14, iPhone 15 Pro Max /
  A17 Pro). ANE admission is instance-fragile per *Surgical Inference* §6.7 —
  every placement claim must be re-proven in the app process, never inferred
  from a harness run.
- **Guardrails:** The receipts rule from `docs/validation-receipts.md` is
  non-negotiable: every number in the paper points to a command, a receipt
  file, or an explicit failure. Negative results stay in the paper.

## Ground Truth Contracts (Do Not Violate)

- **40 ms frame budget:** MRT2 streams at 25 Hz. "Real time" means p99 frame
  production <40 ms *or* an explicitly reported buffering tier — never a
  blended claim.
- **`finite_ratio == 1.0` gates every precision change** before SNR/latency
  are even read (the FP16 decoder lesson, receipts §3).
- **Stateful graphs prove state is READ:** fresh-vs-warmed divergence test +
  N-prediction streaming match past the window (the write-only-state bug,
  paper §6.6). Applies to any new temporal export.
- **Strided `MLMultiArray` reads:** all Swift consumers read via `strides`,
  validated once against a Python reference (the depth-logits misread bug).
- **Placement is proven, not requested:** `MLComputePlan` + Instruments ANE
  intervals + zero app-attributed GPU rows, captured **in the shipping app
  process**, via the existing machine-readable placement gate.
- **No content drift into paper 1:** *Surgical Inference* is frozen. This
  paper cites it; it does not amend it.

## Already Shipped (Do Not Re-Solve)

- **Stateless host-cache temporal boundary:**
  `exporters/convert_temporal_body_carry.py` — compiles the full 12-layer
  stack to a single ANE island (A14 `costWeights=ane:1.000`, p99 14.991 ms),
  parity 0.999984. The artifact exists; what is unsolved is *reliable
  admission in the app*.
- **FP16 NCHW decoder:** `exporters/convert_spectrostream_decoder.py
  --nchw-parallel-layer 5 --fp16-rescale` — finite and ANE-resident (A14 p99
  24.77 ms / 25 frames).
- **In-graph FP16 depth rollout:** `exporters/convert_depth_body_rollout.py` —
  12 levels per prediction, 12.7 ms/frame (A14) / 8.4 ms (A17 Pro), FP32
  token-exact receipt.
- **Delivery architecture:** SPSC ring + reservoir + backpressure in
  Crossfade — validated by the earlier 10-minute zero-underrun runs.
- **Placement-evidence gate:** machine-readable verifier requiring ANE
  intervals per model family and rejecting app-attributed GPU rows (paper
  §6.7).
- **Host-glue validation rules + audio-quality gates:** blind automated
  listening gates with known-bad controls (paper §6.6); thermal soak
  instrumentation (Crossfade commit `e00faff`).

## Fresh Baseline (Current State)

- **A17 Pro:** composed ≈29 ms/frame p50 (temporal 12.2 **on GPU** + depth
  8.4 CPU + decoder 8.3 ANE), ≈1.37× real time, 0 underruns over bounded
  45 s runs — but the 10-minute foreground soak **fails**: both soak arms
  bank ~20 s of lookahead then throttle to ~0.95× real time around minutes
  5–7 (screen on, UI Metal load included).
- **A14:** ≈50 ms/frame — DRAM-bandwidth bound (temporal+depth ≈0.7 GB/frame
  ≈67% of A14 bandwidth). Passes 10 minutes only with a 15 s prefilled
  reservoir.
- **Known gaps:** (1) temporal ANE admission fragile in-app, shipped
  placement `.cpuAndGPU`; (2) temporal weight bytes uncompressed; (3)
  periodic clicking/stutter/dropouts in the composed runtime; (4) soak gate
  unmet.

## Solution Overview

The paper's own findings predict the fix chain. Weight bandwidth is the
invariant (§6.5): cutting temporal weight bytes directly cuts per-frame cost
on *every* compute unit and DRAM heat. ANE residency is the thermal lever
(§6.7 power pair: duty cycle 0.57 vs 0.93). Do both, and the soak gate and
the A14 budget are predicted to follow; then measure everything twice and
write it down.

```
Phase 0            Phase 1                Phase 2               Phase 3
freeze claims  →   temporal on ANE    →   cut weight bytes  →   soak + audio
+ gates            in the app             (palettize/int8)      integrity
                        \__________________________________________/
                                            |
                             Phase 4: evaluation campaign (receipts)
                                            |
                             Phase 5: write the paper (LaTeX/PDF)
                                            |
                             Phase 6: artifacts, audit, submit
```

## Implementation Phases

> Do one phase at a time. Verify before proceeding. Engineering phases 1–3
> land in Crossfade (each as its own numbered Crossfade plan per that repo's
> convention); this plan is the paper-level tracker.

### Phase 0: Freeze Claims, Gates, and Publication Target

**Goal:** A one-page claims ledger the whole plan is accountable to, before
any engineering.

**Tasks:**

- [x] Write the claims ledger: every headline sentence the paper intends to
      make, each mapped to a gate and a receipt path. File:
      `docs/plans/mrt2-system-paper-claims.md` (this repo).
- [x] Define the four hard gates (G1 ANE-in-app, G2 A17 soak, G3 audio
      integrity, G4 A14 tier decision) with exact pass criteria and the
      command/artifact that proves each.
- [x] Publication decision: arXiv-first, venue-neutral, 10–12 pages of main
      text plus references and a compact reproducibility appendix. Conference
      selection and submission rules are out of scope by user direction.
- [x] Artifact-story decision: what of Crossfade must become public (or be
      excerpted into this repo) for the paper's artifact statement. Options:
      publish `Sources/CrossfadeRuntime*` subset here; keep private and state
      so (as *Surgical Inference* Appendix A.2 already does). Owner: user.

**Verification:** `docs/plans/mrt2-system-paper-claims.md` freezes every claim,
gate, publication target, and artifact boundary. Under the user's 2026-07-18
authority to execute the plan and do what is required for immediate
publication, the execution choice is a venue-neutral arXiv paper, with the
Crossfade runtime kept private and available from the author.

---

### Phase 1: Re-Land Temporal on the ANE in the Shipping Runtime (Gate G1)

**Goal:** The stateless host-cache temporal boundary runs ANE-resident inside
the Crossfade app process, reliably across launches, proven by the in-app
placement gate. This is the single highest-leverage phase: it is the thermal
lever (duty cycle) *and* removes the GPU from the hot path, restoring the
paper's "GPU-free" headline.

**Tasks:**

- [x] New Crossfade plan (`README/Plans/015-...`): integrate the carry-boundary
      artifact (`exporters/convert_temporal_body_carry.py` output) into
      `Sources/CrossfadeRuntime*`, replacing the rolling GPU temporal path
      behind a runtime flag (kill switch — see Rollback).
- [x] Root-cause the harness-vs-app admission delta from §6.7. Falsify one
      variable at a time, same discipline as receipts §5.3: model load order,
      concurrent Metal/UI load at compile time, memory pressure, compilation
      cache state (`.mlmodelc` precompiled vs on-device), app entitlements /
      background state. Each experiment gets a note in Crossfade
      `README/Notes/`.
- [x] Implement host-owned K/V mutation in Swift (48 in / 48 out, strided
      reads, preallocated buffers — no per-frame allocation on the producer
      thread).
- [x] Wire the placement-evidence gate to run in-app at session start and
      record its artifact per run; a CPU-fallback session must be *detected*,
      not discovered in Instruments later.
- [x] Cross-prediction state test for the carry path in the app (fresh vs.
      warmed divergence; N-frame streaming match vs. fixtures past the
      41-slot window).

**Verification (2026-07-18): PASS.** Crossfade commit `9798f47` implements the
one-model boundary. A17 Pro and A14 each pass 10/10 cold-process compute-plan
and 64-frame state gates. The A17 trace records 653 temporal and 24 decoder
ANE predictions; the A14 trace records 546 and 19 respectively. Both traces
contain Core ML predictions for temporal, CPU depth, and decoder and contain
zero app-attributed GPU intervals. Public manifests and verifier reports are
under `validation/results/system-paper/{a17pro,a14}/placement/`; the readable
64-step parity receipt is
`validation/results/MRT2TemporalBodyStreamingCarry_validation.*`. Phase audit:
PASS (Swift tests 32/32, Python gate tests 15/15, signed device build, two
10-launch matrices, two model-attributed process traces).

---

### Phase 2: Cut Temporal Weight Bytes (Gates G2/G4 enabler)

**Goal:** Reduce temporal (and secondarily depth) weight bytes via
`coremltools.optimize` — palettization and/or int8 linear weight
quantization — to attack the ≈0.7 GB/frame DRAM invariant. Target: A14
composed <40 ms/frame; A17 Pro thermal headroom.

**Tasks:**

- [x] Export ladder in this repo: new flags on the temporal/depth exporters
      (`exporters/convert_temporal_body_carry.py`,
      `exporters/convert_depth_body_rollout.py`) for 6-bit/4-bit palettized
      and int8-linear weight variants. One variant = one artifact = one
      receipt.
- [x] Per-variant validation, in gate order: `finite_ratio == 1.0` →
      deterministic token mismatches vs. fixtures (temporal→depth composed,
      per receipts §2.1 methodology) → device latency p50/p99 → blind
      audio-quality gates on rendered audio (known-bad controls must still
      reject).
- [x] Apply the per-variant device-sweep admission gate on both phones. All
      six candidates failed deterministic reference parity, so the predeclared
      early-stop rule rejected them before installation, device timing,
      `MLComputePlan`, DRAM estimation, or listening. No compressed candidate
      was eligible for selection.
- [x] Update `MODELS.md` with the ladder and decision. The HF mirror is
      intentionally unchanged because no new variant was selected.

**Verification:** Chosen variant's receipts checked into
`validation/results/`; A14 composed p99 measured against the 40 ms budget
(pass, or explicit tier decision for G4); token-mismatch and audio gates
green. Kill criterion: if no compressed variant survives the audio gate, ship
uncompressed and let G4 resolve as the reservoir tier — do not trade audible
quality for the budget.

**Verification (2026-07-18): PASS — negative result.** The exporter ladder
produces int8-linear, 6-bit palettized, and 4-bit palettized variants for both
temporal and depth. All six remain finite and reduce package bytes to roughly
50%, 38%, and 25% of their respective baselines, but all six fail their
declared deterministic-reference gate. Temporal correlation falls from the
uncompressed 0.999312 to 0.997328, 0.976950, and 0.887407; max absolute error
rises from 2.135 to 4.585, 13.212, and 23.949. Depth argmax mismatch rates rise
from the existing FP16 baseline's 0.170 to 0.463, 0.687, and 0.940. The
machine-readable early-stop receipt is
`validation/results/MRT2WeightCompressionLadder.{json,md}`. Therefore the
system retains the uncompressed streaming temporal artifact and existing FP16
depth artifact. No causal speedup or audio-equivalence claim is made.

---

### Phase 3: Thermal Soak and Audio Integrity (Gates G2, G3)

**Goal:** The corrected, ANE-resident, weight-reduced pipeline passes the
10-minute foreground soak on A17 Pro at ≥1.0× real time with 0 underruns,
and the clicking/stutter/dropout defects are root-caused and closed.

**Tasks:**

- [ ] Root-cause the periodic clicking/stutter (Crossfade): candidate causes
      already implicated are temporal state handling and chunk-boundary
      buffering (decoder lookahead carry). Write-notes per defect; regression
      test per fix (bit-level chunk-boundary continuity check on rendered
      PCM).
- [ ] Controlled soak matrix using the existing soak instrumentation
      (Crossfade `e00faff`): {A17 Pro, A14} × {screen on foreground} ×
      {ANE placement, GPU control arm} × 10 min. Record thermal state
      timeline, ring depth, underruns, per-frame p50/p99 over time.
- [ ] If A17 Pro soak still fails after Phases 1–2: measure the residual —
      UI Metal load (test with minimal UI), decoder cadence, producer
      scheduling — before touching the model again. The duty-cycle math says
      ANE + fewer bytes should clear it; verify rather than assume.
- [ ] A14 tier decision (G4): under budget → report as second passing device;
      not under budget → measure the reservoir tier honestly (startup
      latency, sustained zero-underrun proof) and report as a tier.

**Verification:** Soak receipts (JSON + md) checked in showing A17 Pro
≥1.0× real time for 600 s, 0 underruns, thermal timeline included; audio
gates pass on soak-run audio; defect notes closed with regression tests.

---

### Phase 4: Evaluation Campaign (everything the paper will state)

**Goal:** One receipted measurement matrix, with dispersion, covering every
number in the claims ledger. No number enters the paper that is not in this
matrix.

**Tasks:**

- [ ] Latency: per-stage and composed p50/p90/p99, both phones, N≥5 repeat
      runs per cell for dispersion (the *Surgical Inference* dispersion
      discipline).
- [ ] Placement: per-run placement-gate artifacts; Instruments Core ML traces
      (ANE interval tables + empty app GPU tables) archived for both phones.
- [ ] Power: paired ANE-vs-GPU Power Profiler captures **re-run on the
      corrected pipeline** (the existing pair predates the §6.6 fixes and the
      paper flags it; the sequel must not inherit that asterisk), plus
      duty-cycle analysis.
- [ ] Sustain: the Phase 3 soak matrix is the sustain evidence; add one
      long-form run (≥30 min, A17 Pro) if the venue's story benefits and
      thermal permits.
- [ ] Audio quality: blind audio-judge/embedding-adherence gates on
      corrected-pipeline audio vs. the MLX reference cluster, with known-bad
      controls; L/R correlation and prompt-adherence bands as in paper §6.6.
- [ ] Startup: cold-start to first audio (model load + compile + prime), both
      phones — a system paper gets asked this.
- [ ] Check all receipts into `validation/results/` (and mirror on HF);
      update `docs/validation-receipts.md` with a new "corrected composed
      pipeline" section superseding §0's open-items paragraph.

**Verification:** Every claims-ledger row has a receipt path; a skeptical
read of `docs/validation-receipts.md` finds no number without provenance.

---

### Phase 5: Write the Paper

**Goal:** Full draft → figures → LaTeX → PDF, in this repo under `paper/`.

**Tasks:**

- [ ] Outline against the claims ledger. Working structure: intro (the
      real-time contract), background (MRT2 + heterogeneous SoC, cite
      *Surgical Inference* for the admission findings), system design
      (pipeline, delivery architecture, host-owned state), deployment
      reality (admission fragility + falsification — the §6.7 story with its
      Phase 1 resolution), evaluation (Phase 4 matrix), negative results,
      related work, artifact statement.
- [ ] Draft in markdown first (`paper/draft.md`), then convert — reuse the
      kokoro-coreml toolchain verbatim: `tectonic`, `natbib`, `booktabs`,
      `\usepackage{float}` + `[H]` figure placement from day one (lesson
      learned), Okabe-Ito figure palette, figure sources under
      `paper/figures/src/`.
- [ ] Figures: system/dataflow diagram (TikZ), soak timeline (thermal state +
      ms/frame vs. time — the money figure), latency-per-stage bars both
      phones, paired power/duty-cycle chart, weight-bytes-vs-latency ladder
      from Phase 2.
- [ ] Related work: on-device audio generation, streaming codecs
      (SoundStream/EnCodec lineage), mobile inference systems, ANE
      deployment literature (share `refs.bib` ancestry with paper 1 where
      applicable).
- [ ] Cross-check no claim overlap with *Surgical Inference* — every shared
      finding is cited, not restated as a contribution.

**Verification:** `tectonic paper/main.tex` builds clean; figures placed
adjacent to referring prose (PyMuPDF page-location scan); every number in the
PDF greps back to a receipt.

---

### Phase 6: Artifacts, Audit, and arXiv Publication

**Goal:** Public artifact story executed, final audit passed, and the PDF plus
source bundle published on arXiv.

**Tasks:**

- [ ] Execute the Phase 0 artifact decision (publish runtime subset or state
      private status); update `README.md`, `MODELS.md`, HF mirror, and the
      paper's artifact appendix to match reality.
- [ ] Full consistency audit of the paper (the kokoro-coreml `audit`-style
      pass: number-vs-receipt grep, stale-marker sweep, figure/caption
      contradictions, abstract-vs-body claim parity).
- [ ] arXiv package: PDF metadata (`hyperref` pdftitle/author/keywords),
      source tarball if submitting TeX, license choice.
- [ ] Produce a compact reviewer packet (PDF, source, claims-to-receipts map,
      and artifact URLs) without reformatting to any specific venue.
- [ ] Update `docs/validation-receipts.md` §0 and this plan's status to
      Complete.

**Verification:** arXiv ID exists; the submitted source bundle rebuilds the
same PDF; repo docs and reviewer packet are self-consistent with it.

## Executable Memory

- Regression test (parity, no checkpoint needed):
  `python3 validation/validate_temporal_body.py --skip-pytorch --reference-npz fixtures/reference_temporal_unrolled.npz`
- Regression test (carry path): `python3 validation/validate_temporal_body_carry.py` against its checked-in receipt.
- Not testable by command: ANE placement and soak gates require the two
  physical phones; the manual proof is the checked-in per-run placement-gate
  artifact + soak receipt named in each phase.

## Success Criteria

### Hard Requirements (Must Pass)

- [x] G1: in-app placement gate green for the temporal stack on ≥1 phone
      across 10 consecutive cold launches (or the documented-preconditions
      fallback explicitly accepted by the user).
- [ ] G2: A17 Pro 600 s foreground soak, ≥1.0× real time, 0 underruns,
      receipt checked in.
- [ ] G3: audio-quality gates pass on corrected-pipeline audio; known-bad
      controls rejected; click/stutter defects closed with regression tests.
- [ ] G4: A14 reported as pass (<40 ms/frame p99) or as an explicit measured
      reservoir tier — one or the other, in the paper.
- [ ] Every number in the PDF traces to a receipt in this repo.
- [ ] No contribution overlap with *Surgical Inference* beyond citation.

### Definition of Done

- [ ] All gates green and receipted
- [ ] Paper PDF builds clean and passes final audit
- [ ] arXiv submission live
- [ ] arXiv source bundle independently rebuilds the published PDF

## Open Questions

### Resolved

- **Q:** Is the second paper the conversion findings at more depth?
- **A:** No — those are spent in *Surgical Inference* §6.3–6.5. The second
  paper is the composed real-time system; the findings are cited method.

### Resolved for this execution

- **Q:** Which conference and format should drive the paper?
- **A:** None for this execution. Publish a venue-neutral arXiv paper with
  10–12 pages of main text, references, and a compact reproducibility appendix.
  Judge it by expert scientific scrutiny, not compliance with a CFP.
- **Q:** Does any of Crossfade go public for the artifact statement?
- **A:** No runtime source extraction in this execution. Keep Crossfade
  private; publish exporters, validators, fixtures, paper source, model hashes,
  machine-readable result summaries, exact commands, and runtime pseudocode in
  this repo. State that the private runtime is available from the author for
  artifact review.

### Still open
- **Q:** Do we add a third, newer device (A18/A19-class) to the matrix?
- **Options:** Strengthens generality; costs a device and a full matrix
  column. Lean: only if one is already on hand by Phase 4.

## Modules

### Performance and Latency Budget

| Operation | Target (p99) | Current | Phase |
| --- | --- | --- | --- |
| Composed frame, A17 Pro | <40 ms sustained 600 s | ~29 ms p50 bounded; throttles ~min 5–7 | 1–3 |
| Composed frame, A14 | <40 ms (or explicit tier) | ~50 ms | 2–3 |
| Temporal stage, in-app | ANE-resident, ~15 ms (A14) | GPU, 12.2 ms (A17 Pro) | 1 |
| Cold start → first audio | measured, reported | unmeasured | 4 |

### Degradation and Rollback

- **Runtime kill switch (Phase 1):** temporal path selection behind a runtime
  flag; one flag flip restores the shipping `.cpuAndGPU` rolling path.
- **Phase 2 kill criterion:** no weight-compressed variant that fails the
  blind audio gate ships, period; fall back to uncompressed + G4 tier
  framing.
- **Paper-level fallback:** if G2 cannot be met after Phases 1–3, the honest
  pivot is a "bounded-session real time + measured thermal envelope" paper —
  weaker, still publishable. That decision goes to the user with the soak
  data in hand, not made unilaterally.

### Risks and Mitigations

- **ANE admission remains fragile in-app (highest risk):** blocks the
  GPU-free headline → Phase 1's falsification matrix is designed to convert
  even failure into paper content (deployment-reality section); fallback
  framing pre-agreed in the claims ledger.
- **Quantized temporal audibly degrades music:** blocks the A14 story →
  per-variant blind gates with known-bad controls; kill criterion above.
- **Thermal gate fails for non-model reasons (UI Metal load):** wasted model
  work → Phase 3 isolates UI load with a minimal-UI soak arm before touching
  the model.
- **Scope creep from the Aperture product:** paper stalls → Phases 1–3 land
  as narrow, numbered Crossfade plans; product features are out of scope
  here.
- **Paper bloat:** a venue-neutral draft can expand without discipline → keep
  the main narrative within 10–12 pages and move command/schema detail to the
  reproducibility appendix.

---

## Critical Reminder

> SIMPLER IS BETTER. The fix chain is exactly what the first paper's own
> findings predict: put the temporal stack back on the ANE, cut its weight
> bytes, and measure. Resist inventing new mechanisms before those two levers
> are exhausted.
