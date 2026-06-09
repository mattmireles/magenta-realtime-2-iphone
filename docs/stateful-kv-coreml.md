# Stateful KV Caches for Core ML on Apple Silicon & the ANE — Advanced Developer Field Guide

June 4, 2026

## Related Documentation

- **[Graph teardown](graph-teardown.md)**: Op-level map and state-tensor contracts
  for `mrt2_small` before conversion.
- **[RVQ codec decoder guide](rvq-decoder.md)**: SpectroStream decode —
  RVQ detokenization, causal-conv lookback, iSTFT (separate conversion project).
- **[Validation receipts](validation-receipts.md)**: Parity numbers and device
  evidence for the published exports.
- **[Core ML Tools — Stateful Models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)**:
  Official `StateType` / `make_state()` reference (GPU-oriented examples).

Port context: this project bets that `mrt2_small` can hold **25 Hz / 40 ms per
frame** on iPhone with the hot path on the ANE. Use this guide for the
transformer's **bounded KV cache** once the graph teardown confirms shapes and
state contracts. The published `MRT2TemporalBody.mlpackage` is the working
product of these patterns: a 1-frame stateful export with 48 fp16
`ct.StateType` KV buffers.

## TL;DR
- **Core ML stateful models (iOS 18 / macOS 15+, coremltools 8+) are the correct primitive for a persistent, in-place KV cache**, but the in-place benefit is most cleanly proven on the GPU; on the **Apple Neural Engine (ANE)** the classic "stateful, append-in-place" pattern is reported by practitioners as "a no go," and the production-proven pattern is a **fixed-size sliding-window cache built from `slice_update`/concat-and-slice with static shapes**, verified op-by-op with the Xcode Core ML performance report.
- **The state-update op is where everything silently breaks**: residual dynamic shapes, `view+transpose` before the cache write, non-power-of-two (or non-multiple-of-32) state dimensions, too many separate state tensors, and `MLComputeUnits.all` letting the scheduler bounce the hot path between engines are the dominant failure modes — all of which compile cleanly and produce correct numbers while quietly costing your 40 ms budget.
- **Prove residency, don't assume it**: use Netron to confirm true `State` I/O and a single `coreml_update_state`/`slice_update` (not a deep unroll or full-window recompute), the Xcode Core ML + Neural Engine instruments and per-op performance report to confirm ANE delegation, `powermetrics --samplers ane_power,tasks` to confirm ANE energy draw, and a p50/p99 frame-time histogram over thousands of steps to catch recompilation thrash.

---

## How to read this guide
The central use case is an **autoregressive transformer decoding frame-by-frame in real time** on iPhone/Apple Silicon, with a **fixed-size KV cache updated in place across many sequential predictions**, resident on the ANE, under a **~40 ms/step** budget, using **bounded/sliding-window attention** so the cache is a **fixed-size ring buffer** rather than a growing tensor. Everything below generalizes, but I keep returning to that scenario because it is the hardest variant: it forbids the growing-tensor shortcuts that the official Apple LLM examples (which target the Mac GPU) rely on.

A blunt orientation up front, because it determines your whole architecture:

- **Apple's official stateful KV-cache examples target the GPU.** The coremltools "Stateful Models" guide example and the "On Device Llama 3.1 with Core ML" article both explicitly use `compute_units=ct.ComputeUnit.CPU_AND_GPU` / target the M1 Max GPU. In WWDC24 session 10161, Apple's own demo of state benefit is a GPU result: Mistral-7B on an M3 Max finishes "around 5 seconds while the [non-state] left took approximately 8 seconds, that's a 1.6x speedup when using state." The Hugging Face Mistral blog states plainly that getting the most performance out of the Neural Engine "requires some additional adaptations," which they had not yet done.
- **On ANE, the constraints are different and stricter.** Static shapes only, channels-first 4D `(B, C, 1, S)` layout preference, a ~32 MB on-chip SRAM working-set cliff, and ops like `concat`/`transpose` are expensive memory operations. Stephen Panaro's practitioner writeup is explicit that a long-lived append-in-place K matrix is "a no go on ANE," and that the working pattern is a **sliding window with a single concat per attention block**.

So this guide treats "stateful Core ML model" and "runs the cache hot path on the ANE" as **two separate goals that can conflict**, and tells you how to get both where possible.

---

# PART 1 — IMPLEMENTATION REFERENCE

## 1. The coremltools stateful API and the deployment floor

### Version floor (verified)
- Stateful Core ML models require **iOS ≥ 18.0, macOS ≥ 15.0, and Xcode ≥ 16.0** for the on-device `MLState` runtime; the feature is only available for the `mlprogram` model type. This floor is stated in the [coremltools Stateful Models guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html) and corroborated by the ExecuTorch Core ML backend docs (which gate `take_over_mutable_buffer`/`MLState` on those exact versions).
- State support landed in **coremltools 8.0**; the release notes describe "Updates to the converter to produce Core ML models with the State Type (new type introduced in iOS18/macOS15)" and a "toy stateful attention example model to show how to use in-place kv-cache."
- At conversion you must pass `minimum_deployment_target=ct.target.iOS18` (or `macOS15`). The two are the same conversion target.

### The three building blocks
1. **`ct.StateType`** — wraps a `ct.TensorType`; the `name` must match the PyTorch `register_buffer` name (or the `named_buffers()` key) exactly.
2. **`StateType` only carries a fixed `wrapped_type` shape** — states are static-shaped. (This interacts badly with flexible inputs; see Gotchas.)
3. **MIL-level ops**: `mb.read_state`, `mb.coreml_update_state(state=..., value=...)`, and `StateTensorSpec`. The PyTorch path emits these for you when it detects in-place buffer mutation; you can also author them directly in MIL.

### Minimal correct end-to-end conversion (accumulator)
This is the canonical smallest stateful model, from the coremltools guide, verified working:

```python
import numpy as np, torch, coremltools as ct

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("accumulator", torch.tensor(np.array([0], dtype=np.float16)))
    def forward(self, x):
        self.accumulator += x            # in-place mutation -> becomes state write
        return self.accumulator * self.accumulator

traced = torch.jit.trace(Model().eval(), torch.tensor([1]))
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=(1,))],
    outputs=[ct.TensorType(name="y")],
    states=[ct.StateType(wrapped_type=ct.TensorType(shape=(1,)), name="accumulator")],
    minimum_deployment_target=ct.target.iOS18,
)
```

### Authoring state directly in MIL (when you need exact control over the update op)
```python
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types

@mb.program(input_specs=[mb.TensorSpec((1,), dtype=types.fp16),
                         mb.StateTensorSpec((1,), dtype=types.fp16)])
def prog(x, accumulator_state):
    v = mb.read_state(input=accumulator_state)
    y = mb.add(x=x, y=v, name="y")
    mb.coreml_update_state(state=accumulator_state, value=y)
    return y

mlmodel = ct.convert(prog, minimum_deployment_target=ct.target.iOS18)
```
Writing MIL directly is the most reliable way to guarantee the graph contains exactly one `coreml_update_state` per cache and nothing else — useful when the PyTorch tracer keeps inserting copies.

### Python-side prediction & state inspection
```python
state = mlmodel.make_state()
print(mlmodel.predict({"x": np.array([2.0])}, state=state)["y"])   # (2)^2
print(mlmodel.predict({"x": np.array([5.0])}, state=state)["y"])   # (5+2)^2
# Inspect / reset:
state.read_state(name="accumulator")
state.write_state(name="accumulator", value=np.array([1.0]))
```
**Verification trap (from the guide):** because each `predict` mutates the state, naive numerical comparison against the PyTorch model will diverge after the first call. Reset state (or rebuild the torch buffer) before each comparison.

---

## 2. The PyTorch → Core ML path for KV-cache state

### The decisive design choice: slice-assignment vs concat
There are two ways to express a KV cache that the converter treats very differently:

**(A) Growing-tensor / I/O cache (defeats the purpose):** cache passed in as input, concatenated with new K/V, returned as output, copied back by the driver. This compiles fine and is correct but does multiple data copies per layer per step. Apple's own measurements on Llama-3.1-8B at context 2048: the "KV cache as model I/O" path gives `[Extend] => 100 tokens, throughput: 1.25 tokens/s`, versus `[Extend] => 99 tokens, throughput: 16.26 tokens/s` for the state path — Apple states "the performance is improved ~13 times, compared to key-value cache as I/O for 2048 context size." The copies dominate because, per Apple, "With Llama3.1 8B model, this can go up to ~1GB when using 8192 context size, which would result in huge overhead to copy."

**(B) In-place slice update (the goal):** pre-allocate the full cache as a `register_buffer`, and write new K/V into a slice with `cache[..., begin:end, :] = new_kv`. coremltools detects this slice-update pattern and emits a true in-place state write.

Apple's `SliceUpdateKeyValueCache` (from the On-Device Llama article) is the reference idiom. The update logic is just slicing — "these op patterns are then detected by the Core ML-GPU compiler and allows it to perform in-place updates":

```python
from transformers.cache_utils import Cache
import torch

class SliceUpdateKeyValueCache(Cache):
    def __init__(self, *, shape, dtype=torch.float32):
        super().__init__()
        self.past_seen_tokens = 0
        self.k = torch.zeros(shape, dtype=dtype)   # (layers, batch, kv_heads, ctx, head_dim)
        self.v = torch.zeros(shape, dtype=dtype)
    def update(self, k_state, v_state, layer_idx, cache_kwargs=None):
        pos = cache_kwargs.get("cache_position")
        begin, end = self.past_seen_tokens, self.past_seen_tokens + pos.shape[-1]
        self.k[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state   # slice write
        self.v[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state
        return self.k[layer_idx, :, :, :end, :], self.v[layer_idx, :, :, :end, :]
    def get_seq_length(self, _=0): return self.past_seen_tokens
```

For concreteness, Apple gives the exact cache shape for Llama-3.1-8B at context 2048: each of Key and Value is `(32, 1, 8, 2048, 128)` — "32 attention blocks... 8 key/value heads... size of the projections are 128" (the layout is `(layers, batch, kv_heads, ctx, head_dim)`). The wrapper registers `self.kv_cache.k`/`.v` as buffers named `keyCache`/`valueCache` so they map to `ct.StateType` by name. The toy single-layer version from the coremltools guide uses exactly the same `cache[:, past:end, :] = newly_computed` slice-assignment.

### Trace vs export
- **`torch.jit.trace` is the mature, recommended path.** Apple's On-Device Llama article uses trace and notes "The newer `torch.export` path is still in beta and possibly needs minor changes to the Llama model definition to export." coremltools 8.0 release notes put `torch.export` op coverage at "56% parity with our mature torch.jit.trace converter."
- **`torch.export` supports stateful buffers too** (coremltools release notes include a `state_1.mul_(x)` `torch.export` example) and preserves dynamic-shape metadata more cleanly via `torch.export.Dim`, but for a production KV cache today, trace is lower-risk.
- **Tracing caveat:** the tracer's graph optimizer eliminates no-op arithmetic (e.g. `+ 0`), which matters for a value-cache workaround documented in Part 3 — you must add a value small enough to truncate to zero in fp16 but nonzero in fp32 so the tracer keeps the op.

### Make sure it's TRUE in-place, not copy-in/copy-out
The single most important verification: open the `.mlpackage` in **Netron** (or inspect the MIL) and confirm the cache appears as a **`State`-typed input** and that the graph contains a `coreml_update_state` (GPU/MIL) or a `slice_update` writing back to that state — *not* the cache appearing as both a plain tensor input and a tensor output. If it's the latter, you built pattern (A) and got zero benefit despite a clean compile.

---

## 3. Fixed-window / sliding-window attention as a ring buffer — the crux

This is where ANE deployment diverges hard from the Apple GPU examples.

### Why the "obvious" ring buffer fails on ANE
The intuitive ring buffer — keep one persistent K matrix, overwrite the oldest slot, advance a write pointer — relies on either (a) a stateful in-place scatter into a rotating index, or (b) `roll`. On ANE:
- A long-lived append/overwrite-in-place K is reported as "a no go on ANE" by Panaro.
- `concat` and `transpose` are **memory operations that are slow** and frequently trigger CPU/GPU fallback or buffer copies; Apple's ANE-transformers article stresses minimizing them (the reference implementation incurs exactly **one transpose**, on K, and avoids all reshapes).
- Fully dynamic ops (dynamic reshape, dynamic gather, dynamic slice with a runtime index) drop off the ANE. The coremltools FAQ notes that converting a static reshape to a "fully dynamic reshape" is exactly the kind of thing that knocks a model off the ANE.

### Candidate implementations, ranked

| Formulation | How the window slides | ANE-friendly? | Notes |
|---|---|---|---|
| **Concat-and-slice (Panaro sliding window)** | New 64-token K computed; concat with 448 next-newest (passed as input, already transposed) → 512; oldest 64 discarded by a *second* model when ANE is idle | **Yes (production-proven)** | Exactly one concat per attention block; cache slid by a secondary model so the main model never pays for it during generation |
| **`slice_update` into static buffer (Apple `SliceUpdateKeyValueCache`)** | `cache[..., begin:end, :] = new` | **Conditional** | Detected as in-place on GPU; on ANE works as a stateful write only if state dims meet alignment (see Part 3) and shapes are fully static |
| **Multiple fixed K-chunks, drop-oldest/add-newest** | Pop oldest 64-chunk, push new chunk, concat 8 chunks | **No (slow)** | Zero-cost slot swap but the 8-way concat is "very slow"; Panaro reports this is dead |
| **`roll` / fully dynamic scatter ring** | Shift whole tensor by one each step | **Usually falls back** | `roll`/dynamic-index scatter tend to go to CPU/GPU; measure before trusting |
| **Full-window recompute (no cache)** | recompute all K/V every step | **Runs but wasteful** | Apple baseline Llama-8B at ctx 2048: `[Extend] => 100 tokens, throughput: 0.19 tokens/s`, TTFT 5374.15 ms — the thing you're trying to escape |
| **Lazy/streaming softmax (no concat)** | multiply each K chunk by Q, combine with running stats | **No (slow)** | Trades concat for 7× adds; Panaro found it slower |

### The production pattern (sliding window + secondary cache-update model)
Panaro's design, which he notes is "largely the same as one [Apple] use in their own models":

1. **Pick static Q and K sizes.** Panaro: "We'll give K 512 rows… we'll split the difference and give Q 64 rows… It will take at most 8 calls to process a full 512 token prompt (8\*64=512)… 64 is also a multiple of 8, which aligns with the ANE hardware." 64 balances prefill (≤8 calls for a 512-prompt) against generation waste.
2. **Pass the trailing 448 K/V as *inputs*, already transposed.** The main model does exactly **one concat** (448 + new 64 → 512) and **transposes only the new 64**, which is cheap. Return only the new 64 K (transposed) and 64 V.
3. **Slide with a second model.** A secondary model combines old 448 + new 64 → updated 448, run *when the ANE would otherwise be idle*. During generation you only slide once per 64 tokens; during prefill you slide between every call.
4. **Store each layer's K/V cache as a separate tensor** (Llama-7B has 32), taken in and returned individually, to avoid the concatenations that a single packed cache tensor would require.

Panaro reports: "For example I have a Llama 2 7B model that saw approximately a 4x speedup" versus a cache-less model that processes the same number of tokens.

**Counterpoint for the truly fixed sliding window (your use case):** if your window is genuinely bounded (e.g. last N frames) and you never need prefill of long prompts, you can collapse to a single static graph where the cache is exactly the window size and the "evict oldest, append newest" is a masked overwrite plus a position-shifted causal mask. Whether this stays on ANE depends entirely on the update op — measure it.

---

## 4. Positional encoding under a sliding window

The ring buffer makes **absolute positions ambiguous**: physical slot 0 in the buffer is not logical position 0 of the sequence. Get this wrong and the model degrades subtly (often only on long runs), which is brutal to debug.

- **RoPE (rotary, the modern default):** RoPE encodes position as a phase rotation applied to Q and K; the attention dot product depends on the *relative* offset (i−j), not absolute index. This is exactly what you want for a sliding window: as long as you rotate Q and each cached K by their **true logical position** (a running `past_seen_tokens` + offset, not the physical slot), relative geometry is preserved. **Trap:** if you rotate by physical buffer slot, relative distances wrap when the buffer wraps and attention corrupts. Precompute the cos/sin tables for the full max logical range and index them by logical position.
- **Caching cos/sin:** precompute the rotation tables once (they're expensive to compute per step) and gather the rows for the current logical positions. Keep this a static gather (fixed shape) so it stays ANE-eligible.
- **ALiBi (linear bias):** adds a slope·distance bias to the attention scores; distance is again relative, so it composes naturally with a sliding window as long as the bias is computed from logical distance. It's a fixed additive mask — cheap and ANE-friendly — but you must regenerate/shift the bias matrix as the window slides.
- **Learned absolute positions:** the worst fit for a ring buffer. A learned absolute table assumes a fixed mapping from index → embedding; under a sliding window you must add the embedding for the **logical** position before the token enters the cache (so the cached K already carries it), never re-add per step. The vision-transformer ANE article specifically prefers an added absolute-position table at the input over relative-position-in-attention-matrix schemes for ANE efficiency (it grows linearly, not quadratically, with token length).
- **General correctness check:** the cleanest invariant is "the cache stores K/V *with* position already applied at the correct logical index." Then the sliding window is pure data movement and positional correctness is decoupled from eviction.

---

## 5. Window warmup / cold start before the cache fills

Two viable approaches; for ANE prefer masking within the fixed shape.

- **Masking within the fixed shape (recommended for ANE):** allocate the full fixed cache up front (zeros), and use the causal/attention mask to make positions that haven't been filled yet contribute −inf (so softmax zeroes them). This keeps a single static shape — the ANE's hard requirement — across warmup and steady state. This is exactly how Apple's examples work: the cache is zero-initialized and `causalMask` carries `-inf` in not-yet-valid regions; `make_state()` returns a zeroed buffer.
- **EnumeratedShapes (use sparingly):** you *can* declare a small set of shapes (e.g. progressively larger prefill chunks) but this interacts badly with both ANE and state (see Gotchas). The documented behavior: an `mlprogram` with EnumeratedShapes runs on the ANE **only with the default shape**; other enumerated shapes fall to GPU/CPU. And mixing enumerated + range flexibility raises `EnumeratedWithRangeInputsException`. For warmup, masking is almost always the better tool.
- **Prefill vs decode as two functions:** the more advanced pattern (Anemll, SqueezeBits, CoreML-LLM) is a **multifunction model** sharing weights, with a fixed-shape `prefill` function (e.g. Q=32 or 8-batch) and a fixed-shape `decode` function (Q=1), packaged together so warmup uses one function and steady-state the other — both static.

---

## 6. The Swift runtime (MLState, the decode loop, buffer lifetime)

### The API surface (verified)
Stateful prediction is gated to `macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0`. Apple's own `MLState` doc page describes it tersely as "Handle to the state buffers" and `makeState()` as "Creates a new state object."

Verified signatures/usage (from Apple's SDK header and Apple-engineer-contributed swift-transformers code):

```swift
import CoreML

// Create one long-lived state per decode session:
let model = try MyModel(configuration: config).model   // MLModel
var state: MLState? = model.makeState()                 // -> MLState

// Stateful prediction (async form contributed by Apple in swift-transformers):
let outputs = try await model.prediction(from: inputProvider, using: state!)

// Underlying Obj-C method that Swift refines (from MLModel+MLState.h):
//   - prediction(from:using:)            // MLFeatureProvider + MLState
//   - prediction(from:using:options:)    // + MLPredictionOptions
//   API_AVAILABLE(macos(15.0), ios(18.0), watchos(11.0), tvos(18.0))
```

> **Uncertain / verify in Xcode Quick Help:** the exact Swift method on `MLState` for reading/writing a named buffer (the Swift analog of Python's `read_state`/`write_state`; likely `withMultiArray(for:_:)` / `withMultiArrayForState:`) could not be confirmed from an accessible primary source because Apple's doc page is JS-rendered. Treat the Swift state-buffer-accessor name as **unverified** and confirm against Xcode 16's generated interface before relying on it.

### The decode loop — lifecycle rules
- **Allocate state once per session, reuse across the whole decode loop.** Do **not** call `makeState()` per step — that throws away the cache. swift-transformers resets via `state = model.makeState()` only on an explicit `resetState()` (e.g. new conversation/stream).
- **State is passed by reference and mutated in place;** you don't read it back between steps for normal decode.
- **Reset = new state** (or `write_state`/zero the buffers) when you start a fresh stream or flush the window.

### MLMultiArray buffer lifetime & zero-copy
- When you wrap an existing buffer (e.g. `MLMultiArray(dataPointer:shape:dataType:strides:deallocator:)` or a `reshaped(to:)` that aliases storage), the new array **points into the original's memory and does not copy** — "The caller is responsible for keeping the original object alive, for example using `withExtendedLifetime(originalArray) {...}`." Failing to do so is a use-after-free that manifests as garbage outputs or crashes under ARC.
- For a real-time loop, pre-allocate input `MLMultiArray`/`MLTensor` buffers once and overwrite in place each frame rather than allocating per step; allocation churn shows up directly in your p99.
- `MLTensor` (iOS 18) is the high-level path: it provides `softmax`, argmax, slicing etc. out of the box and removes a lot of fragile manual pointer code, at some abstraction cost. swift-transformers' `preview` branch uses it.

### Threading for a real-time loop
- Core ML prediction is synchronous on the calling thread unless you use the async API. For a hard 40 ms/frame budget, run prediction off the main thread on a dedicated serial queue (or use `async`), and **keep one model + one state pinned to that queue** — `MLState` is a handle to live buffers and is not something to share across concurrent predictions.
- Prewarm: the first prediction after load triggers device specialization/compilation (the "first run on a device may take a while" you see with whisper.cpp's Core ML encoder). Do a dummy prediction (or per-chunk dummy predicts, as CoreML-LLM does at load — ~231 ms total reported) before the real-time loop starts so the first real frame isn't a multi-hundred-ms outlier.

---

# PART 2 — DEBUGGING & VERIFICATION REFERENCE

You cannot trust "it compiled" or even "it's fast on my Mac." Prove each property.

## 7. Prove the graph is what you think (Netron / MIL inspection)
Open the `.mlpackage`'s `model.mil` (or use Netron) and check:
1. **The cache is a `State` input**, e.g. `%k_cache: (1, 16, 128, 128, fp16)(State)` — not a plain tensor that also appears as an output.
2. **Exactly one update per cache**: a single `coreml_update_state` (or `slice_update`) writing back, per layer. Many updates, or a `concat`+full-tensor output, means copy semantics.
3. **No deep unroll**: the attention block appears once per layer, not unrolled across time steps. A graph that ballooned in op count is a sign the tracer captured a Python loop over positions.
4. **No full-window recompute**: K/V projections should be computed only for the new tokens, not the full window.
5. **No residual dynamic ops** on the hot path: dynamic reshape/gather/slice are red flags for CPU fallback.

## 8. Prove ANE residency (Xcode + instruments + powermetrics)

**Xcode Core ML performance report (primary tool):** add the `.mlpackage` to a project, open the **Performance** tab, run on a device with an ANE. It shows **per-operation compute-unit delegation** (ANE vs GPU vs CPU) and estimated cost. This is the authoritative "is this op on the ANE" check. SqueezeBits' workflow: confirm "nearly all operations within both the prefill and decode phases are executed on the ANE with only a few exceptions, namely the token embedding layer (gather), the LM head (linear), and a few minor element-wise operations." A state-update op showing up on CPU here is your bug.

> **Known constraint:** the Xcode performance report itself *fails to generate* for a stateful model whose state width violates ANE alignment, with "There was an error creating the performance report / Unable to compute the prediction using ML Program." That failure is itself a diagnostic (see Part 3, alignment).

**Xcode Instruments — Core ML + Neural Engine templates:** profile the running app to see Core ML API calls, when/where work is dispatched to each engine, and Neural Engine activity. Use the **Core ML instrument** to find recompilation and per-prediction timing; pair with the **Neural Engine** instrument.

**powermetrics (device-level confirmation of ANE energy):**
```bash
# Per-process energy + ANE power (run while your loop is hammering):
sudo powermetrics --samplers tasks --show-process-energy
# ANE power specifically (grep for "ANE Power"):
sudo powermetrics --samplers ane_power -i 200
# Broad view:
sudo powermetrics --samplers tasks,ane_power,gpu_power -i 1000
```
If the ANE power line stays at ~0 mW while your model runs, you are not on the ANE regardless of what `computeUnits` you requested. Caveat from Apple's own docs: powermetrics power values are **estimates**, fine for "is the ANE doing work / relative optimization," not for cross-device comparison.

**LLDB breakpoint trick (last resort):** pause the running app; if a thread named **`H11ANEServicesThread`** exists, Core ML is using the ANE for at least part of the model. Conversely landing in `Espresso::...kernel_cpu::...` frames indicates CPU execution. Core ML is a black box and may split a model across engines, so this only tells you "some of it is on ANE."

**The cheap differential test:** run the same model with `.cpuAndNeuralEngine` vs `.cpuAndGPU` vs `.cpuOnly`. If `.all`/`.cpuAndNeuralEngine` is not meaningfully faster than `.cpuOnly`, the ANE isn't really being used for the hot path.

## 9. Measure the right latency distribution
- Measure **p50 and p99 per-step latency over thousands of sustained steps**, not a mean over 100. Real-time failures live in the tail. CoreML-LLM-style reporting (e.g. second-turn TTFT) is the right granularity.
- **Detect recompilation thrash:** a recurring latency spike every N steps, or whenever a shape changes, indicates Core ML re-specializing the model. Causes: changing input shapes (flexible shapes), state shape surprises, or losing the `mlmodelc` cache. Use the Core ML instrument to see compile events; a stable real-time loop should show **zero** compiles after warmup.
- **mlmodelc caching:** in Python, each new process re-specializes because the temporary `mlmodelc` path changes. For repeatable benchmarking, compile to a fixed `mlmodelc` location once and load from there, or use `CompiledMLModel`. On device, ship/compile once and reuse.
- **Optimization hints:** when loading, you can pass `optimization_hints={"reshapeFrequency": ct.ReshapeFrequency.Infrequent}` (or `.Frequent`) and `{"specializationStrategy": ct.SpecializationStrategy.FastPrediction}`. `FastPrediction` trades load time for lower per-prediction latency; `Infrequent` reshape frequency tells Core ML you won't be changing shapes — appropriate for a fixed ring buffer. (Note: per coremltools #2370, the "Frequent" hint did not reliably move flexible-shape models onto the ANE — don't rely on it as a fallback for static shapes.)

---

# PART 3 — FAILURE MODES, GOTCHAS, AND PRACTICES

## 10. Hidden gotchas (the ones that compile clean and cost you silently)

### G1 — Silent CPU fallback on the state-update op
The state read is usually fine; the **write/update** is what falls off the ANE. Dynamic indices, `view+transpose` before the write, or unsupported scatter formulations move just that op to CPU, forcing an ANE→CPU→ANE round trip every step. Hidden because numbers are correct and the rest of the model is on ANE. **Detect:** per-op performance report; the update op's device column.

### G2 — Copy-in/copy-out masquerading as in-place state
Builds fine, runs, no speedup. The cache is secretly pattern (A). **Detect:** Netron shows the cache as both tensor input and tensor output, no `coreml_update_state`. **Fix:** use slice-assignment that the converter recognizes, or author the `coreml_update_state` in MIL.

### G3 — `view` + `transpose` before the cache write breaks state binding
coremltools issue #2275: a `view`+`transpose` combination before the cache update caused the converted model to fail at runtime with "MIL program input, 'k_cache', not found in Core ML model inputs" (worked with single-token input, broke with batch input). **Fix:** reshape/transpose *after* the state write, or restructure so the write targets the buffer directly.

### G4 — State must be "touched" by a real op
coremltools issue #2548: a stateful MIL program can fail to build unless you perform a non-trivial op on the state. Adding **zero** doesn't work because the tracer/graph optimizer eliminates it. SqueezeBits hit the same thing on value-cache updates in the prefill graph: the fix is to add a constant small enough to truncate to zero in fp16 but nonzero in fp32 — e.g. `torch.finfo(torch.float32).smallest_normal` — right before the value-cache update, so the op survives tracing but is numerically a no-op after fp16 casting. ("This trick is only required for value cache updating in the prefill graph.")

### G5 — Recompilation thrash from residual dynamic shapes
Any dimension that varies at runtime can trigger re-specialization. **RangeDim runs on the ANE only for the default shape** (Apple Developer Forums + coremltools #2370); other shapes drop to GPU/CPU and may recompile. For a fixed ring buffer you should have **no** dynamic dims on the hot path. **Detect:** Core ML instrument compile events; p99 spikes correlated with shape changes.

### G6 — float16 vs float32 dtype mismatches and silent casts
The ANE is **FP16-only**. The ExecuTorch Core ML docs state plainly: "the ANE only supports FP16 precision," and Core ML applies FP16 compute regardless of the PyTorch dtype by default. If your traced model has FP32 buffers/inputs while the rest is FP16, Core ML inserts casts; a cast on the hot path can bounce execution between engines or just burn time. There is also a documented case (coremltools #561) where converting a model to FP16 *prevented* ANE use due to a memory-alignment quirk. **Practice:** declare cache `StateType`, inputs, and outputs as `np.float16` end to end; verify no `cast` ops sit on the per-step path in Netron.

### G7 — The ANE SRAM / tensor-size cliff (~32 MB)
Independent reverse-engineering (maderix; the "Orion" arXiv preprint) characterizes a **~32 MB on-chip SRAM working-set cliff**: a 2048×2048 FP16 matmul (~24 MB working set) hits peak (~5.7 TFLOPS in those benches), while 4096×4096 (~96 MB) drops throughput ~30% as it spills to DRAM. Apple's ANE-transformers article frames the same pressure: keep tensors in L2/on-chip; an improperly singleton last axis gets padded to 64 bytes, blowing up buffer size 32× (FP16) / 64× (INT8) and evicting from cache. **Implication for cache sizing:** a KV cache (× layers × heads × window × head_dim × 2) that exceeds on-chip budget per step costs you cross-compute-unit transfer and DRAM bandwidth. Apple's GPU Llama data shows the analogous effect: beyond context 2048 the cache "becomes too large to consistently fit within the GPU cache," dropping decode speed. **Practice:** size the window and per-step working set to stay under the cliff; prefer smaller per-call K/V chunks (Panaro's 64) and per-layer separate caches. (Caveat: the maderix/Orion numbers are M4 reverse-engineering, author-reported, not Apple-documented — treat as indicative.)

### G8 — `.all` can be actively harmful
`MLComputeUnits.all` lets Core ML split the model and dispatch per-op across CPU/GPU/ANE by its own heuristic. For a real-time, hot, stateful loop this can repeatedly bounce the cache between engines, and ANE↔CPU transitions in particular are cheap enough that Core ML will do them — adding transfer overhead every step. **Practice:** for an ANE-targeted cache, set `config.computeUnits = .cpuAndNeuralEngine` to keep the GPU out of the decision and force the scheduler toward ANE (CPU only for genuinely unsupported ops like the final gather/LM-head). Benchmark `.cpuAndNeuralEngine` vs `.all` explicitly; don't assume `.all` wins. (For the *Mac GPU* path, by contrast, `.cpuAndGPU` is what Apple's own LLM examples use.)

### G9 — Too many separate state tensors fail to compile for ANE
SqueezeBits: "if your model has too many states, it could fail to compile for the ANE." Their fix for Qwen3-0.6B: implement custom HF `Cache`/`CacheLayerMixin` interfaces to store all layers' K/V as a **single concatenated tensor** of shape `(num_hidden_layers, num_key_value_heads, max_cache_len, head_dim)` (batch=1), reducing **56 states → 2 states** (`key_cache`, `value_cache`). Note this **conflicts with Panaro's GPU-idle "separate per-layer cache"** advice — the right answer depends on engine and state-count limits; measure. (They also warn this single-tensor approach needs more work for heterogeneous/sliding-window layers like Gemma-3.)

### G10 — State dimension alignment (power-of-two / multiple-of-32)
Two corroborating reports:
- **SqueezeBits:** if any state dimension *other than the first* is not a power of two, the model "might encounter a runtime error." Example: EXAONE-3.5-2.4B state shape `(20, 8, 512, 80)` — 80 isn't a power of two — fails with `"Unable to compute the prediction using ML Program. It can be an invalid input data or broken/unsupported model."` Fix: **pad to `(20, 8, 512, 128)`** and use only the unpadded part. Recommend the cache interface internally pad non-first dims to the next power of two.
- **Apple Developer Forums (anecdotal, third-party):** a stateful `mlprogram` "fails to run on the Apple Neural Engine (ANE) if the state tensor dimensions (specifically the width) are not a multiple of 32." `(1,3,480,270)` fails on ANE; `(1,3,480,256)` succeeds. Works on CPU/GPU either way.

These are consistent: **pad the trailing state dims to a power of two / multiple of 32**, slice to the logical size inside the graph. This is also why the Xcode performance report can fail to generate for a stateful model.

### G11 — `Failed to build the model execution plan` (error code 14 / -14)
A recurring, generic stateful-conversion failure (coremltools #2325, #2548, SqueezeBits). Known triggers and fixes: state not touched by a real op (G4); fixed + flexible inputs mixed with state (G5/below); SDPA memory blowup at long sequence lengths — SqueezeBits' fix is the otherwise-unused MIL pass **`scaled_dot_product_attention_sliced_q`** (params `min_seq_length`, `seq_length_divider`) added to the conversion pipeline, which chunks Q in SDPA. coremltools 8.x added a related `common::scaled_dot_product_attention_sliced_q` transformers pass reported as "34% faster and used 45% less memory on ANE" for a long-sequence model (Depth-Anything, sequence length 1814).

### G12 — Fixed + flexible inputs + state don't mix
coremltools #2548: a model can't mix enumerated and range flexibility (`EnumeratedWithRangeInputsException`), and stateful MIL with both fixed and flexible inputs present can fail to build. For a ring buffer, keep **everything static**, which sidesteps this entirely.

### G13 — Multi-output / multi-token-per-step interactions
If you emit multiple tokens or multiple codebook levels per step (multi-token prediction, [audio codec levels](RVQ-codec-decoder-guide.md)), each output and any per-output cache write multiplies the state-update surface. Panaro notes the 64-token recompute means multi-token prediction is "basically free" — but each extra write must independently satisfy G1/G4/G10. Verify every update op's device placement, not just the first.

### G14 — Prefill state not fully updated (batch path)
An Apple Developer Forums report: during prefill (multi-token batch into the model in one call) some KV-cache entries are not updated, while single-token decode updates normally. If you use a multifunction prefill path, validate that the prefill function's state write covers the whole batch slice — don't assume decode-path correctness transfers.

## 11. Best practices (consolidated)
- **Cache sizing:** keep per-step working set under the ~32 MB SRAM cliff; per-layer caches sized to the bounded window only. Pad trailing dims to power-of-two/multiple-of-32, slice logically.
- **dtype:** FP16 end to end (inputs, outputs, state). The ANE is FP16-only; eliminate hot-path casts.
- **Layout:** prefer ANE-friendly **channels-first 4D `(B, C, 1, S)`** with the **sequence/window axis last** (last axis must be contiguous, 64-byte aligned, and is *not* packed — don't make it a singleton). Swap `nn.Linear`→`nn.Conv2d` (1×1) where chasing peak ANE; the ANE is fundamentally a convolution engine and 1×1 convs often beat equivalent matmuls.
- **Window strategy:** static sliding window with one concat per attention block (Panaro), or `slice_update` into a static aligned buffer; slide via a secondary model when the ANE is idle.
- **Chunking:** split multi-head attention into per-head/chunked ops (ANE-transformers Principle 2) for L2 residency; chunk Q in SDPA via `scaled_dot_product_attention_sliced_q` for long windows.
- **Compute units:** `.cpuAndNeuralEngine` for the ANE cache path; `.cpuAndGPU` only if you're deliberately targeting the Mac GPU like Apple's LLM examples.
- **Prefill/decode:** multifunction model with fixed-shape prefill + fixed-shape decode sharing weights.
- **Prewarm** with dummy predictions before the real-time loop; pin model+state to one serial queue.

## 12. Worst practices (do not do these)
- Calling `makeState()` inside the decode loop (throws away the cache).
- Letting the cache be a plain input+output tensor and calling it "stateful."
- Leaving any dynamic dim on the hot path "for flexibility."
- Mixing FP32 buffers into an FP16 model and trusting Core ML to "handle it."
- Using `.all` for a real-time ANE loop without benchmarking against `.cpuAndNeuralEngine`.
- A single packed all-layers cache when per-layer separate caches avoid concats (or vice-versa) **without measuring** — the right choice is engine- and state-count-dependent.
- Comparing torch vs Core ML numerics without resetting state between calls.
- Benchmarking with a mean over 100 iterations instead of p50/p99 over thousands.
- Shipping without an Xcode per-op performance report confirming the update op is on the ANE.

---

## 13. Quick-reference checklists

### Implementation checklist
- [ ] `minimum_deployment_target=ct.target.iOS18` (or `macOS15`); coremltools ≥ 8; Xcode ≥ 16.
- [ ] Cache is a `register_buffer`; `ct.StateType.name` matches buffer name exactly.
- [ ] Update via slice-assignment / `coreml_update_state` (in-place), not concat+output.
- [ ] All shapes static; no RangeDim/EnumeratedShapes on the hot path.
- [ ] FP16 for inputs, outputs, and state; no hot-path casts.
- [ ] Trailing state dims padded to power-of-two / multiple-of-32; sliced logically.
- [ ] State count minimized (consider packed all-layers tensor if ANE compile fails).
- [ ] Positional encoding indexed by **logical** position, not physical slot.
- [ ] Warmup handled by masking within fixed shape (not a separate shape).
- [ ] `config.computeUnits = .cpuAndNeuralEngine` (ANE path).
- [ ] One `MLState` per session, reused; prewarm prediction before the loop.
- [ ] Input/output buffers pre-allocated; `withExtendedLifetime` around aliased MLMultiArray storage.

### Debugging checklist
- [ ] Netron: cache is `State`-typed input; exactly one update op per cache; no unroll/recompute.
- [ ] Xcode performance report: update op + attention on ANE (not CPU/GPU).
- [ ] If perf report fails to generate → suspect state-dim alignment (G10).
- [ ] `powermetrics --samplers ane_power,tasks --show-process-energy` shows nonzero ANE power.
- [ ] `.cpuAndNeuralEngine` meaningfully faster than `.cpuOnly` (else not really on ANE).
- [ ] p50/p99 over thousands of steps; no periodic spikes (recompile thrash).
- [ ] Core ML instrument shows zero compiles after warmup.
- [ ] Numerics verified against torch with state reset before comparison.

---

## 14. Source quality & uncertainty notes

### Works cited (primary)

1. [Core ML Tools — Stateful Models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html) — `StateType`, `make_state()`, slice-update KV cache.
2. [Core ML Tools — Flexible input shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html) — EnumeratedShapes / RangeDim ANE behavior.
3. [On Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama) (Nov 2024) — `SliceUpdateKeyValueCache`, GPU-targeted state path.
4. [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/apple-neural-engine-transformers) (2022) — ANE layout and concat/transpose costs.

- **Primary/official:** coremltools "Stateful Models" guide; Apple ML Research articles above; WWDC24 sessions 10159/10161; Apple Developer MLState/Core ML docs; coremltools 8.0 release notes & MIL op reference; ExecuTorch Core ML backend docs; coremltools Flexible Input Shapes guide.
- **Strong practitioner:** Stephen Panaro's ANE KV-cache writeup (sliding-window design, ~4× Llama-2-7B); Hugging Face Mistral Core ML blog; SqueezeBits disaggregated-inference blog (ANE stateful gotchas — state-count, power-of-two alignment, value-cache add-constant trick, SDPA slicing, all verbatim-quoted).
- **Reverse-engineered / treat as indicative:** maderix substack and the "Orion" arXiv preprint for the ~32 MB SRAM cliff, ~19 TFLOPS-vs-38-TOPS, dispatch overhead, and ~119 compiles-per-process limit — independent M4 measurements, not Apple-documented; directionally consistent with Apple's own cache-residency observations but specific numbers are author-reported.
- **Anecdotal (flagged):** the "state width must be a multiple of 32" Apple Developer Forums report (one user, corroborates SqueezeBits' power-of-two finding directionally); the prefill-state-not-fully-updated forum report.
- **Unverified API detail:** the exact Swift `MLState` buffer-accessor method name (Python's `read_state`/`write_state` analog) — Apple's doc page is JS-rendered; confirm in Xcode Quick Help.
- **Version-dependent:** ANE behavior with flexible shapes, the SDPA slicing passes, and several "Failed to build execution plan" triggers are coremltools/OS-version-specific and have shifted across 8.0→8.3 and iOS 18.x; re-verify on your exact toolchain. Note also reports of attention/SDPA correctness regressions on A14-class hardware under macOS/iOS 26 betas — test on your target OS.
