# Reference runtime source (not the shipping app)

These two files are the exact source, at the commit that produced the
corrected A17 Pro sustained-throughput run, hash-bound in
`validation/results/system-paper/a17pro/context12/context12-soak-manifest.json`:

| File | sha256 | Manifest field |
|---|---|---|
| `CrossfadeGenerationRuntime.swift` | `7762b99bd864065b926f20624531fe6ecc2144de73cd4140b0d82a07b4353225` | `artifacts.generationRuntimeSourceSha256` |
| `RenderCore.cpp` | `a0525a65d70c68c51505f58de7f61cbfeef01454093336d75528c8e1ff2f192f` | `artifacts.renderCoreSourceSha256` |

`RenderCore.h` is included alongside `RenderCore.cpp` for compilability
context (it is not independently hash-bound in the manifest).

Run `sha256sum CrossfadeGenerationRuntime.swift RenderCore.cpp` and compare
against the table above, or against the manifest directly, to confirm these
are byte-identical to what produced the paper's Table sustain numbers.

`CrossfadeGenerationRuntime.swift` is the producer loop described in prose and
pseudocode throughout §3 and the Reproducibility "Corrected runtime protocol":
the temporal/depth prediction calls, the 41-frame K/V ring, the 12-frame
decoder-context advance, backpressure against the high watermark, and the
10-second temporal reset. `RenderCore.cpp` is the C++ periodic-Hann inverse
STFT and overlap-add core described in §3.3.

This is reference source for auditing the described algorithm, not the
shipping Crossfade product. The signed, compiled application binary
(`signedExecutableSha256` in the manifest) remains private.
