# Reviewer packet

**Paper:** *Throughput Is Not Liveness: Three Clocks for GPU-Free Music
Generation on iPhone*

This packet is venue-neutral and ready for academic review or arXiv upload.
The headline result is deliberately narrower than "live music solved." A
three-seed, 600-second token-by-decoder crossover shows that the apparent
long-horizon quality failure follows stateless decoder windowing, not the token
source or Core ML graph. A 12-frame causal-context intervention recovers tensor
parity and removes the excess; a corrected 600-second A17 Pro run sustains
throughput and delivery with zero underruns or drops. Separately, a frozen
three-seed no-reset test fails its float-PCM full-scale gate in 3/3 seeds.
Combined reset prevents overrange in 3/3, but the K/V-versus-feedback factorial
is causally ambiguous. The paper therefore claims bounded generation rather
than indefinite model stability. A separate final-code post-ring A17 Pro run
completes 600 seconds and 30 prompt transitions without delivery or proof
faults. Its frozen paired-reference detector misses 22 transitions and measures
0.947-second p95 among eight detections, so the steering tier remains buffered
rather than responsive or live.

## Contents

- `mrt2-three-clocks.pdf` - archival manuscript.
- `mrt2-three-clocks-source.tar.gz` - self-contained Tectonic/arXiv source.
- `mrt2-system-paper-claims.md` - claim-to-gate ledger, including rejected
  claims.
- `validation-receipts.md` - public evidence map and exact receipt paths.
- `mrt2-system-paper-revision-report.json` - machine-checked verdict for the
  crossover, tensor probe, and corrected device run.
- `mrt2-system-paper-crossover-aggregate.json` - seed-level effects and
  diagnostic counts without pooling away the three replications.
- `mrt2-system-paper-g5-manifest.json` - all 12 reset-factorial arms, calibrated
  event-centered vote outcomes, frozen gate, narrow permitted claim, and hashes.
- `mrt2-system-paper-g6-manifest.json` - complete post-ring steering result,
  device/runtime identity, buffered-only claim, and hashes of private evidence.
- `mrt2-context12-soak-manifest.json` - normalized corrected A17 Pro receipt
  with hashes binding the private raw capture and signed runtime.
- `mrt2-depth-rollout-ablation.json` - measured correction to the one-call
  depth-rollout explanation.
- `LICENSE` and `NOTICE` - repository licensing and attribution.

## Public artifacts

- Repository: <https://github.com/mattmireles/magenta-realtime-2-iphone>
- Model mirror: <https://huggingface.co/mattmireles/magenta-realtime-2-iphone>
- Publication dataset: `validation/results/system-paper/` in the repository.

The complete Crossfade product runtime, raw device logs, Instruments traces,
and long WAVs remain private. Their hashes are bound into the checked-in
manifests, and they are available from the author for artifact review. The
paper makes no corrected-pipeline energy or battery-life claim because the
counterbalanced Power Profiler pair produced no valid measurement.

## Rebuild

Extract the source archive and run:

```bash
tectonic main.tex
```

The expected output is a letter-size PDF titled *Throughput Is Not Liveness:
Three Clocks for GPU-Free Music Generation on iPhone*.
