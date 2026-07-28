---
name: coreml-validate
description: Validate the public Magenta RealTime 2 Core ML exporters, packages, fixtures, decoder-history behavior, and evidence receipts against the frozen reference. Use for public artifact maintenance and claim verification. Route private app/device runtime profiling to Crossfade.
---

# Validate public MRT2 artifacts

Bind every result to exporter revision, fixture hashes, package hashes, model
state/history contract, seeds, and published thresholds. Reproduce stateful and
stateless decoder comparisons without rewriting frozen evidence. Keep public
artifact validation here; private Aperture runtime, signing, and device traces
belong in `../crossfade`.
