# Reviewer packet

**Paper:** *The Three Clocks of Live Music Generation: Sustained GPU-Free
MRT2 Inference on iPhone*

This packet is venue-neutral and ready for academic review or arXiv upload.
The headline result is deliberately narrower than "live music solved": the
A17 Pro compute and delivery clocks pass for ten hot foreground minutes, while
the independently frozen 600-second generative-quality gate fails and is
retained as the principal negative result.

## Contents

- `mrt2-three-clocks.pdf` — archival 12-page manuscript.
- `mrt2-three-clocks-source.tar.gz` — self-contained Tectonic/arXiv source.
- `mrt2-system-paper-claims.md` — claim-to-gate ledger, including rejected
  claims.
- `validation-receipts.md` — public evidence map and exact receipt paths.
- `LICENSE` and `NOTICE` — repository licensing and attribution.

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

The expected output is a 12-page letter-size PDF titled *The Three Clocks of
Live Music Generation: Sustained GPU-Free MRT2 Inference on iPhone*.
