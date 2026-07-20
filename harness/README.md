# Generation/decode/probe harness (reference source)

The paper's Reproducibility section (§ "Crossover commands") describes "the
private Crossfade harness" that exposes generation, decoding, summarization,
and decoder-context probe subcommands. These are the scripts it refers to,
published as reference source so the crossover and liveness experiments are
auditable end to end — not just re-analyzable from already-decoded WAVs.

These are extracted from the private Crossfade product repository, not the
product itself. Several call out to a compiled render core or an MLX/Core ML
checkpoint via `subprocess`; running them end-to-end requires the converted
model artifacts (see the exporters and `MODELS.md` in this repository) and,
for the Core ML decode path, the compiled render core described in
`paper/draft.md` §3.3. They are provided so the exact generation, decoding,
and measurement logic is inspectable and portable, not as a turnkey CLI.

| Script | Role |
|---|---|
| `run_mrt2_long_horizon_crossover.py` | Generates the 2x2 token-source x decoder-path crossover (§4.3), the FLOAT32 pre-iSTFT split, and the 12-frame context arms. |
| `run_mrt2_liveness_matrix.py` | Runs the frozen 3-seed x 4-arm reset factorial (§4.4), resumable one pair at a time. |
| `analyze_mrt2_liveness.py` | Computes the full-scale overrange gate and token diagnostics for one refreshed/unrefreshed pair. |
| `aggregate_mrt2_liveness.py` | Aggregates the frozen liveness evidence into the private-then-published G5 candidate. |
| `build_mrt2_liveness_protocol.py` | Freezes the liveness fixtures, arms, analysis, and judge gates before replication. |
| `normalize_mrt2_liveness_judge_votes.py` | Produces the authenticated event-centered judge-vote evidence (§4.4, listening). |
| `probe_mrt2_temporal_kv_state_coreml.py` | The state proof: fresh-vs-warmed divergence and 64-step reference match (§3.1). |
| `probe_spectrostream_rvq_lookup.py` | Decoder-context tensor probe (§4.6): correlation/error vs. retained history depth. |
| `verify_spectrostream_streaming_decode.py` | Sample-level parity test for the C++ periodic-Hann inverse STFT against a NumPy reference (§4.6). |
| `convert_spectrostream_decoder_{conv,prefix,tail}_coreml.py` | Core ML conversion for the three decoder sub-graphs described in §3.3. |
| `summarize_crossfade_event_trace.py` | Turns a raw device event trace into the summary counters reported in §5 (underruns, drops, reservoir trajectory). |
| `run_system_paper_latency_device.py` | Captures the fixed-protocol device latency runs used in the duration-controlled thermal result (§5.3). |

## License

Files carrying a `Copyright 2026 Google LLC` / Apache-2.0 header are adapted
from the [Magenta RealTime](https://github.com/magenta/magenta-realtime)
codebase, per this repository's `NOTICE`. The remainder are original to this
project and covered by the repository's Apache-2.0 `LICENSE`.
