# MRT2 Weight-Compression Ladder

The ladder follows the predeclared early-stop order: finite output, deterministic reference parity, device measurement, then blind audio. A failed parity arm was not installed on either phone and was not rendered for listening.

| Component | Variant | Size (MiB) | Baseline | Key parity result | Disposition |
| --- | --- | ---: | ---: | --- | --- |
| temporal | int8-linear | 175.1 | 0.502x | corr 0.997328; max |err| 4.585 | rejected_at_parity_gate |
| temporal | palettize-6bit | 131.3 | 0.376x | corr 0.976950; max |err| 13.212 | rejected_at_parity_gate |
| temporal | palettize-4bit | 87.8 | 0.252x | corr 0.887407; max |err| 23.949 | rejected_at_parity_gate |
| depth | int8-linear | 35.9 | 0.505x | mismatch 0.463/0.333/0.290 | rejected_at_parity_gate |
| depth | palettize-6bit | 27.0 | 0.380x | mismatch 0.687/0.573/0.480 | rejected_at_parity_gate |
| depth | palettize-4bit | 18.2 | 0.256x | mismatch 0.940/0.917/0.890 | rejected_at_parity_gate |

## Decision

All six compressed component candidates remained finite but failed their declared deterministic-reference gate. The experiment stops there and retains the uncompressed temporal plus existing FP16 depth configuration.

The uncompressed temporal graph remains the selected system artifact. The existing FP16 depth graph remains selected under its previously documented distributional and device-listening acceptance boundary. This experiment does not support a compression-causes-speedup claim: no compressed candidate passed the prerequisite parity gate, so device latency, placement, DRAM estimates, and listening were deliberately not measured for those candidates.
