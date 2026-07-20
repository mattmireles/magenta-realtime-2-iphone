# Publication artifacts

- `pdf/mrt2-three-clocks.pdf` - archival 16-page paper.
- `arxiv/mrt2-three-clocks-source.tar.gz` - self-contained source archive;
  extract it and run `tectonic main.tex`.
- `reviewer/mrt2-three-clocks-reviewer-packet.zip` - paper, source, claims
  ledger, evidence ledger, headline machine-readable receipts, license, notice,
  and reviewer guide.

The source archive was rebuilt with Tectonic in a clean temporary directory on
2026-07-20, after the Artifact statement and Reproducibility sections were
rewritten to link the public evidence dataset and harness. The independent
build completed as a 16-page letter-size PDF with the same title and author
metadata. The reviewer ZIP passes `unzip -t`.

Raw evidence (WAVs, token captures, event traces, device logs) is public at
`huggingface.co/datasets/mattmireles/mrt2-three-clocks-evidence`; the
generation/decode/probe harness and the two hash-bound runtime source files
are public under `harness/` in this repository. See the paper's Artifact
statement for what remains private.

## SHA-256

```text
2d773a58ad3febd34511f0eed541f1a7970988dca8e556ca467fb38ac9a3c1a5  pdf/mrt2-three-clocks.pdf
c3861f461a32066806bd05cda160a398ef940a2541cd2ce7faa6e8b9c2c5f5b7  arxiv/mrt2-three-clocks-source.tar.gz
f7d6152835e4bb1c0b7785dc0241896b39c7a2238b2cafcb9c9d4aed48bd5ebe  reviewer/mrt2-three-clocks-reviewer-packet.zip
```
