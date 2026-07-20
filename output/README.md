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

Raw evidence (WAVs, token captures, event traces, device logs; 139 files,
sha256-verified) is public as release assets at
`github.com/mattmireles/magenta-realtime-2-iphone/releases/tag/evidence-v1`;
the generation/decode/probe harness and the two hash-bound runtime source
files are public under `harness/` in this repository. See the paper's Artifact
statement for what remains private.

## SHA-256

```text
3fe09a4c33d103ecd720d4783ab59493038d8a359f2db1b3d182d18ee6b01e92  pdf/mrt2-three-clocks.pdf
1da59c7c287a33fe56f7562fe13c7ea9fc6f08e9c566f4d37a527a5e28daf8c9  arxiv/mrt2-three-clocks-source.tar.gz
b216efd0a7115d5ea259b6ac1fd5eab04d5a08ba43cbde992e12dcc84e758c83  reviewer/mrt2-three-clocks-reviewer-packet.zip
```
