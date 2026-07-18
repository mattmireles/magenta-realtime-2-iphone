# The Three Clocks of Live Music Generation

This directory contains the readable draft, generated archival LaTeX, figures,
and bibliography for *The Three Clocks of Live Music Generation: Sustained
GPU-Free MRT2 Inference on iPhone*.

## Build

From this directory:

```bash
make manuscript
```

The checked-in figure PDFs are sufficient to build `main.pdf` with Tectonic.
To regenerate quantitative figures, install Matplotlib and run `make figures`
from a Python environment. Every quantitative figure reads a checked-in JSON
receipt under `../validation/results/`; it does not contain hand-entered plot
data.

## Source of truth

- `draft.md` is the readable editorial source.
- `build_manuscript.py` converts that source into the self-contained
  `main.tex`, inserting the archival tables and figures.
- `refs.bib` contains the bibliography.
- `figures/src/` contains quantitative plotting code; `figures/fig-system.tex`
  is the native TikZ system diagram.
- `../docs/plans/mrt2-system-paper-claims.md` maps every headline claim to its
  public gate and receipt.

The paper intentionally reports a failed long-horizon generative-quality gate.
It claims sustained GPU-free inference and audio delivery on A17 Pro, not
arbitrary-horizon musical validity. The Crossfade product runtime is private
and available from the author for artifact review; exporters, fixtures,
normalizers, public verdicts, figure sources, and paper source are in this
repository.
