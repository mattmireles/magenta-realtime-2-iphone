# Paper figures

Every quantitative figure is regenerated from a checked-in machine-readable
receipt. Run from the repository root:

```bash
python3 paper/figures/src/fig_soak.py
python3 paper/figures/src/fig_latency.py
python3 paper/figures/src/fig_compression.py
```

The system diagram is native TikZ in `fig-system.tex`. Colors use the
color-vision-safe Okabe--Ito palette.
