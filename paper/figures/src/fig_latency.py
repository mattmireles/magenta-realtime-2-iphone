#!/usr/bin/env python3
"""Render the repeated-process stage-latency matrix from the public receipt."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "validation/results/system-paper/evaluation/evaluation-manifest.json"
OUTPUT = ROOT / "paper/figures/fig-latency.pdf"
STAGES = (
    ("temporalMs", "temporal", "#0072B2"),
    ("depthMs", "depth", "#E69F00"),
    ("samplingMs", "sampling", "#009E73"),
    ("decoderMs", "decoder", "#CC79A7"),
)
LABELS = {
    "a17pro-ane": "A17 Pro / ANE",
    "a17pro-temporal-gpu": "A17 Pro / CPU+GPU",
    "a14-ane-f32": "A14 / ANE",
    "a14-temporal-gpu-f32": "A14 / CPU+GPU",
}


def main() -> None:
  data = json.loads(RECEIPT.read_text())
  cells = {cell["label"]: cell for cell in data["cells"]}
  order = tuple(LABELS)
  plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
  figure, axis = plt.subplots(figsize=(6.8, 3.3))
  left = [0.0] * len(order)
  for key, label, color in STAGES:
    values = [
        cells[name]["dispersion"][
            f"latencyMsPerEffectiveFrame.{key}.p50"
        ]["median"]
        for name in order
    ]
    axis.barh(range(len(order)), values, left=left, label=label, color=color)
    left = [start + value for start, value in zip(left, values)]
  axis.axvline(
      40,
      color="#D55E00",
      linestyle="--",
      linewidth=1.3,
      label="40 ms deadline",
  )
  axis.set_yticks(range(len(order)), [LABELS[name] for name in order])
  axis.invert_yaxis()
  axis.set_xlabel("median run-level p50 (ms/effective frame)")
  axis.set_xlim(0, 55)
  axis.grid(axis="x", alpha=0.22)
  axis.legend(
      ncol=5,
      frameon=False,
      fontsize=8,
      loc="lower center",
      bbox_to_anchor=(0.5, 1.01),
  )
  figure.tight_layout(rect=(0, 0, 1, 0.9))
  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(OUTPUT, bbox_inches="tight")


if __name__ == "__main__":
  main()
