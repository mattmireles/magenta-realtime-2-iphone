#!/usr/bin/env python3
"""Render the long-horizon decoder crossover from the public aggregate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "validation/results/system-paper/crossover/aggregate.json"
OUTPUT = ROOT / "paper/figures/fig-crossover.pdf"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"


def main() -> None:
  data = json.loads(RECEIPT.read_text())
  effects = data["effects"]
  effect_keys = [
      ("tokenSourceAtStreamingMlxDecoder", "tokens"),
      ("coremlGraphVsMlxGraphAtLegacyDsp", "graph"),
      ("statelessWindowingAndLegacyDspVsStreamingMlx", "window"),
      ("context12InterventionAtCoremlGraph", "+context"),
  ]
  colors = [GREEN, PURPLE, ORANGE, BLUE]
  plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
  figure, axes = plt.subplots(
      1,
      3,
      figsize=(7.35, 2.65),
      gridspec_kw={"width_ratios": [1.35, 1, 0.85]},
  )

  effect_axis = axes[0]
  for index, ((key, _), color) in enumerate(zip(effect_keys, colors)):
    values = [row["mean"] * 100 for row in effects[key]["perSeed"]]
    offsets = np.linspace(-0.12, 0.12, len(values))
    effect_axis.scatter(index + offsets, values, color=color, s=28, zorder=3)
    effect_axis.hlines(
        effects[key]["medianSeedMean"] * 100,
        index - 0.22,
        index + 0.22,
        color="black",
        linewidth=1.6,
        zorder=4,
    )
  effect_axis.axhline(0, color="#666666", linewidth=0.8)
  effect_axis.set_xticks(range(len(effect_keys)), [label for _, label in effect_keys])
  effect_axis.set_ylabel("change in 4-16 Hz pulse share\n(percentage points)")
  effect_axis.grid(axis="y", alpha=0.2)
  effect_axis.set_title("(a) Matched 600 s effects", loc="left", fontweight="bold")

  context_axis = axes[1]
  arms = data["decoderContextProbe"]["arms"]
  contexts = sorted(int(key) for key in arms)
  errors = [arms[str(context)]["maxAbsoluteError"] for context in contexts]
  context_axis.semilogy(contexts, errors, marker="o", color=BLUE, linewidth=1.7)
  context_axis.set_xlabel("retained token frames")
  context_axis.set_ylabel("max STFT error")
  context_axis.set_xticks(contexts)
  context_axis.grid(alpha=0.2)
  context_axis.set_title("(b) Decoder state recovery", loc="left", fontweight="bold")
  context_axis.annotate(
      "corr. 0.108",
      (contexts[0], errors[0]),
      xytext=(8, -15),
      textcoords="offset points",
      fontsize=7.5,
  )
  context_axis.annotate(
      "corr. > 0.999999999",
      (contexts[-1], errors[-1]),
      xytext=(-78, 14),
      textcoords="offset points",
      fontsize=7.5,
      arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.7},
  )

  count_axis = axes[2]
  counts = data["diagnosticThresholdCounts"]
  count_values = [
      counts["streamingMlx"]["overLimit"],
      counts["statelessCoremlLegacyDsp"]["overLimit"],
      counts["context12CoremlTrainedHann"]["overLimit"],
  ]
  labels = ["MLX", "0 ctx", "12 ctx"]
  bars = count_axis.bar(range(3), count_values, color=[GREEN, ORANGE, BLUE], width=0.68)
  count_axis.set_ylim(0, 40)
  count_axis.set_yticks([0, 10, 20, 30, 40])
  count_axis.set_ylabel("windows over 0.070\n(out of 60)")
  count_axis.set_xticks(range(3), labels)
  count_axis.grid(axis="y", alpha=0.2)
  count_axis.bar_label(bars, padding=2, fontsize=8)
  count_axis.set_title("(c) Diagnostic recovery", loc="left", fontweight="bold")

  figure.subplots_adjust(left=0.08, right=0.995, bottom=0.21, top=0.84, wspace=0.48)
  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(OUTPUT, bbox_inches="tight")


if __name__ == "__main__":
  main()
