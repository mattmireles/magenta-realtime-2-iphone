#!/usr/bin/env python3
"""Render the rejected compression ladder from its public receipt."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "validation/results/MRT2WeightCompressionLadder.json"
OUTPUT = ROOT / "paper/figures/fig-compression.pdf"
BLUE = "#0072B2"
ORANGE = "#E69F00"


def main() -> None:
  data = json.loads(RECEIPT.read_text())
  temporal = [row for row in data["candidates"] if row["component"] == "temporal"]
  depth = [row for row in data["candidates"] if row["component"] == "depth"]
  labels = ["int8", "6-bit", "4-bit"]
  x = range(len(labels))
  plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
  figure, (size_axis, quality_axis) = plt.subplots(1, 2, figsize=(6.8, 2.8))
  size_axis.plot(x, [row["fraction_of_uncompressed"] for row in temporal], marker="o", color=BLUE, label="temporal")
  size_axis.plot(x, [row["fraction_of_uncompressed"] for row in depth], marker="s", color=ORANGE, label="depth")
  size_axis.set_xticks(list(x), labels)
  size_axis.set_ylim(0, 0.6)
  size_axis.set_ylabel("package bytes / baseline")
  size_axis.grid(axis="y", alpha=0.25)
  size_axis.legend(frameon=False)

  quality_axis.plot(x, [1 - row["parity_gate"]["correlation"] for row in temporal], marker="o", color=BLUE, label="temporal: 1 - corr.")
  quality_axis.plot(x, [row["parity_gate"]["arms"]["argmax_frames"]["mismatch_rate"] for row in depth], marker="s", color=ORANGE, label="depth: mismatch")
  quality_axis.set_xticks(list(x), labels)
  quality_axis.set_yscale("log")
  quality_axis.set_ylabel("deterministic parity error")
  quality_axis.grid(axis="y", alpha=0.25)
  quality_axis.legend(frameon=False)
  figure.tight_layout()
  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(OUTPUT, bbox_inches="tight")


if __name__ == "__main__":
  main()
