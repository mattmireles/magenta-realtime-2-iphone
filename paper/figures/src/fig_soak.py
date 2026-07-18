#!/usr/bin/env python3
"""Render the A17 Pro sustain figure from the public G2 receipt."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
RECEIPT = ROOT / "validation/results/system-paper/a17pro/soak/g2-manifest.json"
CONTROL = ROOT / "validation/results/system-paper/evaluation/a17pro-cpugpu-control-soak.json"
OUTPUT = ROOT / "paper/figures/fig-soak.pdf"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"


def main() -> None:
  data = json.loads(RECEIPT.read_text())
  control = json.loads(CONTROL.read_text())
  bins = [row for row in data["minuteBins"] if row["startSeconds"] < 600]
  control_bins = [row for row in control["minuteBins"] if row["startSeconds"] < 600]
  minutes = [(row["startSeconds"] + row["endSeconds"]) / 120 for row in bins]
  p99 = [row["p99EffectiveFrameMs"] for row in bins]
  reservoir = [row["endReservoirFrames"] / 48_000 for row in bins]
  control_minutes = [
      (row["startSeconds"] + row["endSeconds"]) / 120 for row in control_bins
  ]
  control_p99 = [row["p99EffectiveFrameMs"] for row in control_bins]
  control_reservoir = [row["endReservoirFrames"] / 48_000 for row in control_bins]

  plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
  figure, (latency_axis, reservoir_axis) = plt.subplots(
      2, 1, figsize=(6.8, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.35, 1]}
  )
  latency_axis.axhline(40, color="#D55E00", linewidth=1.2, linestyle="--", label="40 ms deadline")
  latency_axis.plot(minutes, p99, marker="o", color=BLUE, label="ANE policy p99")
  latency_axis.plot(
      control_minutes,
      control_p99,
      marker="s",
      linestyle="--",
      color=ORANGE,
      label="CPU+GPU policy p99",
  )
  latency_axis.set_ylabel("effective frame time (ms)")
  latency_axis.set_ylim(0, 58)
  latency_axis.grid(axis="y", alpha=0.25)
  latency_axis.legend(ncol=2, frameon=False, loc="upper right")

  reservoir_axis.plot(minutes, reservoir, marker="o", color=GREEN, label="ANE policy")
  reservoir_axis.plot(
      control_minutes,
      control_reservoir,
      marker="s",
      linestyle="--",
      color=ORANGE,
      label="CPU+GPU policy",
  )
  reservoir_axis.axhline(data["reservoirStartFrames"] / 48_000, color="#666666", linestyle=":")
  reservoir_axis.set_ylabel("queued audio (s)")
  reservoir_axis.set_xlabel("foreground generation time (min)")
  reservoir_axis.set_xlim(0, 10)
  reservoir_axis.grid(alpha=0.25)
  reservoir_axis.legend(frameon=False, ncol=2, loc="upper left")
  figure.tight_layout()
  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(OUTPUT, bbox_inches="tight")


if __name__ == "__main__":
  main()
