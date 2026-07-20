#!/usr/bin/env python3
"""Build the archival LaTeX manuscript from the readable Markdown source."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "draft.md"
OUTPUT = ROOT / "main.tex"

CITATIONS = {
    "[Pasini et al., 2025]": "live-music-models",
    "[Pasini et al., 2025b]": "spectrostream",
    "[Google DeepMind, 2026]": "mrt2",
    "[Google, 2026]": "mrt2-model-card",
    "[Copet et al., 2023]": "musicgen",
    "[Caillon and Esling, 2021]": "rave",
    "[Zeghidour et al., 2021]": "soundstream",
    "[Defossez et al., 2022]": "encodec",
    "[Reddi et al., 2022]": "mlperf-mobile",
    "[Apple, 2022; Apple, 2026]": "apple-ane-transformers,coremltools",
    "[Apple, 2022]": "apple-ane-transformers",
    "[Apple, 2026]": "coremltools",
    "[Hannun et al., 2023]": "mlx",
    "[Xu et al., 2025]": "llm-npu",
    "[Mireles, 2026]": "surgical-inference",
    "[Holtzman et al., 2020]": "neural-text-degeneration",
    "[Rohatgi et al., 2025]": "next-token-barrier",
}

REFERENCES = {
    "[Fig. system]": r"\cref{fig:system}",
    "[Fig. soak]": r"\cref{fig:soak}",
    "[Fig. latency]": r"\cref{fig:latency}",
    "[Fig. compression]": r"\cref{fig:compression}",
    "[Fig. crossover]": r"\cref{fig:crossover}",
    "[Table sustain]": r"\cref{tab:sustain}",
    "[Table dispersion]": r"\cref{tab:dispersion}",
    "[Table crossover]": r"\cref{tab:crossover}",
    "[Table liveness]": r"\cref{tab:liveness}",
    "[Table depth]": r"\cref{tab:depth}",
}

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[letterpaper,margin=0.76in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage{enumitem}
\usepackage{natbib}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{cleveref}
\hypersetup{
  colorlinks=true,
  linkcolor=blue!45!black,
  citecolor=blue!45!black,
  urlcolor=blue!55!black,
  pdftitle={Throughput Is Not Liveness: Three Clocks for GPU-Free Music Generation on iPhone},
  pdfauthor={Matt Mireles},
  pdfkeywords={on-device inference, generative audio, Core ML, Apple Neural Engine, real-time systems}
}
\setlist{nosep,leftmargin=1.45em}
\setlength{\emergencystretch}{2em}
\setlength{\parskip}{0.20em}
\setlength{\parindent}{1.1em}
\definecolor{aneBlue}{HTML}{0072B2}
\definecolor{cpuOrange}{HTML}{E69F00}
\title{\textbf{Throughput Is Not Liveness:}\\
Three Clocks for GPU-Free Music Generation on iPhone}
\author{Matt Mireles\\Independent Researcher\\\href{https://github.com/mattmireles}{github.com/mattmireles}}
\date{July 2026}
\begin{document}
\maketitle
"""

SYSTEM_FIGURE = r"""
\begin{figure}[t]
  \centering
  \resizebox{\textwidth}{!}{\input{figures/fig-system.tex}}
  \caption{End-to-end deployment boundary. Blue stages are observed on the ANE;
  orange stages remain deliberate host/CPU work. Temporal recurrence crosses the
  model boundary as ordinary K/V tensors; the stateless decoder retains 12 token
  frames of measured causal context. No claim of an ``all-ANE'' system is made.}
  \label{fig:system}
\end{figure}
"""

RESULTS_TABLE = r"""
\begin{table}[H]
\centering
\small
\caption{Sustained foreground results. Cost is a p99 over 25-token iteration
means, not per-token tail latency. The corrected A17 runtime sustains throughput
and playback; its 21 s queue does not satisfy low-latency steering. The CPU+GPU
control and A14 rows predate the decoder-context repair and are conservative
duration/hardware controls.}
\label{tab:sustain}
\begin{tabular}{@{}lrrrrll@{}}
\toprule
Device & p50 & p90 & p99 & Rate & First underflow & Verdict \\
 & \multicolumn{3}{c}{iteration-normalized ms/token} & ($\times$RT) & & \\
\midrule
A17 ANE + context & 23.50 & 24.33 & 24.81 & 1.030 & none / 610 s & throughput pass \\
A17 CPU+GPU & 23.56 & 43.55 & 49.23 & 1.008 & none / 610 s & compute fail \\
A14 ANE & 43.88 & 47.25 & 58.59 & 0.897 & 294.08 s & fail \\
\bottomrule
\end{tabular}
\end{table}
"""

LATENCY_TABLE = r"""
\begin{table}[H]
\centering
\small
\caption{Independent-process pre-context latency campaign, five runs per cell. Entries are
median (IQR) across run-level percentiles. The CPU+GPU control changes only the
requested temporal policy; depth remains CPU and decoder remains ANE.}
\label{tab:dispersion}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Device & Temporal & p50 & p99 & Temporal p50 & Startup \\
 & policy & \multicolumn{3}{c}{iteration-normalized ms/token} & s \\
\midrule
A17 Pro & ANE & 20.77 (.03) & 22.35 (.42) & 11.25 (.01) & 4.26 (.04) \\
A17 Pro & CPU+GPU & 27.32 (.25) & 38.51 (1.78) & 16.35 (.15) & 6.22 (.05) \\
A14 & ANE & 40.38 (1.00) & 50.72 (3.30) & 13.85 (.63) & 7.52 (.01) \\
A14 & CPU+GPU & 49.73 (1.11) & 50.64 (1.41) & 24.85 (.17) & 9.76 (.14) \\
\bottomrule
\end{tabular}
\end{table}
"""

LATENCY_FIGURE = r"""
\begin{figure}[H]
  \centering
  \includegraphics[width=0.94\textwidth]{figures/fig-latency.pdf}
  \caption{Stage decomposition for the repeated-process campaign. Bars use
  the median of run-level stage p50 values. The selected ANE policy reduces
  temporal cost on both phones. These are quantiles of iteration means, not
  per-token latency quantiles. CPU+GPU denotes a requested Core ML policy, not
  a GPU-placement proof.}
  \label{fig:latency}
\end{figure}
"""

SOAK_FIGURE = r"""
\begin{figure}[H]
  \centering
  \includegraphics[width=0.94\textwidth]{figures/fig-soak.pdf}
  \caption{A17 Pro matched pre-context ten-minute trajectories. The selected
  ANE-policy run keeps iteration-normalized p99 flat. The CPU+GPU-policy control
  moves from 38.51 ms in five short processes to 49.23 ms here and crosses 40 ms
  after minute six. The duration effect is visible even though buffering prevents
  an underrun.}
  \label{fig:soak}
\end{figure}
"""

CROSSOVER_TABLE = r"""
\begin{table}[H]
\centering
\small
\caption{Three-seed, 600 s crossover. Effects are changes in 4--16 Hz envelope
pulse share; entries are median seed mean [range]. The context intervention is
paired against the same graph and corrected DSP without context.}
\label{tab:crossover}
\begin{tabular}{@{}lrr@{}}
\toprule
Intervention & Effect & Positive windows \\
\midrule
Token source at MLX decoder & +.00181 [-.00057, .00627] & 35/60 \\
Core ML vs MLX decoder graph & +.00018 [.00016, .00088] & 44/60 \\
Stateless window vs streaming MLX & +.01706 [.01613, .01796] & 60/60 \\
Add 12-frame context, same Core ML/DSP & -.01650 [-.01772, -.01574] & 0/60 \\
\bottomrule
\end{tabular}
\end{table}
"""

LIVENESS_TABLE = r"""
\begin{table}[H]
\centering
\small
\caption{Frozen 600 s unrefreshed liveness factorial. Entries are
float-PCM samples with $|x|\geq1.0$ (peak $|x|$). Reset arms intervene every
10 s while preserving RNG state and absolute position. Zero was required.
The factorial prevents overrange but does not identify a unique causal state.}
\label{tab:liveness}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Seed & No reset & K/V only & Feedback only & Both \\
\midrule
20260718 & 2 (1.0028) & 0 (.9919) & 0 (.8616) & 0 (.9383) \\
271828   & 57 (1.1730) & 0 (.9264) & 13 (1.0359) & 0 (.8561) \\
1618033  & 46 (1.2382) & 0 (.9923) & 0 (.9923) & 0 (.9923) \\
\bottomrule
\end{tabular}
\end{table}
"""

CROSSOVER_FIGURE = r"""
\begin{figure}[H]
  \centering
  \includegraphics[width=\textwidth]{figures/fig-crossover.pdf}
  \caption{Causal localization. (a) Each dot is one 600 s seed; black bars are
  medians. (b) Twelve retained token frames recover stateful MLX pre-iSTFT
  output to correlation above 0.999999999. (c) The original prompt-specific
  0.070 diagnostic fires in 35/60 stateless Core ML windows, but only 13/60
  corrected windows versus 11/60 stateful MLX windows.}
  \label{fig:crossover}
\end{figure}
"""

DEPTH_TABLE = r"""
\begin{table}[H]
\centering
\small
\caption{Measured A14 depth rollout ablation. The in-graph boundary removes host
boundaries but still traverses transformer weights at each dependent RVQ level;
the major gain is FLOAT16, not a fictitious single weight traversal.}
\label{tab:depth}
\begin{tabular}{@{}lrr@{}}
\toprule
Depth boundary & Precision & ms/token frame \\
\midrule
12 separate full-pass predictions & FLOAT32 & 40.2 \\
One in-graph 12-level prediction & FLOAT32 & $\sim$37.0 \\
One in-graph 12-level prediction & FLOAT16 & 12.7 \\
\bottomrule
\end{tabular}
\end{table}
"""

COMPRESSION_FIGURE = r"""
\begin{figure}[H]
  \centering
  \includegraphics[width=0.94\textwidth]{figures/fig-compression.pdf}
  \caption{Rejected post-training compression ladder. Every artifact remains
  finite and becomes substantially smaller, but deterministic parity degrades
  monotonically. The gate order stops all six candidates before device timing
  or listening, avoiding a speed result for the wrong function.}
  \label{fig:compression}
\end{figure}
"""

THREE_CLOCKS = r"""
\begin{align}
\mathcal{C}_{\mathrm{compute}} &\equiv R_{\mathrm{producer}} \ge 1, \\
\mathcal{C}_{\mathrm{delivery}} &\equiv
  (U=0) \land (D=0) \land (0 < B(t) \le B_{\max}), \\
\mathcal{C}_{\mathrm{gen}} &\equiv \bigwedge_{k=1}^{K} m_k \in \mathcal{A}_k.
\end{align}
Here $R_{\mathrm{producer}}$ is sustained generated-audio rate relative to
playback, $U$ and $D$ are callback underruns and producer drops, $B$ is queued
PCM, and $B_{\max}$ is chosen from an audible steering-latency budget. Each
$m_k$ is a prompt-conditional audio measure with acceptance set $\mathcal{A}_k$.
No score from one clock substitutes for another.
"""


def escape_plain(text: str) -> str:
  """Escape prose while preserving citations, code, bold, and emphasis."""
  placeholders: list[str] = []

  def hold(value: str) -> str:
    token = f"ZZPH{len(placeholders)}ZZ"
    placeholders.append(value)
    return token

  for label, key in CITATIONS.items():
    text = text.replace(label, hold(rf"\citep{{{key}}}"))
  for label, reference in REFERENCES.items():
    text = text.replace(label, hold(reference))
  text = text.replace(" -> ", hold(r" $\rightarrow$ "))

  text = re.sub(
      r"`([^`]+)`",
      lambda match: hold(r"\texttt{\detokenize{" + match.group(1) + "}}"),
      text,
  )
  text = re.sub(
      r"\*\*([^*]+)\*\*",
      lambda match: hold(r"\textbf{" + escape_plain(match.group(1)) + "}"),
      text,
  )
  text = re.sub(
      r"(?<!\*)\*([^*]+)\*(?!\*)",
      lambda match: hold(r"\emph{" + escape_plain(match.group(1)) + "}"),
      text,
  )
  replacements = {
      "\\": r"\textbackslash{}",
      "&": r"\&",
      "%": r"\%",
      "$": r"\$",
      "#": r"\#",
      "_": r"\_",
      "{": r"\{",
      "}": r"\}",
      "~": r"\textasciitilde{}",
      "^": r"\textasciicircum{}",
      "≤": r"\(\leq\)",
      "≥": r"\(\geq\)",
      "×": r"\(\times\)",
      "→": r"\(\rightarrow\)",
  }
  text = "".join(replacements.get(char, char) for char in text)
  for index, value in enumerate(placeholders):
    text = text.replace(f"ZZPH{index}ZZ", value)
  return text


def paragraphs(lines: list[str]) -> list[tuple[str, str]]:
  blocks: list[tuple[str, str]] = []
  index = 0
  while index < len(lines):
    raw = lines[index].rstrip()
    if not raw:
      index += 1
      continue
    if raw.startswith("## "):
      blocks.append(("section", raw[3:]))
      index += 1
      continue
    if raw.startswith("### "):
      blocks.append(("subsection", raw[4:]))
      index += 1
      continue
    if raw.startswith("- "):
      items = []
      while index < len(lines) and lines[index].startswith("- "):
        item = lines[index][2:].strip()
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith("- "):
          item += " " + lines[index].strip()
          index += 1
        items.append(item)
        while index < len(lines) and not lines[index].strip():
          index += 1
      blocks.append(("items", "\n".join(items)))
      continue
    if re.match(r"\d+\. ", raw):
      items = []
      while index < len(lines) and re.match(r"\d+\. ", lines[index]):
        item = re.sub(r"^\d+\. ", "", lines[index]).strip()
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(r"\d+\. ", lines[index]):
          item += " " + lines[index].strip()
          index += 1
        items.append(item)
        while index < len(lines) and not lines[index].strip():
          index += 1
      blocks.append(("enumerate", "\n".join(items)))
      continue
    text = raw.strip()
    index += 1
    while index < len(lines) and lines[index].strip() and not lines[index].startswith("## "):
      text += " " + lines[index].strip()
      index += 1
    blocks.append(("paragraph", text))
  return blocks


def render_blocks(blocks: list[tuple[str, str]]) -> str:
  output: list[str] = []
  for kind, value in blocks:
    clean_heading = re.sub(r"^\d+(?:\.\d+)?\.?\s*", "", value)
    if kind == "section":
      if clean_heading == "Reproducibility":
        output.append(r"\appendix")
      if clean_heading == "Experimental method":
        output.append(SYSTEM_FIGURE)
      if clean_heading == "Background and related work":
        output.append(THREE_CLOCKS)
      output.append(r"\section{" + escape_plain(clean_heading) + "}")
      if clean_heading == "Results":
        output.append(RESULTS_TABLE)
    elif kind == "subsection":
      output.append(r"\subsection{" + escape_plain(clean_heading) + "}")
      if clean_heading.startswith("Cross-process dispersion"):
        output.append(LATENCY_TABLE)
        output.append(LATENCY_FIGURE)
      if clean_heading.startswith("Duration changes"):
        output.append(LATENCY_TABLE)
        output.append(LATENCY_FIGURE)
        output.append(SOAK_FIGURE)
      if clean_heading.startswith("A14 boundary"):
        output.append(SOAK_FIGURE)
      if clean_heading.startswith("Generative clock"):
        output.append(CROSSOVER_TABLE)
      if clean_heading.startswith("Crossover localizes"):
        output.append(CROSSOVER_TABLE)
        output.append(CROSSOVER_FIGURE)
      if clean_heading.startswith("Unrefreshed generation"):
        output.append(LIVENESS_TABLE)
      if clean_heading.startswith("One-call depth"):
        output.append(DEPTH_TABLE)
      if clean_heading.startswith("Compression ladder"):
        output.append(COMPRESSION_FIGURE)
    elif kind in {"items", "enumerate"}:
      env = "itemize" if kind == "items" else "enumerate"
      output.append(rf"\begin{{{env}}}")
      for item in value.splitlines():
        output.append(r"\item " + escape_plain(item))
      output.append(rf"\end{{{env}}}")
    else:
      if value == "`(temporal + depth + sampling + decoder) / 25`.":
        output.append(
            r"\[t_{\mathrm{eff}} = "
            r"\frac{t_{\mathrm{temporal}} + t_{\mathrm{depth}} + "
            r"t_{\mathrm{sampling}} + t_{\mathrm{decoder}}}{25}.\]"
        )
      else:
        output.append(escape_plain(value) + "\n")
  return "\n".join(output)


def main() -> None:
  lines = SOURCE.read_text().splitlines()
  if not lines or not lines[0].startswith("# "):
    raise SystemExit("draft must begin with a Markdown title")
  abstract_index = lines.index("## Abstract")
  first_section = next(index for index, line in enumerate(lines) if line.startswith("## 1."))
  abstract = render_blocks(paragraphs(lines[abstract_index + 1:first_section]))
  body = render_blocks(paragraphs(lines[first_section:]))
  ending = r"""
\bibliographystyle{plainnat}
\bibliography{refs}
\end{document}
"""
  OUTPUT.write_text(
      PREAMBLE
      + "\\begin{abstract}\n"
      + abstract
      + "\n\\end{abstract}\n\n"
      + body
      + ending
  )


if __name__ == "__main__":
  main()
