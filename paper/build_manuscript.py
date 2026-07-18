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
}

REFERENCES = {
    "[Fig. system]": r"\cref{fig:system}",
    "[Fig. soak]": r"\cref{fig:soak}",
    "[Fig. latency]": r"\cref{fig:latency}",
    "[Fig. compression]": r"\cref{fig:compression}",
    "[Table sustain]": r"\cref{tab:sustain}",
    "[Table dispersion]": r"\cref{tab:dispersion}",
    "[Table audio]": r"\cref{tab:audio}",
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
  pdftitle={The Three Clocks of Live Music Generation: Sustained GPU-Free MRT2 Inference on iPhone},
  pdfauthor={Matt Mireles},
  pdfkeywords={on-device inference, generative audio, Core ML, Apple Neural Engine, real-time systems}
}
\setlist{nosep,leftmargin=1.45em}
\setlength{\emergencystretch}{2em}
\setlength{\parskip}{0.24em}
\setlength{\parindent}{1.1em}
\definecolor{aneBlue}{HTML}{0072B2}
\definecolor{cpuOrange}{HTML}{E69F00}
\title{\textbf{The Three Clocks of Live Music Generation:}\\
Sustained GPU-Free MRT2 Inference on iPhone}
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
  model boundary as ordinary K/V tensors. No claim of an ``all-ANE'' system is made.}
  \label{fig:system}
\end{figure}
"""

RESULTS_TABLE = r"""
\begin{table}[H]
\centering
\small
\caption{Sustained foreground results. The selected A17 policy passes compute
and delivery; its CPU+GPU-policy control fails compute despite buffered
delivery. A14 is a bounded-reservoir failure. Deadline: 40 ms.}
\label{tab:sustain}
\begin{tabular}{@{}lrrrrll@{}}
\toprule
Device & p50 & p90 & p99 & Rate & First underflow & Verdict \\
 & \multicolumn{3}{c}{effective frame (ms)} & ($\times$RT) & & \\
\midrule
A17 ANE & 20.26 & 20.81 & 21.66 & 1.031 & none / 610 s & pass \\
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
\caption{Independent-process latency campaign, five runs per cell. Entries are
median (IQR) across run-level percentiles. The CPU+GPU control changes only the
requested temporal policy; depth remains CPU and decoder remains ANE.}
\label{tab:dispersion}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Device & Temporal & p50 & p99 & Temporal p50 & Startup \\
 & policy & \multicolumn{3}{c}{ms/effective frame} & s \\
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
  temporal cost on both phones; A14 reaches the deadline before its tail is
  considered. CPU+GPU denotes a requested Core ML policy, not a GPU-placement
  proof.}
  \label{fig:latency}
\end{figure}
"""

SOAK_FIGURE = r"""
\begin{figure}[t]
  \centering
  \includegraphics[width=0.94\textwidth]{figures/fig-soak.pdf}
  \caption{A17 Pro matched ten-minute trajectories. The selected ANE-policy
  run begins in \texttt{serious} thermal state and keeps p99 below deadline
  while queued audio grows. The CPU+GPU-policy control crosses the deadline
  after minute six and drains its banked reservoir, despite ending without an
  underrun. Both curves are generated from public receipts.}
  \label{fig:soak}
\end{figure}
"""

AUDIO_TABLE = r"""
\begin{table}[t]
\centering
\small
\caption{Long-horizon A17 audio gate. Arrows show the required direction;
bold values fail the frozen threshold. Lower temperature fixes only the pulse
detector and is therefore rejected.}
\label{tab:audio}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Arm & Clip $\downarrow$ & L/R $\uparrow$ & Prompt $\uparrow$ & Ref. $\uparrow$ & Pulse $\downarrow$ & Blind \\
\midrule
Threshold & $10^{-5}$ & .970 & .300 & .800 & .070 & $\geq$4/5 \\
$T=1.0$, 600 s & $4.05\!\times\!10^{-6}$ & .986 & .312 & .854 & \textbf{.136} & \textbf{2/5} \\
$T=0.5$, 600 s & \textbf{$3.27\!\times\!10^{-5}$} & \textbf{.946} & \textbf{.268} & \textbf{.784} & .042 & --- \\
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
\mathcal{C}_{\mathrm{compute}} &\equiv
  Q_{0.99}(t_{\mathrm{eff}}) < 40\,\mathrm{ms}, \\
\mathcal{C}_{\mathrm{delivery}} &\equiv
  (U=0) \land (D=0) \land (B_{\mathrm{end}} \ge B_{\mathrm{start}}), \\
\mathcal{C}_{\mathrm{gen}} &\equiv \bigwedge_{k=1}^{K} m_k \in \mathcal{A}_k.
\end{align}
Here $U$ and $D$ are callback underruns and producer drops, $B$ is queued
PCM, and each $m_k$ is a predeclared audio measure with acceptance set
$\mathcal{A}_k$. A complete live-system pass is the conjunction of all three
clocks; no score from one clock substitutes for another.
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
      if clean_heading.startswith("A14 boundary"):
        output.append(SOAK_FIGURE)
      if clean_heading.startswith("Generative clock"):
        output.append(AUDIO_TABLE)
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
