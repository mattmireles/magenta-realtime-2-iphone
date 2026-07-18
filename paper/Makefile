.PHONY: all figures manuscript

PYTHON ?= python3

all: manuscript

figures:
	$(PYTHON) figures/src/fig_soak.py
	$(PYTHON) figures/src/fig_latency.py
	$(PYTHON) figures/src/fig_compression.py
	$(PYTHON) figures/src/fig_crossover.py

manuscript:
	$(PYTHON) build_manuscript.py
	tectonic main.tex --keep-logs --keep-intermediates
