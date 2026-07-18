.PHONY: all figures manuscript

all: manuscript

figures:
	python3 figures/src/fig_soak.py
	python3 figures/src/fig_latency.py
	python3 figures/src/fig_compression.py

manuscript:
	python3 build_manuscript.py
	tectonic main.tex --keep-logs --keep-intermediates
