.PHONY: help setup data panel experiments evaluate all smoke test lint clean

PY := python
SRC := PYTHONPATH=src

help:
	@echo "make setup        install the package and its dependencies"
	@echo "make data         download every input (free; ~2.4 GB, then ~10 min to measure turnover)"
	@echo "make panel        build the anomaly panel and its features"
	@echo "make experiments  run the walk-forward horse race  (hours)"
	@echo "make evaluate     build every table and figure     (seconds)"
	@echo "make all          data -> panel -> experiments -> evaluate"
	@echo "make capacity     re-run on large caps only, then compare (hours)"
	@echo "make smoke        3 folds, 1 seed: end-to-end in minutes"
	@echo "make test         run the test suite"

setup:
	$(PY) -m pip install -e ".[dev]"

data:
	$(SRC) $(PY) scripts/00_pull_data.py

panel:
	$(SRC) $(PY) scripts/01_build_panel.py

experiments:
	$(SRC) $(PY) scripts/02_run_experiments.py

evaluate:
	$(SRC) $(PY) scripts/03_evaluate.py

all: data panel experiments evaluate

capacity:
	$(SRC) $(PY) scripts/01_build_panel.py --config configs/capacity.yaml
	$(SRC) $(PY) scripts/02_run_experiments.py --config configs/capacity.yaml
	$(SRC) $(PY) scripts/03_evaluate.py --config configs/capacity.yaml
	$(SRC) $(PY) scripts/04_capacity_report.py

smoke:
	$(SRC) $(PY) scripts/02_run_experiments.py --smoke
	$(SRC) $(PY) scripts/03_evaluate.py

test:
	$(SRC) $(PY) -m pytest tests/ -q

lint:
	ruff check src scripts tests

clean:
	rm -rf outputs/predictions report/figures/* report/tables/*
