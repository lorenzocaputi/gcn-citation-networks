PY ?= .venv/bin/python

.PHONY: check lint test benchmark figures reproduce

check: lint test

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest -q

benchmark:
	$(PY) scripts/benchmark.py

figures:
	$(PY) scripts/figures.py

reproduce: benchmark figures
