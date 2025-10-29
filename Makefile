PYTHON ?= python3
PIP ?= pip3

.PHONY: install dev fmt lint test

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e .[dev]

fmt:
	ruff --fix || true
	black rag_gs || true

lint:
	ruff rag_gs
	black --check rag_gs
	mypy rag_gs

test:
	pytest -q
