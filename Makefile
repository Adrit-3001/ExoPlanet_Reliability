# ExoReliability Lab — common commands. Python commands use the project virtualenv.
PY ?= .venv/bin/python
DATA_CONFIG ?= configs/data/kepler_dr25_small.yaml
EXP_CONFIG ?= configs/experiments/bls_smoke.yaml
LIMIT_KOIS ?= 20
LIMIT_TARGETS ?= 3

.PHONY: help install install-web catalog lightcurves preprocess bls pipeline test lint format typecheck api web web-lint web-build check

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install: ## Create .venv and install the package with dev tools
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

install-web: ## Install frontend dependencies (needs Node >= 20 on PATH)
	cd apps/web && npm install

catalog: ## Fetch DR25 KOI table (cached) and write a small KOI manifest
	$(PY) scripts/fetch_catalog.py --config $(DATA_CONFIG) --limit $(LIMIT_KOIS)

lightcurves: ## Download Kepler long-cadence light curves for a few manifest targets
	$(PY) scripts/fetch_lightcurves.py --config $(DATA_CONFIG) --limit $(LIMIT_TARGETS)

preprocess: ## Clean/normalize/detrend downloaded light curves and write diagnostic plots
	$(PY) scripts/preprocess_lightcurves.py --config $(EXP_CONFIG)

bls: ## Blind BLS search + Gaussian-noise demonstration (~1 min per target per severity)
	$(PY) scripts/run_bls.py --config $(EXP_CONFIG)

pipeline: catalog lightcurves preprocess bls ## Full Milestone 1 smoke pipeline on real data

test: ## Unit + integration tests (offline)
	$(PY) -m pytest

lint: ## Ruff lint + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format: ## Apply Ruff formatting and safe fixes
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

typecheck: ## mypy on src/ and apps/api/
	$(PY) -m mypy

api: ## Start the FastAPI server on :8000
	$(PY) -m uvicorn apps.api.main:app --reload --port 8000

web: ## Start the Next.js dev server on :3000
	cd apps/web && npm run dev

web-lint: ## Frontend ESLint + TypeScript
	cd apps/web && npm run lint && npm run typecheck

web-build: ## Production build of the frontend
	cd apps/web && npm run build

check: lint typecheck test web-lint ## Everything CI runs
