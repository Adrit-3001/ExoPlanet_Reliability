# ExoReliability Lab — common commands. Python commands use the project virtualenv.
PY ?= .venv/bin/python
DATA_CONFIG ?= configs/data/kepler_dr25_small.yaml
EXP_CONFIG ?= configs/experiments/bls_smoke.yaml
LIMIT_KOIS ?= 20
LIMIT_TARGETS ?= 3

.PHONY: help install install-web catalog lightcurves preprocess bls pipeline splits dataset-smoke dataset-small dataset-report detrend-eval profile-bls test lint format typecheck api web web-lint web-build check

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install: ## Create .venv and install the package with dev tools and PyTorch
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,ml]"

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

splits: ## Create KIC-grouped train/val/test splits (refuses to overwrite existing ones)
	$(PY) scripts/create_splits.py

dataset-smoke: ## Build the 9-star smoke ML dataset
	$(PY) scripts/build_dataset.py --config configs/data/kepler_dr25_smoke.yaml --overwrite

dataset-small: ## Build the ~100-star small ML dataset (~700 MB download on first run)
	$(PY) scripts/build_dataset.py --config configs/data/kepler_dr25_small.yaml --overwrite

dataset-report: ## Counts, leakage, data-quality and bias report for the small dataset
	$(PY) scripts/dataset_report.py --dataset kepler_dr25_small

detrend-eval: ## Synthetic transit-preservation benchmark + real-star old/new comparison
	$(PY) scripts/evaluate_detrending.py --seeds 10 --real-limit 200

profile-bls: ## Profile BLS runtime on one real star
	$(PY) scripts/profile_bls.py --kepid 5374854

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
