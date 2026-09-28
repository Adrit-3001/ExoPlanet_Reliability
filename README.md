# ExoReliability Lab

**When should we trust a model that says it has found a transiting exoplanet, and under
what observational conditions does it fail?**

ExoReliability Lab studies the *reliability* of transit detection in Kepler light curves:
calibration, robustness to controlled perturbations (noise, missing cadence, shallower
transits), and failure modes. It compares learned models against a classical Box Least
Squares (BLS) baseline. It is not a leaderboard project, and model scores are not
probabilities that a planet exists.

> **Status: Milestone 1 (data + classical baseline).** No neural network has been trained.
> The repository fetches real Kepler DR25 catalog data and light curves, preprocesses them,
> runs a blind BLS search, applies a seeded Gaussian-noise perturbation, and exposes the
> results through a small API and web page. See [`docs/progress.md`](docs/progress.md).

## Quick start

Requirements: Python ≥ 3.11 (developed on 3.12), Node.js ≥ 20 for the web app (developed
on 24 LTS), internet access for the data-fetch steps only. CUDA is not needed.

```bash
# 1. Backend
make install                     # creates .venv and installs the package with dev tools

# 2. Tiny real-data pipeline (~20 MB download, ~12 min in total, dominated by BLS)
make catalog                     # DR25 KOI table (cached) + 20-KOI manifest (10 planet / 10 FP)
make lightcurves                 # Kepler long-cadence FITS for 3 stars from the manifest
make preprocess                  # clean/normalize/detrend -> data/processed, plots -> artifacts/figures
make bls                         # blind BLS + noise demo -> artifacts/experiments/<timestamp>_bls_smoke

# 3. Checks (offline)
make test                        # pytest: unit + API integration tests on synthetic fixtures
make lint typecheck              # ruff + mypy

# 4. Run the app
make api                         # FastAPI on http://localhost:8000 (docs at /docs)
make install-web && make web     # Next.js on http://localhost:3000
```

The underlying scripts accept options, for example
`python scripts/fetch_catalog.py --limit 20`,
`python scripts/fetch_lightcurves.py --kepid 10811496 --quarters 1 2 3`,
`python scripts/run_bls.py --kepid 5374854 --no-noise-demo`.
Re-running a fetch step reuses cached files without network access.

Web-app settings: copy `apps/web/.env.local.example` to `apps/web/.env.local` if the API is
not at `http://localhost:8000`. The API's allowed origins are set with
`EXORELIABILITY_CORS_ORIGINS` (see `.env.example`).

## Architecture

```text
NASA Exoplanet Archive (TAP) ─► catalog snapshot ─► labelled KOI manifest
MAST (Lightkurve) ─► raw FITS ─► preprocessing ─► processed parquet ─► BLS / perturbations ─► run folders
                                                           └──────────► FastAPI (read-only) ─► Next.js + Plotly
```

- `src/exoreliability/`: library (data access, preprocessing, BLS, perturbations, provenance)
- `scripts/`: one reproducible CLI per pipeline stage
- `configs/`: validated YAML for data selection and experiments
- `apps/api`, `apps/web`: read-only API and web interface
- `docs/`: [architecture](docs/architecture.md), [data contract](docs/data_contract.md),
  [experiment protocol](docs/experiment_protocol.md), [decisions](docs/decisions/)

Large data (`data/raw`, `data/interim`, `data/processed`) and experiment outputs are
git-ignored.

## Example: BLS pipeline-validation run

`make bls` writes `artifacts/experiments/<timestamp>_bls_smoke/` with `config.yaml`,
`environment.json`, `results.json`, `summary.csv`, downsampled periodograms and periodogram
figures. Each target is searched blind: detrending does not use the catalog ephemeris, and
catalog periods are compared only after the search. The same search is then repeated with
Gaussian noise added at severities 0.5, 1 and 2. This checks that the pipeline runs end to
end on a few individual stars. It is not an estimate of detection performance.

## Scientific limitations (current)

- Labels (`dr25_conservative_v1`) use archive CONFIRMED vs DR25 FALSE POSITIVE and exclude
  unresolved candidates. Confirmed planets are biased toward high-SNR signals.
- The running-median detrender partially absorbs transit depth on strongly variable stars
  and is biased at data-gap edges (documented and tested; see `docs/progress.md`).
- BLS depths are box depths, not physical transit depths.
- No train/validation/test split exists yet, so no reported number is a generalisation estimate.

## Language

Model outputs are "model predictions" or "candidate-like signals". Nothing produced by this
repository is a planet discovery or confirmation.
