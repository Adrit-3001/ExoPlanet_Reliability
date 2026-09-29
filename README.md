# ExoReliability Lab

**When should we trust a model that says it has found a transiting exoplanet, and under
what observational conditions does it fail?**

ExoReliability Lab studies the *reliability* of transit detection in Kepler light curves:
calibration, robustness to controlled perturbations (noise, missing cadence, shallower
transits), and failure modes. It compares learned models against a classical Box Least
Squares (BLS) baseline. It is not a leaderboard project, and model scores are not
probabilities that a planet exists.

> **Status: Milestone 2 (reproducible ML dataset).** No neural network has been trained.
> The repository builds a KOI-level dataset from real Kepler DR25 light curves: strict
> labels, transit-preserving detrending, star-grouped (leakage-safe) splits, fixed-length
> phase-folded views and a PyTorch `Dataset`. It also has the Milestone 1 BLS baseline, API
> and web page. See [`docs/progress.md`](docs/progress.md).

## Quick start

Requirements: Python ≥ 3.11 (developed on 3.12), PyTorch (installed by `make install`),
Node.js ≥ 20 for the web app (developed on 24 LTS), internet access for the data-fetch steps
only. CUDA is not needed for anything so far.

```bash
# 1. Backend
make install                     # creates .venv and installs the package with dev tools

# 2. Tiny real-data pipeline (~20 MB download, ~12 min in total, dominated by BLS)
make catalog                     # DR25 KOI table (cached) + 20-KOI manifest (10 planet / 10 FP)
make lightcurves                 # Kepler long-cadence FITS for 3 stars from the manifest
make preprocess                  # clean/normalize/detrend -> data/processed, plots -> artifacts/figures
make bls                         # blind BLS + noise demo -> artifacts/experiments/<timestamp>_bls_smoke

# 3. ML dataset (Milestone 2)
make splits                      # KIC-grouped train/val/test over the whole catalog (once; never overwritten)
make dataset-smoke               # 9 stars, seconds
make dataset-small               # 103 stars / 130 KOI examples (~700 MB download on first run, ~1 min build)
make dataset-report              # counts, leakage checks, data quality, bias analysis, figures
make detrend-eval                # synthetic transit-preservation benchmark + real-star comparison

# 4. Checks (offline)
make test                        # pytest: unit + API integration tests on synthetic fixtures
make lint typecheck              # ruff + mypy

# 5. Run the app
make api                         # FastAPI on http://localhost:8000 (docs at /docs)
make install-web && make web     # Next.js on http://localhost:3000
```

The underlying scripts accept options, for example
`python scripts/fetch_catalog.py --limit 20`,
`python scripts/fetch_lightcurves.py --kepid 10811496 --quarters 1 2 3`,
`python scripts/run_bls.py --kepid 5374854 --no-noise-demo`.
Re-running a fetch step reuses cached files without network access.

Loading the dataset for training (next milestone):

```python
from exoreliability.training.dataset import KOIDataset, make_dataloader

train = KOIDataset("data/processed/datasets/kepler_dr25_small", "train")  # [2, 2048] per example
batch = next(iter(make_dataloader(train, batch_size=32, shuffle=True, seed=42)))
batch["global"].shape, batch["label"].shape  # (32, 2, 2048), (32,)
```

Web-app settings: copy `apps/web/.env.local.example` to `apps/web/.env.local` if the API is
not at `http://localhost:8000`. The API's allowed origins are set with
`EXORELIABILITY_CORS_ORIGINS` (see `.env.example`).

## Architecture

```text
NASA Exoplanet Archive (TAP) ─► catalog snapshot ─► label policy ─► KIC-grouped splits (fixed)
MAST (Lightkurve) ─► raw FITS ─► masked spline detrending ─► per-KOI global/local views ─► dataset (.npy + parquet) ─► PyTorch
                            └─► blind running median ─► BLS / perturbations ─► run folders ─► FastAPI ─► Next.js + Plotly
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

- Labels (`dr25_clean_v2`): archive CONFIRMED (and DR25 CANDIDATE) vs FALSE POSITIVE in
  both fields. Unresolved candidates and conflicts are excluded from training but kept.
- Measured biases a model could exploit: multiplicity (planets sit on multi-KOI stars,
  AUC 0.74), planet radius and impact parameter (FPs are large/grazing eclipsing
  binaries), period (FPs shorter). **FPs have higher SNR than planets**, not lower. See
  `docs/data_contract.md` §9.
- The masked spline detrender preserves transit depth to ≤ 2% systematic error on
  realistic variability, but fails on very fast rotators (~2-day, 1% amplitude). The BLS
  path still uses the blind running median, which loses up to ~30% of depth on variable
  stars.
- The small dataset (130 examples) is for pipeline validation. It is too small for
  reliable model comparisons.
- BLS depths are box depths, and BLS takes ~47 s per star, which is too slow for the full
  dataset without a methodology change.

## Language

Model outputs are "model predictions" or "candidate-like signals". Nothing produced by this
repository is a planet discovery or confirmation.
