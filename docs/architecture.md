# Architecture

## Data flow (Milestone 1)

```text
NASA Exoplanet Archive (TAP)                     MAST (Lightkurve search, astroquery download)
          │                                                     │
  data/archive.py ──► data/raw/catalogs/*.csv + .meta.json      │
          │                                                     │
  data/catalog.py  (labels, seeded subset)                      │
          │                                                     │
          ▼                                                     ▼
  data/interim/manifests/<name>_kois.csv ──────────► data/mast.py ──► data/raw/lightcurves/kepler/kic_*/
                                                                        *_llc.fits + products.json
                                                                                   │
                                                  preprocessing/pipeline.py (clean → normalize → detrend → clip)
                                                                                   │
                                                  data/processed/lightcurves/kic_*/lightcurve.parquet + preprocessing.json
                                                          │                            │
                                               plotting.py (figures)       baselines/bls.py (+ perturbations/)
                                                          │                            │
                                               artifacts/figures/          artifacts/experiments/<run>/
                                                          └───────────┬────────────────┘
                                                              targets.py (read-only repository)
                                                                      │
                                                          apps/api (FastAPI, GET only)
                                                                      │
                                                          apps/web (Next.js + Plotly)
```

Each arrow is a script in `scripts/`. Every stage reads files written by the previous stage
and writes its own provenance record. No stage mutates another stage's outputs.

## Components

| Path | Responsibility |
|---|---|
| `src/exoreliability/config.py` | Project paths and Pydantic models for every YAML config (validated before any work) |
| `data/archive.py` | Only module that calls the archive. TAP query building, CSV parsing, snapshot caching with checksum |
| `data/catalog.py` | Label rule `dr25_conservative_v1`, deterministic stratified subsets, KOI ephemerides |
| `data/mast.py` | Product search/selection, cached downloads with per-target manifest, raw FITS reader |
| `data/cache.py` | Deterministic paths, SHA-256, atomic JSON writes |
| `data/contracts.py` | `LightCurveData`, `Ephemeris` |
| `preprocessing/` | `clean`, `normalize`, `detrend`, `phase_fold`, `resample` (each independently tested), `pipeline` (orchestration + persistence) |
| `baselines/bls.py` | Wrapper around `astropy.timeseries.BoxLeastSquares`: period grid, parallel periodogram, stats, catalog comparison |
| `perturbations/` | `Perturbation` protocol + `GaussianNoise` |
| `experiments/results.py` | Timestamped run folders with `config.yaml` + `environment.json` |
| `training/reproducibility.py` | Seeding and environment capture (torch-optional) |
| `targets.py` | Read-only service over local files, used by the API |
| `plotting.py` | Matplotlib diagnostic figures |
| `apps/api` | FastAPI: `/health`, `/targets`, `/targets/{id}`, `/targets/{id}/lightcurve`, `/targets/{id}/phase-folded`, `/targets/{id}/bls` |
| `apps/web` | Next.js (App Router, strict TS). Client-side fetches to the API, Plotly (`scattergl`) plots |

## Design rules in effect

- **No network at import time.** Lightkurve and astroquery are imported inside functions.
- **No network in HTTP handlers.** The API only reads what the scripts produced. A missing
  target is a 404 with a message, never a download.
- **Raw data is immutable.** FITS files are never modified. Cleaning is a recorded step.
- **Injectable I/O.** `fetch_target` takes `searcher`/`downloader` callables and
  `run_tap_query` takes an `httpx.Client`, so tests run offline.
- **Determinism.** Subsets use a seeded sample over a sorted table. Perturbations take an
  explicit `numpy.random.Generator`. Per-(target, severity) seeds are derived from
  `[seed, kepid, round(1000·severity)]`. Parallel BLS gives the same result as serial (tested).

## Performance notes

A full-baseline Kepler light curve (~50–65k cadences) with the default grid
(~480k periods × 6 durations) takes ~5 min in one process and ~50–60 s with 16 processes
(`bls.n_jobs: 0`). The grid size is fixed by `oversample`, the period range and the
baseline. `max_grid_size` guards against accidental blow-ups.

## Not yet present (by design)

Splits, PyTorch dataset/model/training, calibration, the sweep runner, and the prediction
and experiment endpoints. The target tree in `CLAUDE.md` lists where they will go.
