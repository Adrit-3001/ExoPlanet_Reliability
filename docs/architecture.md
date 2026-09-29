# Architecture

## Data flow

```text
NASA Exoplanet Archive (TAP)
  └─ data/archive.py ─► data/raw/catalogs/q1_q17_dr25_koi.csv (+ .meta.json, SHA-256)
        └─ data/labels.py (policy dr25_clean_v2)
              └─ data/splits.py ─► data/splits/{train,val,test}.csv + split_metadata.json   (once, whole catalog, KIC-grouped)
                    └─ data/selection.py (stars for smoke/small/full)
                          └─ data/mast.py ─► data/raw/lightcurves/kepler/kic_*/ (*.fits + products.json)
                                └─ data/examples.py::build_star (parallel per star)
                                      preprocessing/pipeline.py: clean → normalise → masked robust spline → clip
                                      preprocessing/views.py: per KOI → global [2048] + local [201] views, coverage, contamination
                                └─ data/processed/datasets/<name>/ (examples.parquet, *.npy, lightcurves/, dataset.json)
                                      ├─ training/dataset.py: KOIDataset / DataLoader → tensors [2, 2048] (+ [2, 201])
                                      └─ scripts/dataset_report.py → artifacts/reports/<name>/

Milestone 1 path (unchanged): raw FITS → blind running-median preprocessing → data/processed/lightcurves/
  → baselines/bls.py → artifacts/experiments/<run>/ → targets.py → apps/api → apps/web
```

Each arrow is a script in `scripts/`. Every stage reads files written by the previous stage
and writes its own provenance record. No stage mutates another stage's outputs.

## Components

| Path | Responsibility |
|---|---|
| `src/exoreliability/config.py` | Project paths and Pydantic models for every YAML config (validated before any work) |
| `data/archive.py` | Only module that calls the archive. TAP query building, CSV parsing, snapshot caching with checksum |
| `data/labels.py` | Label policy `dr25_clean_v2` (explicit vocabulary; unknown values raise) |
| `data/catalog.py` | Deterministic KOI subsets (Milestone 1 manifest), KOI ephemerides |
| `data/splits.py` | KIC-grouped, stratified, checksummed, overwrite-protected splits over the whole catalog |
| `data/selection.py` | Star selection per dataset scale from the fixed split assignment |
| `data/examples.py` | Per-star build (one example per KOI), dataset storage and loading |
| `data/mast.py` | Product search/selection, cached downloads with per-target manifest, raw FITS reader |
| `data/cache.py` | Deterministic paths, SHA-256, atomic JSON writes |
| `data/contracts.py` | `LightCurveData`, `Ephemeris` |
| `preprocessing/` | `clean`, `normalize`, `detrend` (running median; masked robust spline), `phase_fold`, `resample`, `views` (global/local views, coverage, multi-KOI contamination), `pipeline` |
| `evaluation/detrending.py` | Synthetic transit-preservation benchmark (known true trend) |
| `training/dataset.py` | `KOIDataset`, missing-bin filling, per-example normalisation, seeded `DataLoader` |
| `baselines/bls.py` | Wrapper around `astropy.timeseries.BoxLeastSquares`: period grid, parallel periodogram, stats, catalog comparison |
| `perturbations/` | `Perturbation` protocol + `GaussianNoise` |
| `experiments/results.py` | Timestamped run folders with `config.yaml` + `environment.json` |
| `training/reproducibility.py` | Seeding and environment capture (torch-optional) |
| `targets.py` | Read-only service over local files, used by the API |
| `plotting.py` | Matplotlib diagnostic figures |
| `apps/api` | FastAPI: `/health`, `/targets`, `/targets/{id}`, `/targets/{id}/lightcurve`, `/targets/{id}/phase-folded`, `/targets/{id}/bls`, `/datasets`, `/datasets/{name}` |
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

- BLS: ~47 s per full-baseline star with all CPUs, ~4 min single-process. Profiling is in
  `experiment_protocol.md` §4.
- Dataset build: ~4 s of CPU per star (spline detrending + views). The 103-star small dataset
  builds in ~50 s with 8 worker processes (parallelism does not change results).
- `KOIDataset` memory-maps arrays and prepares each item on the CPU on demand.

## Not yet present (by design)

Models, training loop, calibration, the sweep runner, and the prediction and experiment
endpoints. The target tree in `CLAUDE.md` lists where they will go.
