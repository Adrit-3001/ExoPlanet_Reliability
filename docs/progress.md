# ExoReliability Progress

Last updated: 2026-09-29 (Milestone 2 session). Milestone 1 was committed as `f07d91e`.
Nothing from Milestone 2 is committed yet.

## Completed

### Milestone 0–1 (previous session; unchanged unless noted)

Repository foundation; DR25 catalog access with checksummed snapshot; MAST light-curve
retrieval with caching; preprocessing; blind BLS baseline with a Gaussian-noise
demonstration; FastAPI + Next.js viewer; offline tests.

### Milestone 2 — reproducible ML dataset

- **Audit** of the existing pipeline (findings under Decisions and Known Issues).
- **Label policy `dr25_clean_v2`** (`data/labels.py`): table-driven with an explicit
  vocabulary; `NOT DISPOSITIONED` handled; unknown values raise. Verified identical to v1
  on all 8,054 real rows (2,675 positive / 3,964 negative / 1,415 excluded). 19 tests.
- **Transit-preserving detrending** (`robust_spline_trend`): masked cubic B-spline,
  0.3-d knots, knots kept out of data holes, local-median outlier clipping. Passes the
  documented acceptance criterion on all 7 synthetic acceptance cases × 10 seeds.
  Diagnostic figure produced. On 138 real KOIs the median depth/catalog ratio went
  0.897 → 0.966. Validity limit found and enforced (mask window ≤ 2 d). 16 tests + 1.
- **KOI-level learning unit**: one example per KOI, folded on its own ephemeris. Multi-KOI
  stars give several examples that share the star's split.
- **Multi-KOI contamination analysis** with per-example diagnostics and a figure. Evidence:
  other KOIs fold incoherently except same-period EB pairs and few-cycle long-period KOIs.
- **Leakage-safe splits** (`data/splits.py`, `scripts/create_splits.py`): KIC-grouped,
  stratified by star label composition, seed 42, over the whole catalog; checksummed;
  overwrite-protected. 0 shared KICs, 0 duplicate KOIs. 16 tests.
- **Fixed-length representation**: global view 2048 bins (default input `[2, 2048]` =
  flux + observation mask) and a stored local view 201 bins over ±2 durations. Resolution
  justified from catalog statistics.
- **Missing-bin policy**: NaN + counts stored; interpolation + mask channel at load time;
  coverage-based `example_status`.
- **Dataset build** (`scripts/build_dataset.py`): smoke / small / full configs,
  budgeted downloads, per-star failure isolation, parallel and deterministic (a rebuild is
  bit-identical).
- **Dataset storage**: `.npy` (memory-mapped) + `examples.parquet` + `dataset.json` with
  checksums and provenance.
- **Validation and bias report** (`scripts/dataset_report.py`): counts, leakage, data
  quality, population- and dataset-level bias statistics, and figures.
- **PyTorch `KOIDataset` / `make_dataloader`**: CPU, memory-mapped, seeded, optional
  metadata. `next(iter(loader))` verified on real data. PyTorch 2.14.0+cu130 installed
  (CUDA visible: RTX 4060) but not used for training.
- **BLS profiling** (`scripts/profile_bls.py`): time breakdown and scaling; no methodology change.
- **API**: `GET /datasets`, `GET /datasets/{name}` (read-only summaries). Tested.

### Real data built this milestone

| | smoke | small |
|---|---:|---:|
| stars | 9 | 103 (100 sampled + 3 anchors) |
| KOI examples | 16 | 130 |
| OK examples | 15 | 129 |
| trainable (planet / FP) | 13 (11 / 2) | 125 (73 / 52) |
| train / val / test trainable | 9 / 3 / 1 | 79 (43/36) / 23 (15/8) / 23 (15/8) |
| stars per split | 6 / 2 / 1 | 72 / 16 / 15 |
| multi-KOI stars | 4 | 17 (44 examples; includes a 5-KOI system) |
| failed stars | 0 | 0 |
| examples flagged `detrend_mask_too_long` | 1 (K05802.01, FP, 86 h) | 1 (K01439.01, planet, 24.1 h) |
| excluded (candidates, kept) | 2 | 4 |

The 103 small stars needed 695 MB (1,605 FITS files), and all downloaded successfully.

## Partially Completed

- **CI** workflow updated (CPU PyTorch). It has still never run on GitHub.
- **BLS scalability:** profiled; the required methodology change (period grid) is only
  proposed.
- **Multi-KOI masking strategy:** proposed (data_contract §10), not implemented, pending
  evidence from model evaluation.

## Not Started

- Milestone 3+: CNN, training loop, checkpointing, metrics, calibration, reliability
  sweeps, other perturbations, `/predict` and `/experiments` endpoints, frontend dataset
  view.
- Full-scale dataset build (~5,800 stars, ~40 GB). The config exists; it was deliberately
  not run.
- Notebooks, docker-compose, LICENSE (MIT in `pyproject.toml` is still a placeholder).

## Decisions Made

- **Label policy** `dr25_clean_v2` (identical labels to v1, stricter semantics). The
  premise that v1 made DR25 CANDIDATEs positive was checked: every v1 positive is
  archive-CONFIRMED.
- **Learning unit:** one KOI signal per example. **Split unit:** the star (KIC), assigned
  once across the whole catalog.
- **Detrending:** masked robust spline for the ML dataset. The blind running median stays
  for BLS (a blind search may not use ephemerides).
- **Mask all catalogued KOIs on a star during trend fitting**, including candidates and
  FPs. Do not remove siblings from folds.
- **Acceptance thresholds:** systematic depth error ≤ 2% (≈ 0.6σ of the median positive
  KOI's statistical depth error), random ≤ 5% RMS. The latter was set after observing the
  long-transit case, and is disclosed as such.
- **Detrending validity rule:** mask windows > 2 d (transits > 24 h) →
  `detrend_mask_too_long`, excluded from training, kept in the table (limit mapped on
  synthetic data after K05802.01 failed on real data).
- **Representation:** global 2048 (default input) + local 201; depth normalisation per
  example; interpolation + observation mask for empty bins.
- **Storage:** `.npy` + Parquet + JSON; no HDF5/Zarr dependency.
- **Small dataset:** star-balanced (50 with a planet / 50 FP-only, apportioned by split)
  plus the 3 Milestone 1 anchor stars. The full scale uses the natural population.
- **PyTorch:** CUDA build installed locally (next milestone trains on the GPU); CI uses
  the CPU wheel.
- mypy targets Python 3.12 syntax; Node.js is user-local (unchanged from Milestone 1).

## Known Issues

1. **Detrending validity regime.** Examples with transits > 24 h are excluded
   (`detrend_mask_too_long`; 127 population KOIs, 98% FP). Strongly variable stars with
   few transits, or orbits synchronised with rotation, can still carry large depth errors.
   They are flagged by metadata only (`star_trend_rms_ppm`, `n_transits_observed`) and must
   be stratified in evaluation.
2. **Masked detrending depends on catalog ephemerides.** Wrong ephemerides or strong TTVs
   leave transits partly unmasked. This has not been quantified on real data.
3. **Detrending fails for very fast rotators** (1% amplitude, ~2-d rotation): +45% mean
   depth error on the synthetic extreme case. Such stars are not flagged yet.
4. **Trend-estimate noise:** ~27 ppm RMS on a 100-ppm star (+~4% total noise).
5. **Multiplicity shortcut risk:** planets sit on multi-KOI stars (AUC 0.74). Sibling
   residue in global views could let a model partly infer multiplicity. Metrics must be
   stratified by single/multi-KOI stars.
6. **Class-dependent catalog variables:** planet radius / impact parameter (sep. ~0.70),
   period (AUC 0.66). FPs have *higher* SNR and deeper transits than planets. Nothing is
   rebalanced.
7. **Few-cycle long-period KOIs** (e.g. K01788.02, P = 369 d, 8 quarters) have
   sibling-contaminated global views and barely pass coverage (global coverage 0.50–0.54
   for a few examples).
8. **Small dataset size:** val/test have only 23 trainable examples each (8 FPs), which is
   far too few for meaningful metrics. It exists to validate the pipeline.
9. **BLS runtime** (47 s/star, parallelism saturating at ~4.75×) blocks full-scale BLS
   evaluation without a grid change.
10. BLS results from Milestone 1 used the blind running median. They are not comparable to
   ML inputs detrended differently (ADR 0002).
11. **Disk:** the per-star light curves stored with each dataset take ~3.3 MB per star
    (~19 GB at full scale, on top of ~40 GB of raw FITS). Arrays themselves are small.
12. The Starlette `TestClient`/httpx deprecation warning persists (tests pass).

## Commands Verified

Run 2026-09-29 in this environment (Python 3.12.3, PyTorch 2.14.0+cu130, Node 24.21.0):

| Command | Result |
|---|---|
| `python scripts/create_splits.py` | splits written; a second run refused without `--overwrite` |
| `python scripts/fetch_lightcurves.py --allow-large --limit 200 --kepid …` (103 stars) | 103/103 available, ~16 min |
| `python scripts/build_dataset.py --config configs/data/kepler_dr25_smoke.yaml` | 16 examples, 5.9 s |
| `python scripts/build_dataset.py --config configs/data/kepler_dr25_small.yaml --overwrite` | 130 examples, ~50–60 s; rebuild bit-identical |
| `python scripts/dataset_report.py --dataset kepler_dr25_small` | report + 4 figures; all leakage checks 0 |
| `python scripts/evaluate_detrending.py --seeds 10 --real-limit 200` | passes acceptance; 138 real KOIs compared; see data_contract §6 |
| `python scripts/profile_bls.py --kepid 5374854` | profile.json; full run 47 s |
| `KOIDataset(...)` + `make_dataloader` on real small dataset | batches `[16, 2, 2048]` / `[16, 2, 201]` |
| `pytest` | 143 passed (offline) |
| `ruff check .`, `ruff format --check .`, `mypy` | clean (45 files) |
| `npm run lint`, `npm run typecheck` (apps/web) | clean |

## Next Recommended Milestone

**Milestone 3: CNN baseline**, but first decide the training-set scale:

1. The 125-example small dataset is enough to *debug* a training loop, not to measure
   anything. Before training for results, build a larger dataset: e.g. a "medium" config
   of ~1,000–1,500 stars (~7–10 GB, ~3–4 h download), or the full ~5,800 stars
   (~40 GB). This needs your go-ahead for the download size.
2. Implement the 1D CNN on the `[2, 2048]` global view (CLAUDE.md §10), BCE-with-logits,
   early stopping on val loss, seeded, with checkpointing.
3. Evaluation with the mandatory stratifications (experiment_protocol §5): single vs
   multi-KOI stars, FP-flag type, period/depth/SNR bins.
