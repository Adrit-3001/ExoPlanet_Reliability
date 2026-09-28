# ExoReliability Progress

Last updated: 2026-09-28 (Milestones 0–1 session).

## Completed

- **Repository foundation (M0):** `src/exoreliability` package, `pyproject.toml` (setuptools,
  pytest, Ruff, mypy), `.gitignore` (raw/interim/processed data and experiment/model
  artifacts ignored; `.gitkeep` skeleton kept), `.env.example`, `Makefile`, docs, configs,
  test layout, CI workflow file.
- **Catalog access (M1):** TAP client with an injectable HTTP client. The DR25 KOI schema
  was verified from `TAP_SCHEMA`. Full-table snapshot is cached with query, timestamp and
  SHA-256, and the checksum is verified on reload. Real snapshot fetched: 8,054 KOIs /
  6,923 stars.
- **Label rule `dr25_conservative_v1`:** implemented, tested and documented, with original
  dispositions preserved. **Needs owner review** (see Decisions).
- **Deterministic KOI manifest:** seeded, label-stratified, order-independent. Real
  manifest: 20 KOIs (10 planet / 10 FP).
- **MAST retrieval:** Lightkurve search, astroquery download to deterministic paths, a
  per-target `products.json`, offline cache reuse (verified: second run made no network
  calls), graceful per-target failures, a download budget, and a 25-target hard cap. Real
  data: 44 long-cadence FITS files (17 + 14 + 13 quarters, ~19 MB) for KIC 4049108,
  5374854 and 5689351, all `DATA_REL = 25`.
- **Preprocessing:** finite + quality-bitmask cleaning, per-quarter median normalization,
  segment-aware running-median detrending (optional catalog-transit masking, off by
  default), upward-only outlier clipping, phase folding, fixed-length binning. Persisted
  with a full provenance record.
- **Diagnostic plots** for 3 real targets: `artifacts/figures/kic_*_diagnostic.png`.
- **BLS baseline:** Astropy wrapper, drift-limited log-period grid, deterministic parallel
  periodogram, SDE, odd/even depths, top peaks, catalog period comparison. Run on 3 real
  targets (see below).
- **Gaussian-noise perturbation:** `Perturbation` protocol, documented severity semantics,
  seeded, severity 0 is identity, tested.
- **API:** `/health`, `/targets`, `/targets/{id}`, `/targets/{id}/lightcurve`,
  `/targets/{id}/phase-folded`, `/targets/{id}/bls`. Integration-tested on a synthetic
  fixture project and manually verified against the real processed data.
- **Web:** target list and target page (KOI table with both dispositions and derived label,
  light curve, phase fold on catalog or BLS ephemeris, BLS metrics, periodogram, noise-demo
  table). Explicit loading/404/error states and no placeholder data. Screenshot of a real
  target: `artifacts/figures/web_target_kic_005374854.png`.

### Real-data BLS pipeline check (run `2026-09-28T033037Z_bls_smoke`)

One seeded noise realisation per severity. These are three individual stars, not a performance estimate.

| KIC | Manifest KOI | BLS clean period [d] | Matches catalog KOI | SNR (s=0 → 2) | SDE (s=0 → 2) | Period kept at s=2? |
|---|---|---|---|---|---|---|
| 4049108 | K00495.01 (FP, 4.80447 d) | 4.80449 | K00495.01, 1/1 | 69.7 → 30.2 | 98.4 → 87.8 | yes |
| 5374854 | K00645.01 (CONFIRMED, 8.50339 d) | 23.78310 | **K00645.02** (23.78321 d), 1/1 | 34.4 → 15.1 | 66.9 → 42.4 | yes |
| 5689351 | K00505.04 (CONFIRMED, 8.34819 d) | 13.76711 | **K00505.01** (13.76710 d), 1/1 | 53.8 → 22.2 | 71.2 → 54.0 | yes |

On multi-KOI stars the strongest BLS peak was a *different* KOI from the manifest entry.
The KOI table is per signal, but a single-peak BLS search is per star. This has to be
handled explicitly before BLS is used as a per-KOI baseline.

## Partially Completed

- **CI:** `.github/workflows/ci.yml` is written, but it has never run on GitHub (no remote
  push was made). The same commands pass locally.
- **Configs:** `configs/data/kepler_dr25_small.yaml` and
  `configs/experiments/bls_smoke.yaml` exist. `kepler_dr25_full.yaml`, the model configs and
  the other experiment configs are not created, because nothing uses them yet.
- **Detrending:** works, but has the known limitations listed below.

## Not Started

- Milestone 2: grouped train/val/test splits, fixed-length ML representations, dataset
  manifests, PyTorch Dataset, class-distribution report.
- Milestone 3+: CNN, training loop, calibration, the sweep runner, the other perturbations,
  `/predict` and `/experiments` endpoints, transformer, TESS.
- Notebooks, `docker-compose.yml`, `LICENSE` (no licence chosen; `pyproject.toml` says MIT
  as a placeholder, which the owner should confirm).

## Decisions Made

- Kepler Q1–Q17 DR25 KOI table + Kepler SOC long-cadence PDCSAP, DATA_REL 25
  (ADR 0001).
- **Labels:** positive = archive CONFIRMED ∧ DR25 CANDIDATE. Negative = FALSE POSITIVE in
  both fields. Unresolved candidates (1,358) and disposition conflicts (57) are excluded.
  This is my conservative choice under the spec and needs owner sign-off before supervised
  training.
- Full catalog snapshot fetched once and subsets drawn locally, which is more reproducible
  than server-side `top N` queries.
- BLS input detrending is blind to the catalog ephemeris. Catalog ephemerides are used only
  for display folds and the post-hoc comparison.
- Gaussian noise is applied after preprocessing. σ_ref is the light curve's own robust
  point-to-point scatter, and flux errors are inflated in quadrature.
- BLS period grid: Δln P = min duration / (3 · baseline), ~480k periods for a 4-year
  baseline. Parallelised rather than coarsened.
- PyTorch is not installed yet (optional extra `ml`), following the "no dependency without
  a need" rule.
- mypy targets Python 3.12 syntax because NumPy ≥ 2.3 stubs need it. The package still
  declares ≥ 3.11.
- Frontend pinned to TypeScript 5.9 and ESLint 9 rather than the new TS 7 / ESLint 10 majors.
- Spec file `CALUDE.md` was renamed to `CLAUDE.md` (its contents are identical to `AGENTS.md`).

## Known Issues

1. **Detrending absorbs depth on strongly variable stars.** When the trend changes within the
   1.5-day window by much more than the noise, the running median shifts at in-transit
   points. On a synthetic 1% / 20-day sinusoid, ~20% of a 1,000 ppm depth was absorbed
   (`test_running_median_removes_slow_trend_but_absorbs_depth_on_steep_gradients`).
   KIC 5689351 (~1% spot modulation) is in this regime. Options: robust local-linear
   filter (e.g. biweight with slope), iterative in-transit rejection, or catalog masking
   for non-blind uses only.
2. **Edge bias:** the window is one-sided at gap/quarter edges, which biases the trend by
   ~slope × window/4 (tested).
3. BLS box depths are 25–30% below catalog depths for the three targets (for example
   188 vs 257 ppm). This is expected in part (box vs limb-darkened model) and in part
   probably Issue 1. It is not investigated further.
4. BLS runtime is ~50–60 s per target per severity on 16 CPUs. Sweeps over hundreds of
   targets will need a coarser or adaptive grid, or light-curve binning. Any such change
   must be recorded as a methodology change.
5. The periodogram SDE is computed on the raw power spectrum, and power rises with period.
   SDE is comparable only within one grid configuration.
6. Starlette warns that `httpx`-based `TestClient` is deprecated (tests still pass).
7. Node.js is not a system install on the dev machine. It was placed at
   `~/.local/opt/node-v24.21.0-linux-x64` and must be on `PATH` for `make web*`.
8. Experiment `environment.json` shows `git.dirty: true` and the initial commit, because
   nothing from this session has been committed yet.

## Commands Verified

Run on 2026-09-28 in this environment (Linux, Python 3.12.3, Node 24.21.0, RTX 4060 present
but unused):

| Command | Result |
|---|---|
| `python scripts/fetch_catalog.py --limit 20` | real TAP query, 8,054 rows, manifest written. Re-run from cache gives a byte-identical manifest |
| `python scripts/fetch_lightcurves.py --limit 3` | 44 files downloaded (~33 s). Re-run: all `cached`, no network |
| `python scripts/preprocess_lightcurves.py` | 3 targets processed + 3 figures |
| `python scripts/run_bls.py` | 12 BLS runs (3 targets × 4 severities), ~10.5 min |
| `pytest` | 81 passed (offline) |
| `ruff check .`, `ruff format --check .` | clean |
| `mypy` | no issues (35 files) |
| `npm run lint`, `npm run typecheck`, `npm run build` (apps/web) | clean / success |
| `uvicorn apps.api.main:app` + curl on all endpoints with real data | 200s; 404/422 on bad input |
| `next dev` + headless Chrome on `/target/5374854` | page rendered with real data (screenshot saved) |

## Next Recommended Milestone

**Milestone 2: reproducible ML dataset**, after the owner confirms the label rule. First
steps:

1. Owner review of `dr25_conservative_v1` and the Known Issues 1–2 detrending limitation.
   Decide whether to improve the detrender before generating fixed-length examples, since
   those will inherit its depth bias.
2. Decide the example unit (per KOI, with other KOIs on the same star masked?) and the
   representation (global + local phase-folded views on the catalog ephemeris is the usual
   choice).
3. Implement `data/splits.py`: grouped by `kepid`, label-stratified, seeded, written once
   to `data/splits/`, with tests for no overlap and determinism.
4. Scale the fetch to a "small" config (a few hundred stars) with the existing budget caps.
