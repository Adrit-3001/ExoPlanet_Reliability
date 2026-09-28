# Experiment Protocol

Status: Milestone 1. Only the BLS pipeline-validation run exists. Sections marked
*planned* describe commitments for later milestones and are not implemented.

## 1. General rules

- Every run writes a folder `artifacts/experiments/<UTC timestamp>_<name>/` containing the
  validated `config.yaml`, `environment.json` (git commit + dirty flag, Python and package
  versions, torch/CUDA/GPU if installed, catalog snapshot metadata) and its results.
- Configs are validated (Pydantic) before any computation or download.
- No parameter is tuned on test data. Before Milestone 2 there is no test split, so
  **no result so far is a performance estimate.**
- Single-object outcomes are reported as pipeline checks, never as detection rates.

## 2. BLS baseline (implemented)

**Input:** `flux_detrended` from preprocessing. Its detrending does *not* use the catalog
ephemeris (`mask_known_transits: false`), so the search is blind to catalog information.

**Search** (`astropy.timeseries.BoxLeastSquares`, objective `likelihood`):

- Periods: log-uniform from `min_period_days` (0.5) to
  `min(max_period_days (50), baseline / (min_transits − 1))`, step
  Δln P = `min(durations) / (oversample · baseline)`. This keeps the accumulated phase drift
  across the baseline below `min_duration / oversample`.
- Durations: 1, 2, 3, 4, 6, 8 h (all must be shorter than the minimum period).

**Reported per run:** best period, duration, mid-transit epoch (BKJD), box depth ± error,
depth SNR, peak log-likelihood, SDE = (peak − mean)/std of the raw periodogram, odd/even
depths, harmonic Δlog-likelihood, number of transits with data, grid description and top-5
distinct peaks.

**Catalog comparison** (after the search): for every KOI on the star, compute
P_BLS / P_catalog and label it `match` (ratio 1 ± 0.1%), `harmonic n/m` (n, m ≤ 4) or `none`.
BLS box depth is not a limb-darkened model depth, so depths are not compared numerically.

## 3. Perturbations

### 3.1 Gaussian observational noise (implemented)

Severity `s` adds i.i.d. N(0, (s·σ_ref)²) to every flux sample. σ_ref is the light curve's own
point-to-point scatter, `1.4826 · MAD(Δflux) / √2`, or a caller-supplied `sigma_ref`.
For white input noise the total scatter becomes σ_ref·√(1+s²). s = 1 ≈ a star 0.75 mag
fainter in the photon-limited regime. s = 0 is an exact copy and draws no random numbers.
Flux errors are inflated in quadrature by default. Sample count and timestamps are
unchanged. The noise is applied after preprocessing, to the detrended light curve.

The Milestone 1 demonstration uses severities 0, 0.5, 1, 2 with one seeded realisation each.
This is a single realisation per target, so it cannot support statistical statements.

### 3.2 Planned (Milestone 5)

Missing cadence (random fraction, then contiguous gaps), transit-depth attenuation
(in-transit points only, using the catalog ephemeris; baseline preserved), stellar
variability, outliers/artifacts. Each will get documented severity semantics and unit tests.

## 4. Planned evaluation (Milestones 2–5)

- Grouped (by `kepid`), label-stratified train/val/test splits, saved to `data/splits/`
  and never regenerated implicitly.
- Discrimination: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix.
- Calibration: Brier score, log loss, reliability diagram, ECE (binning scheme to be fixed
  in this document before first use). Calibrators (temperature, Platt, isotonic) are fitted
  on validation data only.
- Robustness: metric-vs-severity curves with multiple seeds per severity, and uncertainty
  from bootstrap over targets (resampling stars, not KOIs).
- BLS and learned models are evaluated on the same held-out targets, with a BLS decision
  statistic chosen on validation data.
- Stratified reporting by depth, period, duration, Kepler magnitude and model SNR, keeping
  in mind that positive labels are biased toward high SNR (see `data_contract.md`).
