# Experiment Protocol

Status: Milestone 2 (dataset foundation). BLS pipeline-validation runs and the ML dataset
exist; no model has been trained. Sections marked *planned* are commitments for later
milestones and are not implemented.

## 1. General rules

- Every run writes a folder `artifacts/experiments/<UTC timestamp>_<name>/` containing the
  validated `config.yaml`, `environment.json` (git commit + dirty flag, Python and package
  versions, torch/CUDA/GPU if installed, catalog snapshot metadata) and its results.
- Configs are validated (Pydantic) before any computation or download.
- No parameter is tuned on test data. The test split (`data/splits/test.csv`) exists
  but has not been used for any decision. Detrending and representation choices were
  made on **synthetic** light curves and catalog statistics only.
- **No result so far is a performance estimate.**
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

For the ML dataset, perturbations will need to be applied to the stored star-level light
curves (`data/processed/datasets/<name>/lightcurves/`), followed by re-folding. Whether
noise should be injected before or after detrending is a Milestone 5 design decision, to
be documented here.

### 3.2 Planned (Milestone 5)

Missing cadence (random fraction, then contiguous gaps), transit-depth attenuation
(in-transit points only, using the catalog ephemeris; baseline preserved), stellar
variability, outliers/artifacts. Each will get documented severity semantics and unit tests.

## 4. BLS runtime (profiled, not changed)

`scripts/profile_bls.py` → `artifacts/reports/bls_profiling/profile.json` (KIC 5374854,
51,747 cadences, 1,459-d baseline, 483,928-period production grid):

- 100% of the time is inside Astropy's compiled `bls_fast` (a single call).
  `compute_stats` takes 0.06 s, and Python overhead is negligible.
- Runtime is linear in the number of periods (≈ 0.49 s per 1,000 periods single-process,
  ≈ 238 s for the full grid). It depends only weakly on the number of durations
  (1 → 6 durations: 10.0 → 12.2 s on a 1/20 sub-grid), because per-period cost is
  dominated by folding all N points.
- Process-level parallelism saturates: 4 workers 2.6×, 8 → 3.5×, 16 → 4.75× (10 physical
  cores). Every worker count gives periodograms bit-identical to single-process. The
  default (`n_jobs: 0` = all CPUs) is already the fastest safe setting: **47 s per full
  run**.
- Astropy's internal `oversample` (phase-binning resolution) 10 → 5 saves only 15%, and it
  changes numerical precision. Measured, not adopted.

The only large lever is the period grid (Δln P = D_min / (3·T) ⇒ ~480k periods).
Candidate methodology changes, each to be benchmarked for recovery before adoption:
Ofir (2014)-style optimal frequency sampling, a duration grid that scales with period,
`oversample` 3 → 2, or a GPU BLS. At 47 s per star the full dataset (~5,800 stars) would
take ~75 h on this laptop, so a change will be needed before BLS is evaluated at full
scale. Any change will be recorded here as a methodology change.

## 5. Evaluation commitments for the first CNN (Milestones 3–5)

- **Data:** `data/processed/datasets/<name>/`, splits from `data/splits/` (KIC-grouped,
  fixed). Training uses `split == "train"` and `training_status == "train_eligible"`.
  Model selection and calibration use `val`. `test` is used once per reported result.
- **Input:** global view `[2, 2048]` (normalised flux + observation mask), depth-normalised
  (`data_contract.md` §7–8).
- Discrimination: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix.
- Calibration: Brier score, log loss, reliability diagram, ECE (binning scheme to be fixed
  here before first use). Calibrators are fitted on validation data only.
- Robustness: metric-vs-severity curves with multiple seeds per severity. Uncertainty via
  bootstrap over **stars** (not KOIs), because KOIs on one star are correlated.
- BLS and learned models are evaluated on the same held-out stars.
- **Mandatory stratified reporting, driven by measured biases (`data_contract.md` §9):**
  - single-KOI vs multi-KOI stars (multiplicity AUC 0.74);
  - period, depth, duration, model SNR and Kepler magnitude bins;
  - negatives by FP flag (not-transit-like / stellar eclipse / centroid / ephemeris match);
  - examples with `contam_max_relative_to_depth > 0.25`.
  - stellar variability (`star_trend_rms_ppm`) × number of observed transits
    (`n_transits_observed`), the regime where masked detrending is least reliable.
- A result that is good overall but driven by one easy stratum (e.g. deep eclipsing
  binaries) must be reported as such.
