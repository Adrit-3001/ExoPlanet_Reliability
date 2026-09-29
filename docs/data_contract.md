# Data Contract

This document defines where data comes from, how labels are derived, and what each pipeline
stage writes. Change it together with the code it describes.

## 1. Catalog source

| Item | Value |
|---|---|
| Archive | NASA Exoplanet Archive |
| Interface | TAP, synchronous endpoint `https://exoplanetarchive.ipac.caltech.edu/TAP/sync`, `format=csv` |
| Table | `q1_q17_dr25_koi` (Kepler Q1–Q17 Data Release 25 KOI table) |
| Access code | `src/exoreliability/data/archive.py` (the only module that talks to the archive) |
| Query | `select <35 columns> from q1_q17_dr25_koi order by kepoi_name` (exact text stored in the snapshot metadata) |
| Snapshot | `data/raw/catalogs/q1_q17_dr25_koi.csv` + `.meta.json` (query, endpoint, UTC retrieval time, row count, SHA-256) |

The full table (8,054 KOIs on 6,923 stars at retrieval on 2026-09-28) is small, so it is
downloaded once and every subset is drawn locally from that snapshot. The snapshot checksum is
verified on every load; a mismatch raises an error rather than silently re-downloading.

The cumulative KOI table is **not** used. All KOI rows come from the DR25 table, so the
population reflects one uniform Kepler pipeline run.

### 1.1 Verified schema

Column names, types and units were read from `TAP_SCHEMA.columns` for `Q1_Q17_DR25_KOI`
(2026-09-28), not guessed. Relevant fields:

| Column | Archive description | Unit |
|---|---|---|
| `kepid` | KIC Identification Number | — |
| `kepoi_name` | Kepler Object of Interest Name | — |
| `kepler_name` | Kepler Name (confirmed planet) | — |
| `koi_disposition` | **Exoplanet Archive Disposition** | — |
| `koi_pdisposition` | **Disposition using Kepler Data** | — |
| `koi_disp_prov` | Disposition Provenance (`q1_q17_dr25_koi` for all rows) | — |
| `koi_score` | KOI (Robovetter) score | — |
| `koi_fpflag_nt/ss/co/ec` | Not-transit-like / significant secondary / centroid offset / ephemeris-match false-positive flags | 0/1 |
| `koi_period` | Transit Period | days |
| `koi_time0bk` | Time to First Transit | BKJD = BJD − 2454833 |
| `koi_duration` | Transit Duration | hours |
| `koi_depth` | Transit Depth | ppm |
| `koi_model_snr` | Model Signal-to-Noise Ratio | — |
| `koi_kepmag` | Kepler-band magnitude | mag |
| `koi_steff`, `koi_slogg`, `koi_srad` | Stellar parameters | K, log10(cm s⁻²), R☉ |

The DR25 TCE table (`q1_q17_dr25_tce`, 34,032 rows) has an `av_training_set` column, but it is
empty for every DR25 row, so it is not a label source.

### 1.2 Observed disposition values (snapshot of 2026-09-28)

| `koi_disposition` \ `koi_pdisposition` | CANDIDATE | FALSE POSITIVE |
|---|---:|---:|
| CANDIDATE | 1,358 | 0 |
| CONFIRMED | 2,675 | 56 |
| FALSE POSITIVE | 1 | 3,964 |

Documented vocabularies (archive column documentation, re-checked 2026-09-29):

- `koi_disposition` ("Exoplanet Archive Disposition"): CANDIDATE, FALSE POSITIVE,
  NOT DISPOSITIONED or CONFIRMED. "All KOIs marked as CONFIRMED are also listed in the
  Exoplanet Archive Confirmed Planet table", so CONFIRMED status can rest on follow-up
  observations outside Kepler photometry. CONFIRMED status **is** available in the DR25
  table (2,731 rows). `kepler_name` is set for 99.9% of CONFIRMED rows and for none of
  the CANDIDATE rows.
- `koi_pdisposition` ("Disposition Using Kepler Data"): the DR25 pipeline (Robovetter)
  outcome. Documented values are FALSE POSITIVE, NOT DISPOSITIONED and CANDIDATE. It never
  contains CONFIRMED.

The two fields disagree for 57 KOIs:

- 56 archive-CONFIRMED planets that the Robovetter flagged as FALSE POSITIVE (flags among
  them: not-transit-like 25, stellar eclipse 19, centroid offset 11, ephemeris match 1).
- K00242.01 (Kepler-503 b): archive FALSE POSITIVE but DR25 CANDIDATE. It has a Kepler
  planet name, i.e. it was once a validated planet that the archive later reclassified.
  This is direct evidence that archive dispositions change after DR25, which is why the
  retrieval timestamp and checksum are recorded.

`NOT DISPOSITIONED` is documented but does not occur in the snapshot.

## 2. Label policy — `dr25_clean_v2`

Implemented in `exoreliability.data.labels` (table-driven; `catalog.assign_labels`
delegates to it). Positive = a planet in the archive's Confirmed Planets table that DR25
also passed. Negative = a false positive in both fields. Everything else is excluded from
supervised training but **kept** in every manifest and dataset.

| Archive disposition (`koi_disposition`) | DR25 disposition (`koi_pdisposition`) | Training status | Label | Reason | Count |
|---|---|---|---|---|---:|
| CONFIRMED | CANDIDATE | train_eligible | 1 `planet` | confirmed and passed DR25 vetting | 2,675 |
| FALSE POSITIVE | FALSE POSITIVE | train_eligible | 0 `false_positive` | documented FP in both | 3,964 |
| CANDIDATE | CANDIDATE | excluded | — | `unresolved_candidate` | 1,358 |
| CONFIRMED | FALSE POSITIVE | excluded | — | `disposition_conflict` | 56 |
| FALSE POSITIVE | CANDIDATE | excluded | — | `disposition_conflict` | 1 |
| CANDIDATE | FALSE POSITIVE | excluded | — | `disposition_conflict` | 0 |
| NOT DISPOSITIONED (either field) | | excluded | — | `not_dispositioned` | 0 |
| missing (either field) | | excluded | — | `missing_disposition` | 0 |
| any other value | | **error** | — | `UnexpectedDispositionError` (or `unexpected_value` if explicitly requested) | 0 |

Matching normalises case and whitespace only. Added columns: `label` (nullable Int8),
`label_name`, `training_status`, `exclusion_reason`, `label_policy`. The original
`koi_disposition` / `koi_pdisposition` columns are never modified.

**v1 → v2.** On this snapshot, v2 assigns exactly the same labels as v1 (verified
row-by-row on all 8,054 KOIs). v1 already required archive CONFIRMED for a positive; the
`koi_pdisposition == CANDIDATE` condition only removes the 56 conflicts. What changed is
strictness: v1 silently mapped any unrecognised value to `disposition_conflict`, while v2
has an explicit vocabulary, handles `NOT DISPOSITIONED` explicitly and raises on unknown
values.

Caveats:

- CONFIRMED depends on follow-up. Positives are therefore not a random sample of real
  planets. The measured consequences are in §9.
- Negatives mix astrophysical FPs (eclipsing binaries, background sources) with
  instrumental and non-transit-like signals. Among population negatives:
  not-transit-like 24.5%, stellar eclipse 52.6%, centroid offset 43.9%, ephemeris
  match 27.4% (flags overlap). These flags are kept as metadata for subgroup analysis.
- Excluding candidates makes the supervised task easier than real vetting. An experiment
  that uses candidates must define a new policy name.

## 3. Identifiers and the learning unit

- A **star** is identified by its KIC ID (`kepid`). A **KOI** (`kepoi_name`) is one
  catalogued periodic signal on a star.
- **Learning unit: one KOI signal = one example.** A star with N KOIs yields N examples,
  each folded on its own catalog period, epoch and duration. All N share the star's
  split.
- On-disk names use `kic_<9-digit zero-padded kepid>`. The API accepts `10811496`,
  `KIC 10811496` or `kic_010811496`.

## 4. Splits (leakage prevention)

`src/exoreliability/data/splits.py`, `scripts/create_splits.py`, config
`configs/data/splits_dr25.yaml`. Output: `data/splits/{train,val,test}.csv` plus
`split_metadata.json` (small, committable).

- **Group key: `kepid`.** A star is assigned to exactly one split, and all its KOIs,
  including excluded candidates, go with it.
- **Scope: the whole DR25 snapshot (6,923 stars), assigned once.** Every dataset scale
  selects stars from this fixed assignment, so growing a dataset can never move a star
  between splits.
- **Stratification:** each star gets a stratum from its train-eligible KOIs
  (`pos_only`, `neg_only`, `mixed`, `unlabelled_only`). Within a stratum, KIC IDs are
  sorted, permuted with `numpy.random.default_rng([seed, stratum_index])` (seed 42), and
  cut 70/15/15 by largest-remainder rounding (algorithm `kic_stratified_shuffle_v1`).
- **Immutability:** `write_splits` refuses to replace existing files unless
  `--overwrite` is given. `load_splits` verifies each file's SHA-256 and re-checks the
  no-leakage invariant on every load. `build_dataset.py` refuses to run if the catalog
  snapshot differs from the one the splits were made from, or if any label disagrees.

Realised split (seed 42):

| Split | Stars | KOIs | Train-eligible KOIs | Planets | FPs | Positive fraction | Share of eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 4,845 | 5,607 | 4,628 | 1,850 | 2,778 | 0.400 | 69.7% |
| val | 1,040 | 1,228 | 1,009 | 416 | 593 | 0.412 | 15.2% |
| test | 1,038 | 1,219 | 1,002 | 409 | 593 | 0.408 | 15.1% |

Leakage checks recorded in the metadata: shared KICs train∩val = train∩test = val∩test =
0, and duplicate KOI IDs = 0.

## 5. Light-curve products

| Item | Policy |
|---|---|
| Search | `lightkurve.search_lightcurve("KIC <id>", mission="Kepler", author="Kepler", exptime=1800)` |
| Product | Kepler SOC long-cadence `*_llc.fits`, one file per quarter; `DATA_REL` recorded (25 for every file so far) |
| Flux | `PDCSAP_FLUX` / `PDCSAP_FLUX_ERR` |
| Time | BKJD. The reader rejects files whose `BJDREFI+BJDREFF ≠ 2454833` |
| Download | `astroquery.mast.Observations.download_file`, written to `.part` then renamed |
| Location | `data/raw/lightcurves/kepler/kic_XXXXXXXXX/<productFilename>` + `products.json` (policy, URI, size, SHA-256) |
| Cache | No network access if `products.json` matches the policy and all files exist |
| Limits | per-config `max_total_download_mb`, a 25-target cap in `fetch_lightcurves.py`, and a 300-star cap in `build_dataset.py` (both overridable with `--allow-large`) |

## 6. Preprocessing and detrending

Per star, on quarterly FITS (raw files never modified):

1. Drop non-finite time/flux/error and cadences with `SAP_QUALITY & 1130799` (the
   Lightkurve "default" bitmask, stored explicitly).
2. Divide each quarter by its median flux, using only this star's data.
3. Detrend: see below.
4. Clip only *upward* outliers (> 5 robust σ) on the detrended flux, so dips are never clipped.

**ML-dataset detrending: masked robust B-spline** (`preprocessing/detrend.py::robust_spline_trend`):

- Least-squares cubic spline per contiguous segment (never across gaps > 0.5 d), with
  knots every **0.3 d**. A knot is kept only if ≥ 10 fit samples lie in its interval
  *and* within ±0.15 d of it, so no knot falls inside a data hole.
- **Every catalogued KOI on the star** (any disposition) is masked from the fit within
  ±1 catalog duration of each predicted mid-transit (`mask_duration_factor: 2`). The trend
  is still evaluated there, so transits are divided by an interpolated baseline.
- 3 fits in total. Between fits, samples deviating more than 3 robust σ from the *local
  running median of the residuals* are dropped. Using the local median keeps smooth
  misfit next to masked windows from being clipped, which otherwise ran away.
- Masking uses only the catalog ephemeris (period, epoch, duration), which is available
  for both classes, so it adds no label information.

**Why not the Milestone 1 running median.** Any location estimator (median, biweight)
shifts upward inside a transit when the star's flux changes across the window by much more
than the noise. Removing in-transit points from the middle of the window's value
distribution biases the median. A spline follows slope and curvature instead.

**Acceptance criterion.** Over 10 noise realisations of each synthetic case
(`evaluation/detrending.py`: trapezoid transits × quasi-periodic spot trends × white
noise, Kepler cadence, two segments):

- *systematic* relative depth error (mean) ≤ 2%, and
- *random* error (RMS over realisations) ≤ 5%.

Rationale: the median positive KOI has model SNR 28.6, i.e. a statistical depth
uncertainty of ≈ 1/28.6 = 3.5%. A 2% systematic error is ≈ 0.6σ for that KOI and 10×
smaller than the 20% steps of the planned depth-attenuation sweep. The 5% RMS tolerance
was set **after** seeing the long-transit case (±4% scatter from interpolating a 1-day
mask on a 1%-amplitude, 5-day-rotation star). It is disclosed as such, not presented as
chosen in advance. Depth error is measured against the same noise realisation divided by
the true trend, so noise cancels.

**Results** (`artifacts/reports/detrending/`, figure
`artifacts/figures/detrending_synthetic_cases.png`): mean relative depth error.

| Case (SYNTHETIC) | M1 running median, blind | Masked robust spline |
|---|---:|---:|
| quiet | −1.1% | +0.1% |
| low variability (500 ppm, 15 d) | −1.4% | +0.1% |
| moderate (0.3%, 8 d) | −22.4% | +0.1% |
| M1 regression (1%, 20 d) | **−29.8%** | +0.1% |
| strong (1%, 5 d) | −36.0% | +0.5% |
| strong, 12-h transit | −76.2% | +0.8% |
| strong, gaps adjacent to transits | +19.0% | +1.7% |
| extreme (1%, 2 d), outside acceptance | +160% | +45% |

The masked spline passes on all acceptance cases. It fails in the extreme 2-day-rotator
regime, where no ≥ 0.3-d-knot spline can follow the variability. Costs: the trend
estimate adds ≈ 27 ppm RMS on 100 ppm white noise (≈ +4% in quadrature). Without masking
(the blind mode), 0.3-d knots absorb long transits almost completely. **The BLS pipeline
therefore keeps the blind running median** (`configs/experiments/bls_smoke.yaml`), and
the ML dataset uses the masked spline.

**Real stars** (138 KOIs on 106 downloaded stars; `real_comparison.csv`). There is no
ground truth, so the catalog fitted depth is used as context. Depth is measured as the
mean over the central 50% of the transit, which sits slightly below a U-shaped transit's
fitted depth at minimum.

| | M1 running median (blind) | masked robust spline |
|---|---:|---:|
| median depth / catalog depth | 0.897 | 0.966 |
| median abs log(depth / catalog) | 0.118 | 0.065 |
| KOIs where this method gives the deeper transit | 17% | 83% |

Examples: K01439.01 (confirmed, 24-h transit) 362 → 2,014 ppm (catalog 2,013);
K03878.02 50 → 119 (117); K01788.02 511 → 957 (946).

**Validity limits (found on real data, then mapped on synthetic data).** K05802.01 (FP,
86-h catalog duration) came out at 512,466 ppm, because a 7-day masked hole cannot be
bridged. A synthetic scan (P = 35 d, 3 seeds) gives:

| Mask window (2 × duration) | moderate variability (0.3%, 8 d) | strong (1%, 5 d) |
|---|---:|---:|
| 1 d (12 h transit) | +0.4% | +139% |
| 2 d (24 h) | −2.9% | +612% |
| 3 d (36 h) | +1,107% | +312% |
| 7.2 d (86 h) | catastrophic | catastrophic |

The masked fraction of the light curve alone is not the problem: up to 67% masked still
gives < 1% error at moderate variability. Strong variability fails in the scan when there
are **few transits** (P = 35 d gives ~5 transits in the scan) or when the **orbit is commensurate
with rotation** (P = P_rot or P_rot/2 gave +24–28%). Per-transit interpolation errors only
average out over many incommensurate transits. That is the most plausible explanation for
this pattern, but it hasn't been isolated experimentally. Close eclipsing binaries are often tidally
synchronised, so this regime is physically relevant.

Rules adopted:

- `max_mask_window_days: 2.0`: examples whose mask window exceeds 2 d (catalog duration
  > 24 h) get `example_status = detrend_mask_too_long`, stay in the table, and are
  excluded from training. In the population this affects 127 train-eligible KOIs (1.9%,
  98% FPs), which removes a mostly-FP subpopulation from training. In the small dataset
  it removed one confirmed planet (K01439.01, 24.11 h), whose depth was actually recovered
  correctly. The rule is kept as-is rather than tuned for that case.
- The remaining risk (strong variability × few transits, or synchronised orbits) is not
  excluded. It is exposed through per-example `star_trend_rms_ppm`, `n_transits_observed`
  and `star_mask_fraction` for stratified evaluation.

Output per star (ML dataset): `data/processed/datasets/<name>/lightcurves/kic_*.parquet`
(`time_bkjd`, `quarter`, `flux_norm`, `flux_err_norm`, `trend`, `flux_detrended`,
`flux_err_detrended`, `in_known_transit_mask`) plus a JSON record with the config, input
SHA-256s and per-quarter counts.

## 7. Fixed-length representation

Per KOI, from the star's detrended flux (`preprocessing/views.py`):

| View | Bins | Coverage | Bin statistic |
|---|---|---|---|
| **global** | 2048 non-overlapping bins over phase [−0.5, 0.5), transit at 0 | whole orbit (secondary eclipses, out-of-transit shape) | median of `flux − 1`, plus count |
| **local** | 201 centres over ±2 transit durations; each bin 0.16 durations wide (overlapping) | transit shape at a resolution independent of period | median of `flux − 1`, plus count |

Why 2048 global bins: over the 6,639 train-eligible KOIs, the number of bins across the
transit (N·D/P) has quantiles 5/25/50/75/95% = 3.9 / 17.8 / 44 / 124 / 383 at N = 2048,
versus 2.0 / 8.9 / 22 at N = 1024. At 2048, 6.7% of KOIs still get < 5 bins across the
transit, and doubling to 4096 only reduces that to 2.5% while halving the samples per bin
(~25 → ~12 for a 17-quarter star). The long-period tail is served better by the local
view than by more global bins. Memory is not a constraint: 2 × 2048 float32 = 16 KB per
example.

**The default model input is the global view only**, as specified for the first CNN,
with shape **`[2, 2048]`**: channel 0 = normalised flux, channel 1 = observation mask. The
local view (`[2, 201]`) is stored and loadable (`views=("global", "local")`), but it is
not the default input.

Per-example metadata (`examples.parquet`): `kepid`, `kepoi_name`, `split`, `label`,
`label_name`, `training_status`, `exclusion_reason`, `label_policy`, `koi_disposition`,
`koi_pdisposition`, `koi_score`, FP flags, period, epoch, duration, depth, `koi_ror`,
impact, model SNR, number of transits, planet radius, Teq, stellar Teff/logg/radius,
Kepler magnitude, `koi_count`, `n_kois_on_star`, `n_points`, `n_quarters`, source files,
coverage and contamination diagnostics, `mask_window_days`, `n_transits_observed`,
`star_mask_fraction`, `star_trend_rms_ppm`, and `example_status`. The build config and
provenance are in `dataset.json`.

## 8. Missing bins and example validity

- Stored arrays keep **NaN for empty bins** plus per-bin counts, so nothing irreversible
  happens at build time.
- At load time (`training/dataset.py`), empty bins are filled by **linear interpolation**
  between the nearest observed bins (circular in phase for the global view, edge-held
  for the local view). A second input channel carries the **observation mask** (1 =
  observed, 0 = interpolated), so the model always knows which values are real. This is
  also needed for the Milestone 5 missing-cadence perturbations.
- Normalisation (per example and view, from observed bins only): subtract the median and
  divide by (median − minimum), so the baseline is 0 and the deepest bin is −1. This
  deliberately removes the absolute depth/SNR scale, which strongly depends on class
  (§9). It falls back to a robust σ when there is no dip. `normalization="none"` gives
  raw `flux − 1`.
- `example_status` rules (thresholds in the dataset config), in order: `missing_ephemeris`,
  `insufficient_points` (< 1000 cadences), `detrend_mask_too_long` (mask window > 2 d, §6),
  `insufficient_global_coverage` (< 50% of
  global bins observed), `insufficient_transit_coverage` (< 50% of in-transit global bins
  observed). Failing examples stay in the table and arrays but are skipped by
  `KOIDataset`.

## 9. Known dataset biases (what a model could exploit)

Univariate ROC-AUC of each catalog variable as a planet-vs-FP score over the 6,639
train-eligible DR25 KOIs (`scripts/dataset_report.py`; 0.5 = no information):

| Variable | Median planet | Median FP | KS | AUC |
|---|---:|---:|---:|---:|
| KOIs on the star (`koi_count`) | 2 | 1 | 0.47 | **0.74** |
| Planet radius [R⊕] | 2.15 | 17.1 | 0.52 | 0.30 |
| Impact parameter | 0.41 | 0.73 | 0.32 | 0.31 |
| Period [d] | 11.2 | 3.9 | 0.34 | 0.66 |
| Stellar Teff [K] | 5618 | 5853 | 0.20 | 0.37 |
| Depth [ppm] | 447 | 808 | 0.34 | 0.40 |
| Model SNR | 28.6 | 47.7 | 0.30 | 0.41 |
| Stellar radius [R☉] | 0.96 | 1.01 | 0.17 | 0.42 |
| Duration [h] | 3.47 | 4.03 | 0.12 | 0.44 |
| Kepler magnitude | 14.62 | 14.49 | 0.08 | 0.52 |

Findings:

1. **FPs, not planets, have the higher SNR and deeper transits.** The expected "confirmed
   planets are easier" effect does not show up as higher SNR among positives here: the
   negative class is dominated by deep eclipsing binaries. Depth normalisation of the
   model input (§8) removes the depth scale but not transit *shape* (V vs U) or relative
   noise.
2. **Multiplicity is the strongest single cue (AUC 0.74).** Multi-KOI stars are mostly
   planets ("validation by multiplicity"). The model never receives `koi_count`, but other
   planets' transits leave (mostly incoherent) residue in a KOI's global view (§10), so
   an indirect multiplicity signal exists. Evaluation must report metrics separately for
   single- and multi-KOI stars.
3. **Period differs** (FPs median 3.9 d, planets 11.2 d). In a phase-folded view, period
   shows up through transit width in phase (D/P).
4. **Magnitude is not informative (AUC 0.52).** Brightness-stratified results will not be
   confounded with class.
5. The small dataset is star-balanced by construction and amplifies some differences
   (median SNR 33.8 vs 86.1). Full-scale datasets should use the natural population.

No variable is rebalanced automatically; these are to be controlled in evaluation.

## 10. Multi-KOI stars

Other KOIs on the same star are **not** removed from a KOI's fold (they are only masked
during trend fitting). Evidence from the small dataset (44 examples on 17 multi-KOI stars,
`multi_koi_contamination.png`), measured as the largest expected dip that other KOIs'
transits contribute to any D/2-wide phase bin, relative to the KOI's own depth:

- median 0.08. Different-period siblings fold incoherently.
- Inside a KOI's own transit window the contribution is ≤ 91 ppm (≤ 12% of own depth).
- 5 of 44 examples exceed 0.25:
  - two same-period FP pairs (K03338.01/.02 at P = 10.9938 d): an eclipsing binary's
    primary/secondary catalogued separately. The coherent dip at phase 0.5 is genuine FP
    evidence and must be kept.
  - K01788.02 (P = 369 d, only ~2 cycles in 8 quarters): a sibling with 6× deeper
    transits leaves 3.1× own-depth dips elsewhere in its global view, because too few
    cycles exist to average them out.
  - K00896.03 (a candidate, excluded) and K01422.04 (0.28).

Each example stores `contam_max_bin_fraction`, `contam_max_expected_ppm`,
`contam_in_transit_expected_ppm` and `contam_max_relative_to_depth`. Proposed later
strategy, if evaluation shows it matters: when folding KOI A, drop samples inside other
KOIs' transit windows *unless* the other KOI has the same period (±0.1%) or a small
integer ratio, since those carry eclipsing-binary information. Also report metrics for
`contam_max_relative_to_depth > 0.25` separately.

## 11. Dataset storage and scales

`data/processed/datasets/<name>/` (git-ignored): `examples.parquet`,
`global_flux.npy` (float32 [N, 2048], NaN = empty), `global_count.npy` (int32),
`local_flux.npy`, `local_count.npy`, `lightcurves/`, `build_log.csv`, `dataset.json`
(config, label policy, catalog SHA-256, split-metadata SHA-256, counts, per-file
SHA-256). `.npy` files are memory-mapped at load, so they're never fully loaded into RAM
or GPU. There are no new dependencies. The full scale (6,883 KOIs on 5,805 stars) would be ≈ 124 MB of arrays.
The stored per-star detrended light curves dominate disk use: ~3.3 MB per star (336 MB of
the small dataset's 339 MB), so ~19 GB at full scale. If that becomes a problem, they can
be written as float32 or reduced to the columns needed for perturbation experiments.

| Scale | Config | Stars | Status |
|---|---|---|---|
| smoke | `kepler_dr25_smoke.yaml` | 3 Milestone 1 anchors + 6 sampled | built |
| small | `kepler_dr25_small.yaml` | 3 anchors + 100 sampled (50 with a planet / 50 FP-only, apportioned 70/15/15 by split) | built |
| full | `kepler_dr25_full.yaml` | every star with ≥ 1 train-eligible KOI (5,805 stars, 6,883 KOIs; ~40 GB of light curves) | defined, not built |

## 12. BLS run output

`scripts/run_bls.py` → `artifacts/experiments/<UTC timestamp>_<name>/`: `config.yaml`,
`environment.json`, `results.json`, `summary.csv`, `periodograms/*.json`, `figures/`
(unchanged from Milestone 1; BLS still uses the blind running-median detrending).
