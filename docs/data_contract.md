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

`koi_pdisposition` is the homogeneous DR25 Robovetter outcome based on Kepler data alone.
`koi_disposition` additionally reflects the archive's confirmation status from follow-up
and the literature. The archive can update that status over time, which is why the
retrieval timestamp and checksum are recorded.

## 2. Label mapping — rule `dr25_conservative_v1`

Implemented in `exoreliability.data.catalog.assign_labels`.

| Derived label | Condition | Count |
|---|---|---:|
| `planet` (1) | `koi_disposition == CONFIRMED` **and** `koi_pdisposition == CANDIDATE` | 2,675 |
| `false_positive` (0) | `koi_disposition == FALSE POSITIVE` **and** `koi_pdisposition == FALSE POSITIVE` | 3,964 |
| excluded: `unresolved_candidate` | both CANDIDATE | 1,358 |
| excluded: `disposition_conflict` | the two fields disagree (CONFIRMED/FP: 56, FP/CANDIDATE: 1) | 57 |
| excluded: `missing_disposition` | either field null | 0 |

Output columns added next to the **unchanged** original columns: `label` (nullable Int8),
`label_name`, `label_rule`, `exclusion_reason`.

**Known caveats (to be revisited before supervised training):**

- "Confirmed" status depends on follow-up. It favours deeper, higher-SNR signals around
  brighter stars. Positives are therefore not a random sample of real planets, and the
  label distribution is confounded with SNR, depth and magnitude. Reliability analyses
  stratified by those variables need to account for this.
- DR25 false positives include instrumental and non-transit-like signals
  (`koi_fpflag_nt`) as well as astrophysical false positives (eclipsing binaries,
  contamination). The FP flags are kept in the snapshot so these can be separated later.
- Excluding unresolved candidates makes the supervised task easier than the real vetting
  problem. Any later experiment that includes candidates must use a new rule name.

## 3. Target identifiers

- A **target** is a star, identified by its KIC ID (`kepid`). One star can host several KOIs.
- On-disk directory names use `kic_<9-digit zero-padded kepid>`, matching MAST `kplr` names.
- The API accepts `10811496`, `KIC 10811496` or `kic_010811496`.
- Future train/val/test splits must group by `kepid` (Milestone 2). No splits exist yet.

## 4. Subset manifests

`scripts/fetch_catalog.py` writes `data/interim/manifests/<name>_kois.csv` plus `.json`:

- Rows are sorted by `kepoi_name` first, so the result doesn't depend on the archive's row order.
- With `stratify_by_label: true`, the limit is split as evenly as possible between labels,
  and each label group is sampled with `pandas.DataFrame.sample(random_state=seed)`.
- The JSON records the selection config, label counts, catalog query, snapshot checksum and
  the manifest's own checksum.

The default `kepler_dr25_small` manifest (seed 42, 20 KOIs) has 10 planets and
10 false positives on 20 distinct stars.

## 5. Light-curve products

| Item | Policy |
|---|---|
| Search | `lightkurve.search_lightcurve("KIC <id>", mission="Kepler", author="Kepler", exptime=1800)` |
| Product | Kepler SOC long-cadence `*_llc.fits`, one file per quarter |
| Release check | FITS header `DATA_REL` recorded per file (observed: 25 for every file downloaded so far) |
| Flux | `PDCSAP_FLUX` / `PDCSAP_FLUX_ERR` (configurable to `SAP_FLUX`) |
| Time | `TIME` column, BKJD. The reader rejects files whose `BJDREFI+BJDREFF ≠ 2454833` |
| Quality | `SAP_QUALITY` kept raw. Masking happens in preprocessing |
| Download | `astroquery.mast.Observations.download_file` to a `.part` file, then renamed into place |
| Location | `data/raw/lightcurves/kepler/kic_XXXXXXXXX/<productFilename>` |
| Manifest | `products.json` per target: policy, search info, per-file quarter, URI, size, SHA-256 |
| Cache rule | If `products.json` exists, the policy is unchanged and every file is present, no network access occurs |
| Limits | `max_targets`, `max_products_per_target`, `max_total_download_mb`, plus a hard cap of 25 targets unless `--allow-large` |

Search results are filtered to rows whose `target_name` is exactly `kplr<kepid>`, so cone-search
neighbours are excluded. A per-target summary goes to `data/interim/manifests/<name>_lightcurves.csv`.

## 6. Preprocessing output

`scripts/preprocess_lightcurves.py` → `data/processed/lightcurves/kic_XXXXXXXXX/`:

`lightcurve.parquet` columns:

| Column | Meaning |
|---|---|
| `time_bkjd` | BKJD, sorted ascending |
| `quarter` | Kepler quarter |
| `flux_norm`, `flux_err_norm` | PDCSAP flux / per-quarter median |
| `trend` | running-median trend (1 if detrending disabled) |
| `flux_detrended`, `flux_err_detrended` | `flux_norm / trend`, `flux_err_norm / trend` |

`preprocessing.json` records: the full preprocessing config, package versions, input file
names and SHA-256, per-quarter FITS metadata (DATA_REL, pipeline version, CROWDSAP, FLFRCSAP,
PDC method), per-quarter removal counts, and whether transits were masked during detrending.

Normalization statistics are per target and per quarter. Nothing is shared across targets,
so no information can leak between future splits.

## 7. BLS run output

`scripts/run_bls.py` → `artifacts/experiments/<UTC timestamp>_<name>/`:
`config.yaml`, `environment.json` (git commit, dirty flag, Python/package versions,
torch/CUDA if installed, catalog snapshot metadata), `results.json` (per target: catalog KOIs,
per-severity BLS stats, perturbation parameters and RNG seed entropy, catalog period
comparison), `summary.csv`, `periodograms/*.json` (max-pooled to ≤ 2,000 points), `figures/`.
