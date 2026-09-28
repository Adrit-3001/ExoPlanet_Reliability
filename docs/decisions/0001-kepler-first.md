# ADR 0001 — Kepler Q1–Q17 DR25 first; TESS deferred

- **Status:** accepted
- **Date:** 2026-09-28

## Context

The project studies when an ML transit detector's output can be trusted, and how that trust
degrades under controlled observational perturbations. These results are only interpretable if:

1. labels come from one documented, internally consistent vetting process,
2. light curves come from one pipeline version with known systematics handling,
3. the dataset is small enough to iterate on one laptop (≈8 GB VRAM) but large enough to
   later support stratified reliability analysis.

## Decision

Use the **Kepler Q1–Q17 DR25** KOI table (`q1_q17_dr25_koi`, NASA Exoplanet Archive) for
catalog metadata. Use Kepler SOC **long-cadence PDCSAP light curves** from MAST
(`author=Kepler`, `exptime=1800`) for photometry.

Reasons:

- **Uniform vetting.** DR25 is the final, fully automated Robovetter run over the whole
  4-year mission. `koi_pdisposition` was produced by one algorithm with a published
  completeness/reliability characterisation. It is not a mix of catalog generations like
  the cumulative KOI table.
- **Matched photometry.** MAST Kepler light curves carry `DATA_REL = 25` (verified in every
  file downloaded so far), so catalog ephemerides and photometry share a processing release
  and a time system (BKJD).
- **Long baselines.** ~4 years, 17 quarters, 29.4-min cadence. This supports multi-transit
  folding and controlled cadence-removal experiments.
- **Mature labels.** Years of follow-up mean many KOIs have archive dispositions of
  CONFIRMED or FALSE POSITIVE, which the conservative label rule needs
  (see `docs/data_contract.md`).
- **Manageable scale.** 8,054 KOIs on 6,923 stars; ~0.4–0.5 MB per quarter file. Small
  subsets download in seconds.

## Why TESS is deferred

- TESS sectors are ~27 days, so most targets have short baselines and few transits. That
  changes the detection problem, not just the data source.
- TESS dispositions (TOI catalog) are heterogeneous and still evolving. Mixing them with
  Kepler labels would confound "survey difference" with "label-process difference".
- TESS has larger pixels (21″ vs 4″) and more blending, so the false-positive population differs.
- Cross-survey generalisation is a research question in its own right (Milestone 7). It
  needs a stable Kepler baseline to compare against.

## Consequences

- All Phase-1 code assumes Kepler conventions (BKJD, quarters, `SAP_QUALITY` bitmask). The
  config schema restricts `mission`/`author`/`exptime` to Kepler values, so a different
  survey cannot be substituted silently.
- Positive labels depend on external confirmation and are biased toward high-SNR signals.
  This is documented and must be considered in reliability analyses.
- Short-cadence data, K2 and target-pixel files are out of scope until explicitly added.
