# ADR 0002 — KOI-level examples, KIC-grouped splits, masked spline detrending

- **Status:** accepted
- **Date:** 2026-09-29

## Context

Milestone 1 showed that (a) one star can host several catalogued signals, and a blind BLS
search finds only the strongest; (b) the running-median detrender absorbs up to ~30% of a
transit's depth on variable stars. A supervised dataset built on either would be
mislabelled at the signal level or biased in depth.

## Decisions

1. **Learning unit = one KOI signal.** Each example is the star's light curve folded on
   one KOI's catalog ephemeris. Stars with N KOIs give N examples.
2. **Splits are grouped by KIC ID and assigned once over the whole DR25 catalog**,
   stratified by the star's label composition, seed 42, 70/15/15. Every dataset scale
   reuses that assignment. Replacing it requires an explicit `--overwrite`.
3. **Transit-preserving detrending for the ML dataset:** a cubic B-spline (0.3-d knots)
   fitted with every catalogued KOI transit masked, knots kept out of data holes, and
   outliers judged against a local residual median. The blind running median stays for
   BLS, where no ephemeris may be used.
4. **Other KOIs are not removed from a KOI's fold.** They are masked only during trend
   fitting, and per-example contamination diagnostics are stored. Measured contamination
   is small except for same-period pairs (genuine eclipsing-binary evidence) and
   few-cycle long-period KOIs.
5. **Representation:** global view (2048 bins) as the default model input, plus a stored
   local view (201 bins over ±2 durations), each with an observation-mask channel.
   Per-example depth normalisation.
6. **Label policy `dr25_clean_v2`:** same mapping as v1, with an explicit vocabulary and
   a hard error on unknown values.

## Alternatives considered

- *Split by KOI row:* rejected. KOIs on the same star share systematics, stellar
  properties and sometimes signals, which would leak across splits.
- *Split only the selected subset:* rejected. Growing the dataset would reshuffle stars
  between splits.
- *Running median with a transit mask:* better than blind (−1.3% on the M1 regression
  case), but still −26% on long transits and +32% with gaps next to transits.
- *Shorter median window:* also absorbs more of long transits. It moves the error rather
  than removing it.
- *Removing sibling KOIs from each fold:* not justified by the measured contamination, and
  it would erase eclipsing-binary secondaries that carry FP information.

## Consequences

- The dataset depends on catalog ephemerides for both masking and folding. KOIs with
  wrong ephemerides or strong TTVs will be degraded, which is to be analysed as a failure
  mode.
- The trend estimate adds ~27 ppm RMS noise for a 100-ppm star.
- BLS and ML inputs use different detrending. Comparisons between them must state this.
