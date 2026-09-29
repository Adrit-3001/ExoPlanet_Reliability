"""Deterministic choice of stars for a dataset build.

The sampling unit is the star (KIC), never the KOI, so all KOIs of a selected star enter
the dataset together and inherit the star's pre-assigned split.

For sampled (non-full) builds, ``n_stars`` is apportioned to splits by the split
fractions, and within each split half the stars are drawn from stars with at least one
positive KOI (``pos_only``/``mixed``) and half from ``neg_only`` stars. This makes small
datasets roughly class-balanced at the star level. It is a convenience for pipeline
validation, not a model of the natural class ratio; the ``full`` scale uses every
eligible star instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exoreliability.config import StarSelectionConfig
from exoreliability.data.splits import SPLITS, SplitSet, _cut_sizes

POOLS = ("has_positive", "negative_only")


def eligible_stars(splits: SplitSet) -> pd.DataFrame:
    """Stars with at least one train-eligible KOI: kepid, split, star_stratum, pool."""
    m = splits.manifest
    stars = m.drop_duplicates("kepid")[["kepid", "split", "star_stratum"]]
    stars = stars[stars.star_stratum != "unlabelled_only"].copy()
    stars["pool"] = np.where(stars.star_stratum == "neg_only", "negative_only", "has_positive")
    return stars.sort_values("kepid").reset_index(drop=True)


def select_stars(splits: SplitSet, cfg: StarSelectionConfig) -> pd.DataFrame:
    """Return selected stars with columns kepid, split, star_stratum, selection_reason."""
    all_stars = splits.manifest.drop_duplicates("kepid").set_index("kepid")
    unknown = [k for k in cfg.include_kepids if k not in all_stars.index]
    if unknown:
        raise ValueError(f"include_kepids not in the split manifest: {unknown}")

    anchors = pd.DataFrame(
        {
            "kepid": cfg.include_kepids,
            "split": [all_stars.loc[k, "split"] for k in cfg.include_kepids],
            "star_stratum": [all_stars.loc[k, "star_stratum"] for k in cfg.include_kepids],
            "selection_reason": "include_kepids",
        }
    )
    pool = eligible_stars(splits)
    pool = pool[~pool.kepid.isin(cfg.include_kepids)]

    if cfg.all_eligible:
        sampled = pool.assign(selection_reason="all_eligible")
    else:
        fractions = splits.metadata["target_fractions"]
        per_split = _cut_sizes(cfg.n_stars, fractions)
        parts = []
        for si, split in enumerate(SPLITS):
            per_pool = _cut_sizes(per_split[split], dict.fromkeys(POOLS, 0.5))
            for pi, name in enumerate(POOLS):
                cand = pool[(pool.split == split) & (pool.pool == name)].sort_values("kepid")
                n = min(per_pool[name], len(cand))
                idx = np.random.default_rng([cfg.seed, si, pi]).choice(
                    len(cand), size=n, replace=False
                )
                parts.append(cand.iloc[np.sort(idx)])
        sampled = pd.concat(parts).assign(selection_reason="sampled") if parts else pool.iloc[:0]

    cols = ["kepid", "split", "star_stratum", "selection_reason"]
    out = pd.concat([anchors[cols], sampled[cols]], ignore_index=True)
    return out.drop_duplicates("kepid").sort_values("kepid").reset_index(drop=True)
