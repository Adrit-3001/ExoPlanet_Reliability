"""Validation report for a built KOI dataset: counts, leakage, data quality, bias, figures.

Bias statistics are computed twice: on the whole train-eligible DR25 population (what a
full-scale model would see) and on the built dataset. For each catalog variable the report
gives class medians, the two-sample KS statistic and the univariate ROC-AUC of that single
variable as a planet-vs-FP score (0.5 = no information, 1.0 or 0.0 = perfectly separating).

Outputs: artifacts/reports/<dataset>/report.json, bias_population.csv, bias_dataset.csv and
figures (class_balance.png, distributions.png, example_views.png, multi_koi_contamination.png).

Example:
    python scripts/dataset_report.py --dataset kepler_dr25_small
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

from exoreliability.config import get_paths
from exoreliability.data.archive import DR25_KOI_TABLE
from exoreliability.data.cache import catalog_snapshot_path, kic_dirname, write_json_atomic
from exoreliability.data.catalog import ephemeris_from_row
from exoreliability.data.examples import STATUS_OK, load_dataset
from exoreliability.data.labels import apply_label_policy
from exoreliability.data.splits import SPLITS, check_no_leakage, load_splits
from exoreliability.preprocessing.phase_fold import in_transit_mask, phase_fold

BIAS_VARIABLES = {
    "koi_period": ("Orbital period [d]", True),
    "koi_depth": ("Transit depth [ppm]", True),
    "koi_duration": ("Transit duration [h]", True),
    "koi_model_snr": ("Model SNR", True),
    "koi_kepmag": ("Kepler magnitude", False),
    "koi_prad": ("Planet radius [R_earth]", True),
    "koi_impact": ("Impact parameter", False),
    "koi_steff": ("Stellar Teff [K]", False),
    "koi_srad": ("Stellar radius [R_sun]", True),
    "koi_count": ("KOIs on the star", False),
}
POS_COLOR, NEG_COLOR = "#2563eb", "#dc2626"


def bias_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, (label, _) in BIAS_VARIABLES.items():
        pos = df.loc[df.label == 1, var].dropna().to_numpy(float)
        neg = df.loc[df.label == 0, var].dropna().to_numpy(float)
        if len(pos) < 3 or len(neg) < 3:
            continue
        auc = mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))
        rows.append(
            {
                "variable": var,
                "description": label,
                "n_pos": len(pos),
                "n_neg": len(neg),
                "missing_pos": int(df.loc[df.label == 1, var].isna().sum()),
                "missing_neg": int(df.loc[df.label == 0, var].isna().sum()),
                "median_pos": float(np.median(pos)),
                "median_neg": float(np.median(neg)),
                "p10_pos": float(np.percentile(pos, 10)),
                "p90_pos": float(np.percentile(pos, 90)),
                "p10_neg": float(np.percentile(neg, 10)),
                "p90_neg": float(np.percentile(neg, 90)),
                "ks_statistic": float(ks_2samp(pos, neg).statistic),
                "univariate_auc": float(auc),
                "separability": float(max(auc, 1 - auc)),
            }
        )
    if not rows:  # too few examples per class (e.g. smoke scale): empty table, same schema
        return pd.DataFrame(
            columns=[
                "variable",
                "median_pos",
                "median_neg",
                "ks_statistic",
                "univariate_auc",
                "separability",
            ]
        )
    return pd.DataFrame(rows).sort_values("separability", ascending=False)


def fig_class_balance(ex: pd.DataFrame, out: Path) -> None:
    t = ex[(ex.example_status == STATUS_OK) & (ex.training_status == "train_eligible")]
    counts = t.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(list(SPLITS))
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    x = np.arange(len(SPLITS))
    ax.bar(x - 0.2, counts.get(1, 0), 0.4, color=POS_COLOR, label="planet (1)")
    ax.bar(x + 0.2, counts.get(0, 0), 0.4, color=NEG_COLOR, label="false positive (0)")
    for i, s in enumerate(SPLITS):
        ax.text(
            i,
            max(counts.loc[s]) + 0.5,
            f"{t[t.split == s].kepid.nunique()} stars",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(x, SPLITS)
    ax.set_ylabel("trainable KOI examples")
    ax.legend(fontsize=8)
    ax.set_title("Class balance by split (KIC-grouped)")
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_distributions(pop: pd.DataFrame, ds: pd.DataFrame, out: Path) -> None:
    vars_ = ["koi_period", "koi_depth", "koi_duration", "koi_model_snr", "koi_kepmag", "koi_count"]
    fig, axes = plt.subplots(2, len(vars_), figsize=(4 * len(vars_), 6.5), constrained_layout=True)
    for row, (name, df) in enumerate(
        (("DR25 train-eligible population", pop), ("built dataset", ds))
    ):
        for ax, var in zip(axes[row], vars_, strict=True):
            label, log = BIAS_VARIABLES[var]
            vals = df[var].dropna()
            if var == "koi_count":
                bins = np.arange(0.5, 8.5, 1.0)
            elif log:
                v = vals[vals > 0]
                bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 40)
            else:
                bins = np.linspace(vals.min(), vals.max(), 40)
            for lab, color, name_ in ((1, POS_COLOR, "planet"), (0, NEG_COLOR, "false positive")):
                v = df.loc[df.label == lab, var].dropna()
                # Fraction per bin, not density: with log-spaced bins, density per linear
                # unit inflates the narrow low-value bins and misrepresents the classes.
                ax.hist(
                    v,
                    bins=bins,
                    histtype="step",
                    weights=np.full(len(v), 1 / max(len(v), 1)),
                    color=color,
                    lw=1.5,
                    label=f"{name_} (n={len(v)})",
                )
            if log and var != "koi_count":
                ax.set_xscale("log")
            ax.set_xlabel(label)
            ax.set_title(name, fontsize=8, loc="left")
            ax.legend(fontsize=6)
    fig.suptitle(
        "Catalog-variable distributions by class (fraction of each class per bin; log-spaced bins where log axis)"
    )
    fig.savefig(out, dpi=100)
    plt.close(fig)


def _view_panel(ax_g, ax_l, d, i: int, title: str) -> None:
    g, gc = d.global_flux[i], d.global_count[i]
    loc, lc = d.local_flux[i], d.local_count[i]
    xg = np.linspace(-0.5, 0.5, g.size, endpoint=False) + 0.5 / g.size
    ax_g.plot(xg[gc > 0], g[gc > 0] * 1e6, ".", ms=1.5, color="0.2")
    ax_g.set_title(title, fontsize=7, loc="left")
    ax_g.set_xlabel("phase", fontsize=7)
    xl = np.linspace(-2, 2, loc.size)
    ax_l.plot(xl[lc > 0], loc[lc > 0] * 1e6, "-", color="tab:blue", lw=1)
    ax_l.set_xlabel("transit durations", fontsize=7)
    for ax in (ax_g, ax_l):
        ax.tick_params(labelsize=6)
    ax_g.set_ylabel("Δflux [ppm]", fontsize=7)


def fig_examples(d, out: Path) -> None:
    ex = d.examples
    ok = ex[(ex.example_status == STATUS_OK) & (ex.training_status == "train_eligible")]
    multi_star = ok[ok.n_kois_on_star >= 3].kepid.min() if (ok.n_kois_on_star >= 3).any() else None
    picks = list(ok[ok.label == 1].sort_values("kepoi_name").index[:3]) + list(
        ok[ok.label == 0].sort_values("kepoi_name").index[:3]
    )
    if multi_star is not None:
        picks += list(ok[ok.kepid == multi_star].sort_values("kepoi_name").index[:3])
    fig, axes = plt.subplots(
        len(picks),
        2,
        figsize=(12, 1.9 * len(picks)),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [3, 1]},
    )
    for r, i in enumerate(picks):
        row = ex.loc[i]
        title = (
            f"{row.kepoi_name} KIC {row.kepid} [{row.split}] {row.label_name} ({row.koi_disposition}/{row.koi_pdisposition}) "
            f"P={row.koi_period:.3f} d D={row.koi_duration:.1f} h depth={row.koi_depth:.0f} ppm, KOIs on star={row.n_kois_on_star}"
        )
        _view_panel(axes[r, 0], axes[r, 1], d, int(row.example_index), title)
    fig.suptitle(
        "Global (left) and local (right) views of real Kepler KOIs (bottom rows: one multi-KOI star)"
    )
    fig.savefig(out, dpi=100)
    plt.close(fig)


def fig_contamination(d, root: Path, out: Path) -> list[str]:
    ex = d.examples
    cand = ex[ex.n_kois_on_star > 1].sort_values("contam_max_relative_to_depth", ascending=False)
    stars = list(dict.fromkeys(cand.kepid))[:4]
    rows = [r for _, r in cand.iterrows() if r.kepid in stars]
    rows = sorted(rows, key=lambda r: (stars.index(r.kepid), r.kepoi_name))[:8]
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(12, 1.9 * len(rows)), constrained_layout=True, squeeze=False
    )
    for ax, r in zip(axes[:, 0], rows, strict=True):
        lc = pd.read_parquet(root / "lightcurves" / f"{kic_dirname(int(r.kepid))}.parquet")
        t, f = lc.time_bkjd.to_numpy(), lc.flux_detrended.to_numpy()
        eph = ephemeris_from_row(r)
        siblings = ex[(ex.kepid == r.kepid) & (ex.kepoi_name != r.kepoi_name)]
        other = np.zeros(t.size, bool)
        for _, s in siblings.iterrows():
            e2 = ephemeris_from_row(s)
            if e2 is not None and e2.duration_hours:
                other |= in_transit_mask(t, e2, 1.0)
        ph = phase_fold(t, eph) / eph.period_days
        ax.plot(ph[~other], (f[~other] - 1) * 1e6, ",", color="0.55", rasterized=True)
        ax.plot(
            ph[other],
            (f[other] - 1) * 1e6,
            ".",
            ms=1.5,
            color="tab:red",
            rasterized=True,
            label="samples in another KOI's transit",
        )
        lo = np.nanpercentile((f - 1) * 1e6, 0.2)
        ax.set_ylim(
            min(lo, -1.5 * r.koi_depth if np.isfinite(r.koi_depth) else lo),
            np.nanpercentile((f - 1) * 1e6, 99.9),
        )
        ax.set_title(
            f"{r.kepoi_name} ({r.koi_disposition}) folded on its own P={r.koi_period:.4f} d; siblings: "
            + ", ".join(f"{s.kepoi_name} P={s.koi_period:.4f}" for _, s in siblings.iterrows())
            + f" | max other-signal dip = {r.contam_max_relative_to_depth:.2f} x own depth",
            fontsize=7,
            loc="left",
        )
        ax.set_ylabel("Δflux [ppm]", fontsize=7)
        ax.legend(fontsize=6, loc="lower left")
    axes[-1, 0].set_xlabel("phase")
    fig.suptitle(
        "Multi-KOI stars: do other KOIs' transits fold coherently into this KOI's view? (real Kepler data)"
    )
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return [str(r.kepoi_name) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="kepler_dr25_small")
    args = parser.parse_args()
    paths = get_paths()
    root = paths.datasets / args.dataset
    d = load_dataset(root, verify=True)
    ex = d.examples
    out = paths.reports / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    catalog = apply_label_policy(pd.read_csv(catalog_snapshot_path(paths, DR25_KOI_TABLE)))
    pop = catalog[catalog.training_status == "train_eligible"]
    splits = load_splits(paths.splits)
    ok = ex[ex.example_status == STATUS_OK]
    trainable = ok[ok.training_status == "train_eligible"]

    # Leakage: within the dataset, and consistency with the persisted split manifests.
    leakage = check_no_leakage(ex.rename(columns={}))
    manifest_split = splits.manifest.set_index("kepoi_name").split
    mismatched = int((ex.set_index("kepoi_name").split != manifest_split.loc[ex.kepoi_name]).sum())
    leakage["examples_with_split_differing_from_manifest"] = mismatched
    if mismatched:
        raise SystemExit(f"{mismatched} examples disagree with the split manifest")

    log = pd.read_csv(root / "build_log.csv")
    quality = {
        "stars_attempted": len(log),
        "stars_failed_download_or_missing": int((log.status == "no_lightcurve").sum()),
        "stars_failed_preprocessing": int((log.status == "preprocessing_failed").sum()),
        "example_status": {k: int(v) for k, v in ex.example_status.value_counts().items()},
        "examples_excluded_by_label_policy": {
            k: int(v)
            for k, v in ex.loc[ex.training_status == "excluded", "exclusion_reason"]
            .value_counts()
            .items()
        },
        "examples_near_coverage_threshold": ex.loc[
            (ex.global_coverage < 0.6) | (ex.transit_coverage < 0.6),
            ["kepoi_name", "global_coverage", "transit_coverage"],
        ]
        .round(3)
        .to_dict("records"),
        "population_exclusions": {
            k: int(v)
            for k, v in catalog.loc[catalog.training_status == "excluded", "exclusion_reason"]
            .value_counts()
            .items()
        },
        "population_labelled_missing_depth": int(pop.koi_depth.isna().sum()),
        "population_labelled_missing_ephemeris": int(
            pop[["koi_period", "koi_time0bk", "koi_duration"]].isna().any(axis=1).sum()
        ),
    }

    counts = {
        "n_examples": len(ex),
        "n_unique_kics": int(ex.kepid.nunique()),
        "n_trainable": len(trainable),
        "n_positive": int((trainable.label == 1).sum()),
        "n_negative": int((trainable.label == 0).sum()),
        "n_multi_koi_stars": int((ex.groupby("kepid").size() > 1).sum()),
        "n_examples_on_multi_koi_stars": int((ex.n_kois_on_star > 1).sum()),
        "by_split": {
            s: {
                "n_trainable": int((trainable.split == s).sum()),
                "n_positive": int(((trainable.split == s) & (trainable.label == 1)).sum()),
                "n_negative": int(((trainable.split == s) & (trainable.label == 0)).sum()),
                "n_kics": int(ex[ex.split == s].kepid.nunique()),
            }
            for s in SPLITS
        },
        "kois_per_kic": {
            int(k): int(v)
            for k, v in ex.groupby("kepid").size().value_counts().sort_index().items()
        },
    }

    bias_pop = bias_table(pop)
    bias_ds = bias_table(trainable)
    bias_pop.to_csv(out / "bias_population.csv", index=False)
    bias_ds.to_csv(out / "bias_dataset.csv", index=False)
    fp_flags = (
        pop[pop.label == 0][["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]]
        .mean()
        .round(4)
        .to_dict()
    )

    multi = ex[ex.n_kois_on_star > 1]
    contamination = {
        "n_examples_on_multi_koi_stars": len(multi),
        "median_max_other_signal_relative_to_depth": float(
            multi.contam_max_relative_to_depth.median()
        ),
        "n_over_0.25_of_own_depth": int((multi.contam_max_relative_to_depth > 0.25).sum()),
        "examples_over_0.25": multi.loc[
            multi.contam_max_relative_to_depth > 0.25,
            [
                "kepoi_name",
                "koi_period",
                "contam_max_relative_to_depth",
                "contam_in_transit_expected_ppm",
            ],
        ]
        .round(3)
        .to_dict("records"),
        "max_in_transit_expected_ppm": float(multi.contam_in_transit_expected_ppm.max()),
    }

    figs = {
        "class_balance": out / "class_balance.png",
        "distributions": out / "distributions.png",
        "example_views": out / "example_views.png",
        "multi_koi_contamination": out / "multi_koi_contamination.png",
    }
    fig_class_balance(ex, figs["class_balance"])
    fig_distributions(pop, trainable, figs["distributions"])
    fig_examples(d, figs["example_views"])
    contam_shown = fig_contamination(d, root, figs["multi_koi_contamination"])

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "dataset_created_at_utc": d.info["created_at_utc"],
        "shapes": d.info["shapes"],
        "counts": counts,
        "leakage_checks": leakage,
        "data_quality": quality,
        "bias": {
            "population_train_eligible": bias_pop.round(4).to_dict("records"),
            "dataset_trainable": bias_ds.round(4).to_dict("records"),
            "negative_fp_flag_fractions_population": fp_flags,
        },
        "multi_koi_contamination": {**contamination, "figure_kois": contam_shown},
        "figures": {k: str(v.relative_to(paths.root)) for k, v in figs.items()},
    }
    write_json_atomic(out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("counts", "leakage_checks")}, indent=1))
    print("\nBias (population, train-eligible DR25):")
    print(
        bias_pop[["variable", "median_pos", "median_neg", "ks_statistic", "univariate_auc"]]
        .round(3)
        .to_string(index=False)
    )
    print("\nBias (built dataset):")
    print(
        bias_ds[["variable", "median_pos", "median_neg", "ks_statistic", "univariate_auc"]]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nContamination: {json.dumps(contamination, indent=1)}")
    print(f"\nWrote {out.relative_to(paths.root)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
