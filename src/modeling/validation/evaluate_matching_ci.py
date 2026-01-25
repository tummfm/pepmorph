#!/usr/bin/env python

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA_PROCESSED, FEATURES, FIGS_DIR, RESULTS_DIR, TEAL, set_paper_style
from utils import parse_list_field


def count_k_used(used_features, include_length: bool = True) -> int:
    used = set(parse_list_field(used_features))
    feats = FEATURES if include_length else [f for f in FEATURES if f != "length"]
    return sum(f in used for f in feats)


def row_success_all_targeted(row: pd.Series, include_length: bool = True) -> bool:
    used = set(parse_list_field(row.get("used_features", [])))
    feats = FEATURES if include_length else [f for f in FEATURES if f != "length"]
    targeted = [f for f in feats if f in used]
    if not targeted:
        return False
    for feat in targeted:
        v = row.get(f"match_{feat}", pd.NA)
        if pd.isna(v) or (bool(v) is False):
            return False
    return True


def bootstrap_ci_mean(values, n_boot: int = 5000, alpha: float = 0.05, rng=None):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(0) if rng is None else rng
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(x.mean()), float(lo), float(hi)


def bootstrap_ci_over_conditions(cond_values_df: pd.DataFrame, value_col: str, n_boot: int = 5000, alpha: float = 0.05):
    x = cond_values_df[value_col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(0)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(x.mean()), float(lo), float(hi)


def load_results_pickle(pkl_path: str):
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    if "samples_df" not in results:
        raise KeyError("results.pkl must contain 'samples_df'")
    samples_df = results["samples_df"].copy()
    within_df = results.get("within_df", None)
    cond_meta_df = results.get("cond_meta_df", None)
    return samples_df, within_df, cond_meta_df


def build_condition_summary(samples_df: pd.DataFrame, include_length_in_k: bool = True) -> pd.DataFrame:
    df = samples_df.copy()
    df["k"] = df["used_features"].apply(lambda u: count_k_used(u, include_length=include_length_in_k))
    df["success_all"] = df.apply(lambda r: row_success_all_targeted(r, include_length=include_length_in_k), axis=1)

    rows = []
    for cid, g in df.groupby("cond_idx", sort=True):
        used0 = g["used_features"].iloc[0]
        k = int(count_k_used(used0, include_length=include_length_in_k))

        out = {
            "cond_idx": int(cid),
            "k": k,
            "ood_type": g["ood_type"].iloc[0] if "ood_type" in g.columns else "unknown",
            "all_target_success_rate": float(g["success_all"].mean()),
            "uniq_pct": 100.0 * (g["sequence"].nunique() / len(g)),
        }

        for col in ["sim_gen_within", "sim_gen_all", "sim_train"]:
            if col in g.columns:
                out[col + "_mean"] = float(np.nanmean(g[col].to_numpy(dtype=float)))
            else:
                out[col + "_mean"] = np.nan

        for feat in FEATURES:
            match_col = f"match_{feat}"
            if match_col not in g.columns:
                out[f"match_rate_{feat}"] = np.nan
                out[f"n_targeted_{feat}"] = 0
                continue
            mask_targeted = g["used_features"].apply(lambda u: feat in set(parse_list_field(u))).to_numpy(dtype=bool)
            vals = g.loc[mask_targeted, match_col].dropna()
            out[f"n_targeted_{feat}"] = int(vals.shape[0])
            out[f"match_rate_{feat}"] = float(vals.astype(bool).mean()) if len(vals) else np.nan

        rows.append(out)

    return pd.DataFrame(rows).sort_values("cond_idx").reset_index(drop=True)


def make_ci_table(cond_summary: pd.DataFrame, use_only_in_dist: bool = True) -> pd.DataFrame:
    df = cond_summary.copy()
    if use_only_in_dist and "ood_type" in df.columns:
        df = df[df["ood_type"] == "in_dist"].copy()

    metrics = [
        ("All-target success rate", "all_target_success_rate"),
        ("Within-cond uniqueness (%)", "uniq_pct"),
        (
            "Within-cond mean NED",
            "ned_within_mean_from_within_df" if "ned_within_mean_from_within_df" in df.columns else None,
        ),
        ("Within-cond mean NW %id", "sim_gen_within_mean"),
        ("Gen-vs-train mean NW %id", "sim_train_mean"),
    ]

    for feat in FEATURES:
        metrics.append((f"Match rate: {feat}", f"match_rate_{feat}"))

    rows = []
    for name, col in metrics:
        if col is None or col not in df.columns:
            continue
        mean, lo, hi = bootstrap_ci_over_conditions(df, col)
        rows.append({"metric": name, "mean": mean, "ci_low": lo, "ci_high": hi})

    out = pd.DataFrame(rows)

    def _fmt(row):
        m = row["mean"]
        lo = row["ci_low"]
        hi = row["ci_high"]
        if "rate" in row["metric"].lower() or "%id" in row["metric"].lower():
            return f"{100*m:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
        if "(%)" in row["metric"]:
            return f"{m:.1f} [{lo:.1f}, {hi:.1f}]"
        return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"

    out["mean [95% CI]"] = out.apply(_fmt, axis=1)
    return out[["metric", "mean [95% CI]"]]


def plot_within_condition_ned_by_k(cond_summary: pd.DataFrame, outpath: str):
    df = cond_summary.copy()
    df = df[df["ood_type"] == "in_dist"].copy() if "ood_type" in df.columns else df
    if "ned_within_mean_from_within_df" not in df.columns:
        raise ValueError("Need within-condition NED per condition. Merge within_df first.")
    df = df[np.isfinite(df["ned_within_mean_from_within_df"].to_numpy(dtype=float))]

    ks = sorted(df["k"].unique().tolist())
    data = [df.loc[df["k"] == k, "ned_within_mean_from_within_df"].to_numpy(dtype=float) for k in ks]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bp = ax.boxplot(data, positions=np.arange(1, len(ks) + 1), widths=0.6, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(TEAL[1])
        patch.set_alpha(0.85)
        patch.set_linewidth(0)
    for elem in ["whiskers", "caps", "medians"]:
        for line in bp[elem]:
            line.set_color("#3F3F3F")
            line.set_linewidth(1.2)

    for i, k in enumerate(ks, start=1):
        vals = df.loc[df["k"] == k, "ned_within_mean_from_within_df"].to_numpy(dtype=float)
        mean, lo, hi = bootstrap_ci_mean(vals, n_boot=4000)
        ax.plot([i], [mean], marker="o", markersize=5, color=TEAL[5])
        ax.vlines(i, lo, hi, color=TEAL[5], linewidth=2)

    ax.set_xticks(np.arange(1, len(ks) + 1))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Number of conditioned descriptors $k$")
    ax.set_ylabel("Within-condition mean NED")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(outpath, format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_within_condition_nw_by_k(cond_summary: pd.DataFrame, outpath: str):
    df = cond_summary.copy()
    df = df[df["ood_type"] == "in_dist"].copy() if "ood_type" in df.columns else df
    col = "sim_gen_within_mean"
    if col not in df.columns:
        raise ValueError("Need sim_gen_within_mean in cond_summary (computed from samples_df).")
    df = df[np.isfinite(df[col].to_numpy(dtype=float))]

    ks = sorted(df["k"].unique().tolist())
    data = [df.loc[df["k"] == k, col].to_numpy(dtype=float) for k in ks]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bp = ax.boxplot(data, positions=np.arange(1, len(ks) + 1), widths=0.6, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(TEAL[2])
        patch.set_alpha(0.85)
        patch.set_linewidth(0)
    for elem in ["whiskers", "caps", "medians"]:
        for line in bp[elem]:
            line.set_color("#3F3F3F")
            line.set_linewidth(1.2)

    for i, k in enumerate(ks, start=1):
        vals = df.loc[df["k"] == k, col].to_numpy(dtype=float)
        mean, lo, hi = bootstrap_ci_mean(vals, n_boot=4000)
        ax.plot([i], [mean], marker="o", markersize=5, color=TEAL[5])
        ax.vlines(i, lo, hi, color=TEAL[5], linewidth=2)

    ax.set_xticks(np.arange(1, len(ks) + 1))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Number of conditioned descriptors $k$")
    ax.set_ylabel("Within-condition NW identity")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(outpath, format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_marginal_distribution_overlay(train_df: pd.DataFrame, samples_df: pd.DataFrame, feature: str, pred_col: str, outpath: str):
    tr = train_df[feature].dropna().to_numpy(dtype=float)
    ge = samples_df[pred_col].dropna().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    lo = float(np.nanmin(np.concatenate([tr, ge])))
    hi = float(np.nanmax(np.concatenate([tr, ge])))
    bins = np.linspace(lo, hi, 35)

    ax.hist(tr, bins=bins, density=True, alpha=0.55, label="Training", color=TEAL[0])
    ax.hist(ge, bins=bins, density=True, alpha=0.55, label="Generated", color=TEAL[4])

    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=11)
    plt.tight_layout()
    fig.savefig(outpath, format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def build_error_df(samples_df: pd.DataFrame, feature: str, pred_col: str):
    df = samples_df.copy()
    df["used_set"] = df["used_features"].apply(lambda u: set(parse_list_field(u)))
    df = df[df["used_set"].apply(lambda s: feature in s)].copy()
    df = df[df[pred_col].notna()].copy()

    def get_target(p):
        if isinstance(p, dict) and feature in p:
            return p[feature]
        return np.nan

    df["target"] = df["params"].apply(get_target).astype(float)
    df = df[np.isfinite(df["target"].to_numpy(dtype=float))]
    df["error"] = df[pred_col].astype(float) - df["target"]
    return df[["cond_idx", "k", "ood_type", "error", "target", pred_col, "used_features"]]


def plot_error_box_by_k(error_df: pd.DataFrame, title: str, outpath: str):
    df = error_df.copy()
    df = df[df["ood_type"] == "in_dist"].copy() if "ood_type" in df.columns else df
    ks = sorted(df["k"].unique().tolist())
    data = [df.loc[df["k"] == k, "error"].to_numpy(dtype=float) for k in ks]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bp = ax.boxplot(data, positions=np.arange(1, len(ks) + 1), widths=0.6, patch_artist=True, showfliers=False)

    for patch in bp["boxes"]:
        patch.set_facecolor(TEAL[3])
        patch.set_alpha(0.85)
        patch.set_linewidth(0)

    for elem in ["whiskers", "caps", "medians"]:
        for line in bp[elem]:
            line.set_color("#3F3F3F")
            line.set_linewidth(1.2)

    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="#3F3F3F", alpha=0.8)
    ax.set_xticks(np.arange(1, len(ks) + 1))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Number of conditioned descriptors $k$")
    ax.set_ylabel("Predicted - target")
    ax.set_title(title)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(outpath, format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_results = RESULTS_DIR / "cvae_evaluation_results_with_matches.pkl"
    parser = argparse.ArgumentParser(description="Summarize conditional matching and plot diagnostics.")
    parser.add_argument(
        "--results-pkl",
        type=str,
        default=str(default_results),
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=str(DATA_PROCESSED / "merged_all.csv"),
    )
    parser.add_argument("--output-dir", type=str, default=str(FIGS_DIR))
    parser.add_argument("--plots", nargs="*", default=None)
    parser.add_argument("--ci-out", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    results_path = Path(args.results_pkl).expanduser().resolve()
    if not results_path.exists():
        fallback = (RESULTS_DIR / "cvae_evaluation_results.pkl").resolve()
        if fallback.exists():
            print(f"Results not found at {results_path}. Falling back to {fallback}.")
            results_path = fallback
        else:
            raise FileNotFoundError(f"Missing results file: {results_path}")

    samples_df, within_df, _ = load_results_pickle(str(results_path))
    cond_summary = build_condition_summary(samples_df, include_length_in_k=True)
    if within_df is not None:
        cond_summary = cond_summary.merge(
            within_df[["cond_idx", "mean"]].rename(columns={"mean": "ned_within_mean_from_within_df"}),
            on="cond_idx",
            how="left",
        )

    ci_table = make_ci_table(cond_summary, use_only_in_dist=True)
    print(ci_table.to_latex(index=False, escape=False))

    if args.ci_out:
        ci_path = Path(args.ci_out).expanduser().resolve()
        ci_table.to_csv(ci_path, index=False)
        print("Wrote", ci_path)

    plot_map = {
        "within_ned": lambda: plot_within_condition_ned_by_k(cond_summary, outpath=str(output_dir / "figS_within_ned_by_k.svg")),
        "within_nw": lambda: plot_within_condition_nw_by_k(cond_summary, outpath=str(output_dir / "figS_within_nw_by_k.svg")),
        "ap_marginal": lambda: plot_marginal_distribution_overlay(
            pd.read_csv(args.data_csv, keep_default_na=False, na_values=[""]),
            samples_df[samples_df["ood_type"] == "in_dist"],
            "ap",
            "pred_ap",
            str(output_dir / "fig_ap_marginal.svg"),
        ),
        "ap_error": lambda: plot_error_box_by_k(
            build_error_df(samples_df.assign(k=samples_df["used_features"].apply(lambda u: count_k_used(u, include_length=True))), "ap", "pred_ap"),
            "AP centering error",
            str(output_dir / "fig_ap_error_by_k.svg"),
        ),
    }
    if "pred_ap" not in samples_df.columns:
        print("Missing pred_ap in results. Run condition_matching_report --save-updated-pkl to enable AP plots.")
        plot_map.pop("ap_marginal")
        plot_map.pop("ap_error")

    selected = args.plots or list(plot_map.keys())
    for name in selected:
        if name not in plot_map:
            raise ValueError(f"Unknown plot name: {name}")
        plot_map[name]()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
