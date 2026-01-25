#!/usr/bin/env python

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

from common import DATA_PROCESSED, FIGS_DIR, GEN_PEPTIDES_DIR, RESULTS_DIR, TEAL, load_split_indices, set_paper_style
from utils import AA_ALPHABET, aa_freqs, parse_list_field, read_sequences_csv, read_sequences_txt

AA_LIST = list(AA_ALPHABET)


def load_validation_results(path: Path) -> dict:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return data


def load_train_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    split_idx = load_split_indices()
    if split_idx:
        return df.iloc[split_idx["train"]].copy()
    train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df["length"], random_state=42)
    train_df, _ = train_test_split(train_val_df, test_size=0.1, stratify=train_val_df["length"], random_state=42)
    _ = test_df
    return train_df


def plot_aa_composition_evolution(gen_dir: Path, output_dir: Path) -> None:
    paths = {
        "spheres": {
            "init": "generated_spheres_init_new.txt",
            "filtered": "filtered_ap_peptides_spheres_new.txt",
            "final": "valid_spheres_new.csv",
        },
        "fibers": {
            "init": "generated_fibers_init.txt",
            "filtered": "filtered_ap_peptides_fiber_final.txt",
            "final": "valid_fibers_new.csv",
        },
    }

    def load_three_stages(cfg: dict) -> dict:
        init = read_sequences_txt(gen_dir / cfg["init"])
        filtered = read_sequences_txt(gen_dir / cfg["filtered"])
        final = read_sequences_csv(gen_dir / cfg["final"], column="sequence")
        return {
            "init": {"seqs": init, "freq": aa_freqs(init, AA_ALPHABET) * 100},
            "filter": {"seqs": filtered, "freq": aa_freqs(filtered, AA_ALPHABET) * 100},
            "final": {"seqs": final, "freq": aa_freqs(final, AA_ALPHABET) * 100},
        }

    data = {morph: load_three_stages(paths[morph]) for morph in ["spheres", "fibers"]}

    global_max = 0.0
    for morph in data:
        for stage in ["init", "filter", "final"]:
            global_max = max(global_max, float(data[morph][stage]["freq"].max()))
    ylim_top = max(global_max, 0.15) * 1.10

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    stage_style = [
        ("init", "Initial", TEAL[1]),
        ("filter", "AP-filtered", TEAL[3]),
        ("final", "Validated", TEAL[5]),
    ]

    for ax, morph_title in zip(axes, ["Spheres", "Fibers"]):
        d = data[morph_title.lower()]
        for key, label, color in stage_style:
            frq = d[key]["freq"]
            n = len(d[key]["seqs"])
            ax.plot(AA_LIST, frq, marker="o", linewidth=2, markersize=4, label=f"{label} (n={n})", color=color)
        ax.set_ylim(0, ylim_top)
        sns.despine(ax=ax)
        ax.legend(frameon=True, fontsize=12)

    plt.tight_layout()
    fig.savefig(output_dir / "plot_aa_composition_evolution.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)

    stage_style = [
        ("init", "Initial", TEAL[1]),
        ("filter", "AP-filtered", TEAL[3]),
        ("final", "Validated", TEAL[5]),
    ]

    def plot_single(morph_key: str, outfile: str) -> None:
        d = data[morph_key]
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
        for key, label, color in stage_style:
            frq = d[key]["freq"]
            n = len(d[key]["seqs"])
            ax.plot(AA_LIST, frq, marker="o", linewidth=2, markersize=4, label=f"{label} (n={n})", color=color)
        ax.set_ylim(0, ylim_top)
        sns.despine(ax=ax)
        ax.legend(frameon=True, fontsize=12)
        plt.tight_layout()
        fig.savefig(output_dir / outfile, format="svg", bbox_inches="tight", dpi=300)
        plt.close(fig)

    plot_single("spheres", "plot_aa_composition_spheres.svg")
    plot_single("fibers", "plot_aa_composition_fibers.svg")


def plot_aa_composition_train_vs_generated(samples_df: pd.DataFrame, train_df: pd.DataFrame, output_dir: Path) -> None:
    train_peps = train_df["peptide"].dropna().astype(str).tolist()

    mask_in = samples_df["ood_type"].eq("in_dist")
    gen_in = samples_df.loc[mask_in, "sequence"].dropna().astype(str).tolist()
    gen_ood = samples_df.loc[~mask_in, "sequence"].dropna().astype(str).tolist()

    f_train = aa_freqs(train_peps, AA_ALPHABET)
    f_in = aa_freqs(gen_in, AA_ALPHABET)
    f_ood = aa_freqs(gen_ood, AA_ALPHABET) if len(gen_ood) else None

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(AA_LIST, f_train, marker="o", linewidth=2, markersize=4, color=TEAL[1], label="Training")
    ax.plot(AA_LIST, f_in, marker="o", linewidth=2, markersize=4, color=TEAL[4], label="Generated (common)")
    if f_ood is not None:
        ax.plot(AA_LIST, f_ood, marker="o", linewidth=2, markersize=4, color=TEAL[2], label="Generated (rare)")

    ax.set_ylim(0, max(float(f_train.max()), float(f_in.max()), 0.15) * 1.10)
    sns.despine(ax=ax)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "plot_aa_composition.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_length_confusion(samples_df: pd.DataFrame, output_dir: Path) -> None:
    mask_in = samples_df["ood_type"].eq("in_dist")
    df = samples_df.loc[mask_in, ["target_length", "length"]].dropna().astype(int)

    cm = pd.crosstab(df["target_length"], df["length"]).sort_index(axis=0).sort_index(axis=1)
    row_pct = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    cmap = sns.light_palette(TEAL[4], as_cmap=True)
    sns.heatmap(
        row_pct,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Row %"},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 12},
    )

    ax.set_xlabel("Generated length")
    ax.set_ylabel("Target length")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    fig.savefig(output_dir / "plot_length_confusion_indist.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_positional_aa(samples_df: pd.DataFrame, train_df: pd.DataFrame, output_dir: Path, length: int = 6) -> None:
    def pos_aa_matrix(seq_list: list[str], length: int) -> pd.DataFrame:
        mat = pd.DataFrame(0, index=AA_LIST, columns=[f"{i+1}" for i in range(length)], dtype=float)
        for seq in seq_list:
            if len(seq) != length:
                continue
            for i, ch in enumerate(seq):
                if ch in AA_ALPHABET:
                    mat.iloc[AA_LIST.index(ch), i] += 1
        col_sums = mat.sum(axis=0).replace(0, np.nan)
        mat = mat.div(col_sums, axis=1).fillna(0.0)
        return mat

    train_peps = train_df["peptide"].dropna().astype(str).tolist()
    gen_len = samples_df.loc[
        samples_df["length"].eq(length) & samples_df["ood_type"].eq("in_dist"),
        "sequence",
    ].astype(str).tolist()
    trn_len = [s for s in train_peps if len(s) == length]

    mat_train = pos_aa_matrix(trn_len, length)
    mat_gen = pos_aa_matrix(gen_len, length)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    cmap = sns.light_palette(TEAL[4], as_cmap=True)

    sns.heatmap(
        mat_train,
        ax=axes[0],
        cmap=cmap,
        vmin=0,
        vmax=mat_train.values.max() * 1.0,
        cbar=False,
        linewidths=0.4,
        linecolor="white",
    )
    axes[0].set_title(f"Training ({length}-mers)", color=rcParams["text.color"])
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Amino acid")

    sns.heatmap(
        mat_gen,
        ax=axes[1],
        cmap=cmap,
        vmin=0,
        vmax=mat_train.values.max() * 1.0,
        cbar_kws={"label": "Frequency"},
        linewidths=0.4,
        linecolor="white",
    )
    axes[1].set_title(f"Generated ({length}-mers)", color=rcParams["text.color"])
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("")

    for ax in axes:
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig.savefig(output_dir / "plot_positional_aa_6mer.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_nn_ecdf(samples_df: pd.DataFrame, novelty_df: pd.DataFrame | None, output_dir: Path) -> None:
    if "nn_dist" not in samples_df.columns and isinstance(novelty_df, pd.DataFrame):
        samples_df = samples_df.merge(novelty_df[["sequence", "nn_dist"]], on="sequence", how="left")

    def ecdf(x: np.ndarray):
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    mask_in = samples_df["ood_type"] == "in_dist"
    nn_in = samples_df.loc[mask_in, "nn_dist"].dropna().to_numpy()
    nn_ood = samples_df.loc[~mask_in, "nn_dist"].dropna().to_numpy()

    nn_in = np.concatenate((nn_in, [1.0]))
    nn_ood = np.concatenate((nn_ood, [1.0])) if len(nn_ood) else np.array([1.0])

    x_in, y_in = ecdf(nn_in)
    x_ood, y_ood = ecdf(nn_ood) if len(nn_ood) else (np.array([]), np.array([]))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x_in, y_in, color=TEAL[3], linewidth=2, label="Common Conditions")
    if len(x_ood):
        ax.plot(x_ood, y_ood, color=TEAL[1], linewidth=2, label="Rare Conditions")

    sns.despine(ax=ax)
    ax.legend(frameon=True)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(output_dir / "plot_nn_ecdf.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_within_diversity_vs_k(samples_df: pd.DataFrame, within_df: pd.DataFrame, div_stats: dict, output_dir: Path) -> None:
    if within_df is None:
        raise ValueError("within_df is required for within-diversity plots.")
    k_map = samples_df.groupby("cond_idx")["used_features"].first().apply(lambda v: len(parse_list_field(v)))
    ood_map = samples_df.groupby("cond_idx")["ood_type"].first()

    plot_df = within_df.copy()
    plot_df["k_used"] = plot_df["cond_idx"].map(k_map)
    plot_df["ood_type"] = plot_df["cond_idx"].map(ood_map)
    plot_df = plot_df[plot_df["ood_type"] == "in_dist"].dropna(subset=["k_used", "mean"])
    plot_df["k_used"] = plot_df["k_used"].astype(int)

    across_mean = np.nan
    if isinstance(div_stats, dict) and "across_condition" in div_stats:
        across_mean = float(div_stats["across_condition"].get("mean", np.nan))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=plot_df, x="k_used", y="mean", color=TEAL[4], width=0.6, ax=ax)

    if not np.isnan(across_mean):
        ax.axhline(across_mean, color=TEAL[2], linestyle="--", linewidth=2)

    ax.set_xlabel("Number of conditioned descriptors (k_used)")
    ax.set_ylabel("Within-condition mean NED")

    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(output_dir / "plot_within_diversity_vs_k.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_similarity_kde(samples_df: pd.DataFrame, output_dir: Path) -> None:
    needed = ["sim_train", "sim_gen_within", "sim_gen_all"]
    missing = [c for c in needed if c not in samples_df.columns]
    if missing:
        raise ValueError(f"Missing similarity columns in samples_df: {missing}")

    sim_train = samples_df["sim_train"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    sim_within = samples_df["sim_gen_within"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    sim_all = samples_df["sim_gen_all"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    if len(sim_train):
        sns.kdeplot(sim_train, ax=ax, color=TEAL[1], linewidth=2, label="Sim_train")
    if len(sim_all):
        sns.kdeplot(sim_all, ax=ax, color=TEAL[3], linewidth=2, label="Sim_gen_all")
    if len(sim_within):
        sns.kdeplot(sim_within, ax=ax, color=TEAL[5], linewidth=2, label="Sim_gen_within")

    ax.set_xlabel("Similarity (NW % identity)")
    ax.set_ylabel("Density")

    xmax = max(
        (sim_train.max() if len(sim_train) else 0),
        (sim_within.max() if len(sim_within) else 0),
        (sim_all.max() if len(sim_all) else 0),
        0.2,
    )
    ax.set_xlim(0.0, min(1.0, xmax * 1.1))

    sns.despine(ax=ax)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(output_dir / "plot_njirjak_similarity_kde.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_novelty_similarity_scatter(samples_df: pd.DataFrame, output_dir: Path) -> None:
    needed = ["sim_train", "sim_gen_within", "used_features"]
    missing = [c for c in needed if c not in samples_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for scatter plot: {missing}")

    def k_group(k: int) -> str:
        if k <= 2:
            return "k = 1-2"
        if k <= 4:
            return "k = 3-4"
        return "k = 5-6"

    df = samples_df.copy()
    df["k_used"] = df["used_features"].apply(lambda v: len(parse_list_field(v))).astype(int)
    df["k_group"] = df["k_used"].apply(k_group)

    group_order = ["k = 1-2", "k = 3-4", "k = 5-6"]
    palette_map = {
        "k = 1-2": TEAL[1],
        "k = 3-4": TEAL[3],
        "k = 5-6": TEAL[5],
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for group in group_order:
        sub = df[df["k_group"] == group]
        if len(sub):
            ax.scatter(
                sub["sim_train"] * 100,
                sub["sim_gen_within"] * 100,
                s=9,
                alpha=0.75,
                linewidths=0,
                color=palette_map[group],
                label=group,
            )

    x = df["sim_train"].astype(float)
    y = df["sim_gen_within"].astype(float)
    xmin = min(0.05, float(np.nanpercentile(x, 0.5)))
    ymin = min(0.02, float(np.nanpercentile(y, 0.5)))
    xmax = max(0.12, float(np.nanpercentile(x, 99.5)))
    ymax = max(0.14, float(np.nanpercentile(y, 99.5)))
    ax.set_xlim(xmin * 0.95 * 100, xmax * 1.05 * 100)
    ax.set_ylim(ymin * 0.95 * 100, ymax * 1.05 * 100)

    sns.despine(ax=ax)

    ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor=rcParams["axes.labelcolor"],
        facecolor="white",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=10,
        markerscale=2.2,
        scatterpoints=1,
        handletextpad=0.3,
        borderpad=0.25,
        labelspacing=0.3,
        columnspacing=0.6,
        ncol=1,
    )

    plt.tight_layout()
    fig.savefig(
        output_dir / "plot_novelty_vs_within_similarity_scatter_grouped.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def plot_condition_scatter(samples_df: pd.DataFrame, novelty_df: pd.DataFrame | None, output_dir: Path) -> None:
    if "nn_dist" not in samples_df.columns and isinstance(novelty_df, pd.DataFrame):
        samples_df = samples_df.merge(novelty_df[["sequence", "nn_dist"]], on="sequence", how="left")

    g = samples_df.copy()
    g["k_used"] = g["used_features"].apply(lambda v: len(parse_list_field(v))).astype(int)

    agg = g.groupby("cond_idx").agg(
        median_nn_dist=("nn_dist", "median"),
        mean_sim_within=("sim_gen_within", "mean"),
        k_used=("k_used", "first"),
        ood_type=("ood_type", "first"),
    ).reset_index()

    palette = {k: TEAL[min(k - 1, len(TEAL) - 1)] for k in sorted(agg["k_used"].unique())}

    fig, ax = plt.subplots(figsize=(10, 6))

    for k in sorted(agg["k_used"].unique()):
        sub_in = agg[(agg["k_used"] == k) & (agg["ood_type"] == "in_dist")]
        sub_ood = agg[(agg["k_used"] == k) & (agg["ood_type"] != "in_dist")]
        if len(sub_in):
            ax.scatter(
                sub_in["median_nn_dist"],
                sub_in["mean_sim_within"],
                s=45,
                color=palette[k],
                alpha=0.75,
                edgecolors="none",
                label=f"k={k}" if k not in [*ax.get_legend_handles_labels()[1]] else None,
                marker="o",
            )
        if len(sub_ood):
            ax.scatter(
                sub_ood["median_nn_dist"],
                sub_ood["mean_sim_within"],
                s=65,
                color=palette[k],
                alpha=0.95,
                edgecolors="black",
                linewidths=0.5,
                marker="s",
            )

    x_med = np.nanmedian(agg["median_nn_dist"])
    y_med = np.nanmedian(agg["mean_sim_within"])
    ax.axvline(x_med, color=TEAL[2], linestyle="--", linewidth=2)
    ax.axhline(y_med, color=TEAL[2], linestyle="--", linewidth=2)

    ax.set_xlabel("Per-condition median nearest-train NED")
    ax.set_ylabel("Per-condition mean within similarity (NW % identity)")
    sns.despine(ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="k_used", frameon=False, ncol=min(3, len(labels)))
    plt.tight_layout()
    fig.savefig(output_dir / "plot_condition_novelty_vs_within_similarity.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_ap_vs_predap(gen_dir: Path, output_dir: Path) -> None:
    df = pd.read_csv(gen_dir / "rmoi_and_ap_by_run.csv")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    min_ap, max_ap = 0.959986, 2.89703

    filtered_fibers = pd.read_csv(
        gen_dir / "filtered_ap_peptides_fiber_final.txt",
        header=None,
        names=["peptide", "peptide_3", "pred_ap", "pred_clas"],
    )
    filtered_spheres = pd.read_csv(
        gen_dir / "filtered_ap_peptides_spheres.txt",
        header=None,
        names=["peptide", "peptide_3", "pred_ap", "pred_clas"],
    )

    def unnormalize_ap(ap):
        return (ap * (max_ap - min_ap)) + min_ap

    filtered_fibers["pred_ap"] = unnormalize_ap(filtered_fibers["pred_ap"])
    filtered_spheres["pred_ap"] = unnormalize_ap(filtered_spheres["pred_ap"])

    combined_df = pd.concat([filtered_fibers, filtered_spheres], ignore_index=True)
    df = df.merge(combined_df[["peptide", "pred_ap", "pred_clas"]], on="peptide", how="left")

    point = TEAL[4]
    line = TEAL[2]

    d = df[["aggregation_propensity", "pred_ap"]].dropna()
    x = d["aggregation_propensity"].to_numpy(dtype=float)
    y = d["pred_ap"].to_numpy(dtype=float)

    res = y - x
    mae = np.mean(np.abs(res))
    rmse = np.sqrt(np.mean(res**2))
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((x - x.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lims = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(
        x,
        y,
        s=25,
        alpha=0.9,
        edgecolors=point,
        linewidths=0.5,
        color=[point if cl == "fibers" else TEAL[1] for cl in df["morphology"]],
        zorder=2,
    )
    ax.plot(lims, lims, linestyle="--", color=line, linewidth=1.6, zorder=1)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    sns.despine(ax=ax)

    txt = f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nR^2 = {r2:.3f}"
    ax.text(
        0.75,
        0.2,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.8),
    )

    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=point, markersize=8, label="Fibers"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=TEAL[1], markersize=8, label="Spheres"),
        ],
        loc="upper left",
        frameon=True,
        fontsize=12,
    )

    plt.tight_layout()
    fig.savefig(output_dir / "ap_vs_predap.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_rmoi_df(gen_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(gen_dir / "rmoi_and_ap_by_run.csv")
    df["group"] = "all"
    low_thr, high_thr = 0.35, 0.75

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    _ = low_thr, high_thr
    return df


def plot_rmoi_violin(df: pd.DataFrame, output_dir: Path) -> None:
    if df is None:
        raise ValueError("RMOI dataframe is required for rmoi_violin plot.")
    low_thr, high_thr = 0.35, 0.75

    palette = {"spheres": TEAL[2], "fibers": TEAL[4]}
    line = TEAL[3]
    fill_low, fill_high = TEAL[0], TEAL[0]
    good, bad = TEAL[0], "#B5B5B5"

    fig, ax = plt.subplots(figsize=(5.5, 7))
    sns.violinplot(
        x="group",
        y="RMOI",
        hue="morphology",
        data=df,
        split=True,
        inner=None,
        palette=palette,
        bw=0.2,
        linewidth=1.2,
        gap=0.025,
        ax=ax,
    )
    ax.get_legend().remove()

    ax.set_ylim(0.0, 1)
    ax.axhline(low_thr, color=line, linestyle="--", linewidth=1.5, zorder=1)
    ax.axhline(high_thr, color=line, linestyle="--", linewidth=1.5, zorder=1)
    ax.axhspan(0.0, low_thr, facecolor=fill_low, alpha=0.25, zorder=0)
    ax.axhspan(high_thr, 1.0, facecolor=fill_high, alpha=0.25, zorder=0)

    center = ax.get_xticks()[0]
    offset = 0.18
    for morph, sign in [("spheres", -1), ("fibers", +1)]:
        sub = df.loc[df["morphology"] == morph, "RMOI"].values
        xs = np.random.normal(loc=center + sign * offset, scale=0.025, size=len(sub))
        if morph == "spheres":
            cols = [good if y > high_thr else bad for y in sub]
        else:
            cols = [good if y < low_thr else bad for y in sub]
        ax.scatter(xs, sub, c=cols, edgecolor="k", linewidth=0.4, alpha=0.9, s=32, zorder=2)

    ax.set_xticks([center - offset, center + offset])
    ax.set_xticklabels(["spheres", "fibers"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([0, low_thr, high_thr, 1.0])
    ax.set_yticklabels(["0.0", f"{low_thr:.2f}", f"{high_thr:.2f}", "1.0"])
    ax.grid(False)
    sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(output_dir / "rmoi_split_violin.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_aggregation_kde(df: pd.DataFrame, output_dir: Path) -> None:
    if df is None:
        raise ValueError("RMOI dataframe is required for aggregation KDE plot.")
    agg_sph = df.loc[df["morphology"] == "spheres", "aggregation_propensity"].dropna().values
    agg_fib = df.loc[df["morphology"] == "fibers", "aggregation_propensity"].dropna().values

    x = np.linspace(1.0, 4.0, 400)
    kde_sph = gaussian_kde(agg_sph, bw_method="scott")
    kde_fib = gaussian_kde(agg_fib, bw_method="scott")

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    y_s = kde_sph(x)
    y_f = kde_fib(x)
    ax.plot(x, y_s, color=TEAL[2], linewidth=2.2, linestyle="--", label="spheres")
    ax.fill_between(x, y_s, alpha=0.18, color=TEAL[2])
    ax.plot(x, y_f, color=TEAL[4], linewidth=2.2, linestyle="--", label="fibers")
    ax.fill_between(x, y_f, alpha=0.25, color=TEAL[4])

    ax.axvline(1.8, color=TEAL[2], linestyle="--", linewidth=2)

    jitter = 0.003
    ax.scatter(
        agg_sph,
        np.random.uniform(-jitter, jitter, len(agg_sph)),
        color=TEAL[4],
        edgecolor="k",
        linewidth=0.4,
        s=28,
        alpha=0.9,
        zorder=3,
    )
    ax.scatter(
        agg_fib,
        np.random.uniform(-jitter, jitter, len(agg_fib)),
        color=TEAL[2],
        edgecolor="k",
        linewidth=0.4,
        s=28,
        alpha=0.7,
        zorder=3,
    )

    ax.set_xlim(1.0, 4.0)
    ax.set_xticks([1.0, 2.0, 3.0, 4.0])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.grid(False)
    sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(output_dir / "aggregation_propensity_kde.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validation plots.")
    parser.add_argument(
        "--results-pkl",
        type=str,
        default=str(RESULTS_DIR / "cvae_evaluation_results.pkl"),
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=str(DATA_PROCESSED / "merged_all.csv"),
    )
    parser.add_argument("--gen-dir", type=str, default=str(GEN_PEPTIDES_DIR))
    parser.add_argument("--output-dir", type=str, default=str(FIGS_DIR))
    parser.add_argument(
        "--plots",
        nargs="*",
        default=None,
        help="Optional subset of plots to run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    results_path = Path(args.results_pkl).expanduser().resolve()
    data_path = Path(args.data_csv).expanduser().resolve()
    gen_dir = Path(args.gen_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    results = load_validation_results(results_path)
    samples_df: pd.DataFrame = results["samples_df"]
    novelty_df: pd.DataFrame | None = results.get("novelty_df")
    within_df: pd.DataFrame | None = results.get("within_df")
    div_stats = results.get("diversity_stats", {})

    train_df = load_train_df(data_path)

    rmoi_df = None
    if args.plots is None or any(p in (args.plots or []) for p in ["rmoi_violin", "aggregation_kde"]):
        rmoi_df = load_rmoi_df(gen_dir)

    plot_map = {
        "aa_evolution": lambda: plot_aa_composition_evolution(gen_dir, output_dir),
        "aa_train_gen": lambda: plot_aa_composition_train_vs_generated(samples_df, train_df, output_dir),
        "length_confusion": lambda: plot_length_confusion(samples_df, output_dir),
        "positional_aa": lambda: plot_positional_aa(samples_df, train_df, output_dir, length=6),
        "nn_ecdf": lambda: plot_nn_ecdf(samples_df, novelty_df, output_dir),
        "within_diversity": lambda: plot_within_diversity_vs_k(samples_df, within_df, div_stats, output_dir),
        "similarity_kde": lambda: plot_similarity_kde(samples_df, output_dir),
        "novelty_scatter": lambda: plot_novelty_similarity_scatter(samples_df, output_dir),
        "condition_scatter": lambda: plot_condition_scatter(samples_df, novelty_df, output_dir),
        "ap_vs_predap": lambda: plot_ap_vs_predap(gen_dir, output_dir),
        "rmoi_violin": lambda: plot_rmoi_violin(rmoi_df, output_dir),
        "aggregation_kde": lambda: plot_aggregation_kde(rmoi_df, output_dir),
    }

    selected = args.plots or list(plot_map.keys())
    for name in selected:
        if name not in plot_map:
            raise ValueError(f"Unknown plot name: {name}")
        plot_map[name]()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
