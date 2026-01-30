#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA_RAW, DATA_PROCESSED, RESULTS_DIR, TEAL, ensure_dir, set_paper_style


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)


AMINO_ACID_DICT = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe", "G": "Gly",
    "H": "Hse", "I": "Ile", "K": "Lys", "L": "Leu", "M": "Met", "N": "Asn",
    "P": "Pro", "Q": "Gln", "R": "Arg", "S": "Ser", "T": "Thr", "V": "Val",
    "W": "Trp", "Y": "Tyr",
}
AMINO_ACID_DICT_INV = {v.upper(): k for k, v in AMINO_ACID_DICT.items()}


def load_processed_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PROCESSED / "merged_all_no_norm.csv", keep_default_na=False, na_values=[""])
    data["length"] = data["peptide"].apply(len)
    return data


def load_no_ap() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "peptide_metrics_no_ap.csv")


def load_beyond_ap() -> pd.DataFrame:
    beyond = pd.read_csv(DATA_RAW / "beyond_tri.txt", sep=" ", header=None)
    beyond["peptide"] = beyond[0].apply(convert_to_1_letter)
    beyond.rename(columns={1: "ap_beyond"}, inplace=True)
    beyond.drop(columns=[0], inplace=True)
    beyond["len"] = beyond["peptide"].apply(len)
    return beyond


def convert_to_1_letter(peptide: str) -> str:
    return "".join([AMINO_ACID_DICT_INV[aa] for aa in peptide.split("-")])


def aa_freq_from_peptides(peptide_series: pd.Series) -> pd.Series:
    s = peptide_series.dropna().astype(str).str.upper()
    counts = dict.fromkeys(AA_ORDER, 0)

    for pep in s:
        for ch in pep:
            if ch in AA_SET:
                counts[ch] += 1

    total = sum(counts.values())
    if total == 0:
        return pd.Series({aa: 0.0 for aa in AA_ORDER})

    return pd.Series(counts, dtype=float) / float(total)


def plot_ap_by_length(data: pd.DataFrame, output_path: Optional[Path], show: bool) -> None:
    df = data[["length", "ap"]].dropna().copy()
    df["length"] = df["length"].astype(int)
    df["ap"] = df["ap"].astype(float)
    lengths = np.sort(df["length"].unique())

    stats = (df.groupby("length")["ap"]
               .agg(mean="mean", std="std", median="median")
               .reindex(lengths))

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        [df.loc[df["length"] == L, "ap"].values for L in lengths],
        positions=np.arange(len(lengths)) + 1,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    for b in bp["boxes"]:
        b.set_facecolor(TEAL[0])
        b.set_edgecolor(TEAL[4])
        b.set_linewidth(1.5)
        b.set_alpha(0.9)

    for w in bp["whiskers"]:
        w.set_color(TEAL[4])
    for c in bp["caps"]:
        c.set_color(TEAL[4])
    for m in bp["medians"]:
        m.set_color(TEAL[5])

    x = np.arange(len(lengths)) + 1
    ax.errorbar(
        x,
        stats["mean"].values,
        yerr=stats["std"].values,
        fmt="o",
        markersize=5,
        linewidth=1.5,
        capsize=4,
        color=TEAL[4],
        alpha=0.95,
        label="Mean ± std",
    )

    ax.axhline(y=1.65, linestyle="--", linewidth=2, color="#C44E52", label="AP = 1.65")
    ax.axhline(y=1.80, linestyle="--", linewidth=2, color="#55A868", label="AP = 1.80")

    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("Peptide length")
    ax.set_ylabel("Aggregation propensity (AP)")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=11,
    )

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, format=output_path.suffix.lstrip("."), bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_aa_distribution(data: pd.DataFrame, no_ap: pd.DataFrame, output_path: Optional[Path], show: bool) -> None:
    data_peps = data["peptide"].dropna().astype(str)
    no_ap_peps = no_ap["sequence"].dropna().astype(str)

    no_ap_set = set(no_ap_peps.tolist())
    data_minus_no_ap = data.loc[~data_peps.isin(no_ap_set), "peptide"]

    freq_no_ap = aa_freq_from_peptides(no_ap_peps)
    freq_data_minus = aa_freq_from_peptides(data_minus_no_ap)
    freq_data_all = aa_freq_from_peptides(data_peps)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(AA_ORDER))
    w = 0.28

    ax.bar(
        x - w,
        100.0 * freq_no_ap.values,
        width=w,
        edgecolor=TEAL[4],
        linewidth=1.2,
        alpha=0.95,
        label="Random Sequences",
        color=TEAL[1],
    )
    ax.bar(
        x,
        100.0 * freq_data_minus.values,
        width=w,
        edgecolor=TEAL[4],
        linewidth=1.2,
        alpha=0.95,
        label="PepMorph Dataset \\ Random Sequences",
        color=TEAL[3],
    )
    ax.bar(
        x + w,
        100.0 * freq_data_all.values,
        width=w,
        edgecolor=TEAL[4],
        linewidth=1.2,
        alpha=0.95,
        label="PepMorph Dataset",
        color=TEAL[5],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(AA_ORDER)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Residue frequency (%)")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=11,
    )

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, format=output_path.suffix.lstrip("."), bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_beyond_ap_diff(merged_aps: pd.DataFrame, beyond: pd.DataFrame, output_path: Optional[Path], show: bool) -> None:
    merged = pd.merge(merged_aps, beyond[["peptide", "ap_beyond"]], on="peptide", how="inner")
    merged = merged[merged["len"] > 3]
    print("AP Beyond - AP summary:")
    print((merged["ap_beyond"] - merged["ap"]).describe())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(merged["ap_beyond"] - merged["ap"], bins=30, edgecolor="black")
    ax.set_title("Distribution of Differences between AP Beyond and AP")
    ax.set_xlabel("Difference (AP Beyond - AP)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.75)
    ax.axvline(0, color="red", linestyle="dashed", linewidth=1)

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, format=output_path.suffix.lstrip("."), bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_length_distributions(data: pd.DataFrame, output_dir: Optional[Path], show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data["length"], bins=50)
    ax.set_xlabel("Length of peptide (number of aa)")
    ax.set_ylabel("Frequency")
    ax.set_title("Length Distribution of Peptides")
    plt.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / "length_distribution_all.pdf", bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    data["length"][data["length"] < 11].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xlabel("Length of peptide (number of aa)")
    ax.set_ylabel("Frequency")
    ax.set_title("Length Distribution of Short Peptides")
    plt.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / "length_distribution_short.pdf", bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_hydrophobic_moment_distribution(data: pd.DataFrame, output_path: Optional[Path], show: bool) -> None:
    data_no_nan = data.dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data_no_nan["hydrophobic_moment"], bins=30, alpha=0.5, label="No NaN", edgecolor="black")
    ax.hist(data["hydrophobic_moment"].dropna(), bins=30, alpha=0.5, label="With NaN", edgecolor="black")
    ax.set_title("Distribution of Hydrophobic Moment")
    ax.set_xlabel("Hydrophobic Moment")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, format=output_path.suffix.lstrip("."), bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def print_basic_stats(data: pd.DataFrame) -> None:
    stats = data.groupby("length")["ap"].describe()
    print(stats)
    print("Has beta_sheet_content (no NaN):")
    print(data.dropna()["has_beta_sheet_content"].value_counts())
    print("Has beta_sheet_content (all):")
    print(data["has_beta_sheet_content"].value_counts())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset analysis plots")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-show", action="store_true", help="Do not display figures interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    show = not args.no_show

    set_paper_style()
    data = load_processed_data()
    no_ap = load_no_ap()
    beyond = load_beyond_ap()
    merged_aps = pd.read_csv(DATA_RAW / "merged_aps.csv")

    print_basic_stats(data)

    plot_ap_by_length(data, output_dir / "ap_by_length_boxplot.pdf", show=show)
    plot_aa_distribution(data, no_ap, output_dir / "aa_distribution_three_cohorts.pdf", show=show)
    plot_beyond_ap_diff(merged_aps, beyond, output_dir / "ap_beyond_diff_hist.pdf", show=show)
    plot_length_distributions(data, output_dir, show=show)
    plot_hydrophobic_moment_distribution(data, output_dir / "hydrophobic_moment_dist.pdf", show=show)


if __name__ == "__main__":
    main()
