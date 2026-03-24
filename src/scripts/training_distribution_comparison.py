#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from common import DATA_PROCESSED, DATA_RAW, RESULTS_DIR, TEAL, ensure_dir, set_paper_style
from modeling.validation.common import load_split_indices
from modeling.validation.utils import AA_ALPHABET, aa_freqs


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "modeling" / "masked_cvae" / "config.yaml"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "train_distribution_comparison"
LABEL_DATASET_MINUS_RANDOM = "PepMorph Dataset \\ Random Sequences"
LABEL_DATASET_MINUS_RANDOM_WRAPPED = "PepMorph Dataset\n\\ Random Sequences"
LABEL_DATASET = "PepMorph Dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare train-set distributions with and without synthetic augmentation.")
    parser.add_argument("--data-csv", type=str, default=str(DATA_PROCESSED / "merged_all_no_norm.csv"))
    parser.add_argument("--random-sequences-csv", type=str, default=str(DATA_RAW / "peptide_metrics_no_ap.csv"))
    parser.add_argument("--splits-dir", type=str, default="")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    return payload or {}


def get_git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_train_split(raw_path: Path, splits_dir: Path | None) -> pd.DataFrame:
    df = pd.read_csv(raw_path, keep_default_na=False, na_values=[""])
    df["length"] = df["peptide"].astype(str).str.len()

    split_idx = load_split_indices(splits_dir) if splits_dir is not None else load_split_indices()
    if split_idx is None:
        raise ValueError("Missing split files; expected deterministic train split under data/splits.")

    train_df = df.iloc[split_idx["train"]].copy().reset_index(drop=True)
    train_df["variant_source"] = "literature"
    train_df["is_random_sequence"] = False
    return train_df


def annotate_random_sequences(train_df: pd.DataFrame, random_sequences_csv: Path) -> pd.DataFrame:
    df = train_df.copy()
    random_df = pd.read_csv(random_sequences_csv, keep_default_na=False).rename(columns={"sequence": "peptide"})
    random_set = set(random_df["peptide"].astype(str))
    df["is_random_sequence"] = df["peptide"].astype(str).isin(random_set)
    df.loc[df["is_random_sequence"], "variant_source"] = "random_sequence"
    return df


def build_variants(train_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    full_train = train_df.copy()
    return {
        "train_no_synth": train_df.loc[~train_df["is_random_sequence"]].copy().reset_index(drop=True),
        "full_train": full_train,
    }


def summary_rows(variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, df in variants.items():
        rows.append(
            {
                "variant": name,
                "n_total": int(len(df)),
                "n_random_sequences": int(df["is_random_sequence"].fillna(False).sum()),
                "n_literature": int((~df["is_random_sequence"].fillna(False)).sum()),
                "n_ap_labeled": int(df["ap"].notna().sum()),
                "n_sa_labeled": int(df["is_assembled"].notna().sum()),
                "n_beta_labeled": int(df["has_beta_sheet_content"].notna().sum()),
                "n_hm_labeled": int(df["hydrophobic_moment"].notna().sum()),
                "n_charge_labeled": int(df["net_charge"].notna().sum()),
                "mean_length": float(df["length"].mean()),
                "std_length": float(df["length"].std(ddof=0)),
                "mean_hydrophobic_moment": float(df["hydrophobic_moment"].dropna().mean()) if df["hydrophobic_moment"].notna().any() else np.nan,
                "std_hydrophobic_moment": float(df["hydrophobic_moment"].dropna().std(ddof=0)) if df["hydrophobic_moment"].notna().any() else np.nan,
                "mean_net_charge": float(df["net_charge"].dropna().mean()) if df["net_charge"].notna().any() else np.nan,
                "std_net_charge": float(df["net_charge"].dropna().std(ddof=0)) if df["net_charge"].notna().any() else np.nan,
                "beta_positive_rate": float(df["has_beta_sheet_content"].dropna().mean()) if df["has_beta_sheet_content"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_dual(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_aa_frequencies(variants: dict[str, pd.DataFrame], output_dir: Path) -> None:
    names = ["train_no_synth", "full_train"]
    freqs = {
        name: aa_freqs(variants[name]["peptide"].tolist(), alphabet=AA_ALPHABET)
        for name in names
    }
    x = np.arange(len(AA_ALPHABET))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.bar(x - width / 2, freqs["train_no_synth"], width=width, color=TEAL[3], label=LABEL_DATASET_MINUS_RANDOM)
    ax.bar(x + width / 2, freqs["full_train"], width=width, color=TEAL[5], label=LABEL_DATASET)
    ax.set_xticks(x)
    ax.set_xticklabels(list(AA_ALPHABET))
    ax.set_ylabel("Residue frequency")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_dual(fig, output_dir, "dist_full_vs_no_synth_aa")


def plot_length_distribution(variants: dict[str, pd.DataFrame], output_dir: Path) -> None:
    names = ["train_no_synth", "full_train"]
    lengths = sorted(
        length
        for length in set().union(*[variants[name]["length"].dropna().astype(int).unique().tolist() for name in names])
        if length >= 5
    )
    counts = {
        name: variants[name].loc[variants[name]["length"] >= 5, "length"].value_counts().reindex(lengths, fill_value=0)
        for name in names
    }

    x = np.arange(len(lengths))
    width = 0.42
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(
        x - width / 2,
        counts["train_no_synth"].to_numpy(),
        width=width,
        edgecolor=TEAL[4],
        linewidth=1.2,
        alpha=0.95,
        label=LABEL_DATASET_MINUS_RANDOM,
        color=TEAL[3],
    )
    ax.bar(
        x + width / 2,
        counts["full_train"].to_numpy(),
        width=width,
        edgecolor=TEAL[4],
        linewidth=1.2,
        alpha=0.95,
        label=LABEL_DATASET,
        color=TEAL[5],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in lengths])
    ax.set_xlabel("Peptide length")
    ax.set_ylabel("Count")
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
    save_dual(fig, output_dir, "dist_full_vs_no_synth_length")


def plot_proxy_marginals(variants: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    hm_all = pd.concat([df["hydrophobic_moment"].dropna() for df in variants.values()], ignore_index=True)
    hm_bins = np.linspace(float(hm_all.min()), float(hm_all.max()), 25) if len(hm_all) else np.linspace(0.0, 1.0, 25)

    nc_all = pd.concat([df["net_charge"].dropna() for df in variants.values()], ignore_index=True)
    nc_bins = np.linspace(float(nc_all.min()), float(nc_all.max()), 25) if len(nc_all) else np.linspace(-5.0, 5.0, 25)

    plot_specs = [
        ("hydrophobic_moment", hm_bins, axes[0], "Hydrophobic moment"),
        ("net_charge", nc_bins, axes[1], "Net charge"),
    ]
    for column, bins, ax, title in plot_specs:
        ax.hist(
            variants["full_train"][column].dropna(),
            bins=bins,
            alpha=0.95,
            color=TEAL[5],
            edgecolor=TEAL[4],
            linewidth=1.0,
            label=LABEL_DATASET,
            zorder=1,
        )
        ax.hist(
            variants["train_no_synth"][column].dropna(),
            bins=bins,
            alpha=0.95,
            color=TEAL[3],
            edgecolor=TEAL[4],
            linewidth=1.0,
            label=LABEL_DATASET_MINUS_RANDOM_WRAPPED,
            zorder=2,
        )
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Count")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            fontsize=9.5,
        )

    beta_counts = pd.DataFrame(
        {
            "variant": [
                LABEL_DATASET_MINUS_RANDOM_WRAPPED,
                LABEL_DATASET_MINUS_RANDOM_WRAPPED,
                LABEL_DATASET,
                LABEL_DATASET,
            ],
            "value": [0, 1, 0, 1],
            "count": [
                int((variants["train_no_synth"]["has_beta_sheet_content"] == 0).sum()),
                int((variants["train_no_synth"]["has_beta_sheet_content"] == 1).sum()),
                int((variants["full_train"]["has_beta_sheet_content"] == 0).sum()),
                int((variants["full_train"]["has_beta_sheet_content"] == 1).sum()),
            ],
        }
    )
    sns.barplot(
        data=beta_counts,
        x="value",
        y="count",
        hue="variant",
        palette=[TEAL[3], TEAL[5]],
        ax=axes[2],
    )
    axes[2].set_title(r"Has $\beta$-strand")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Count")
    axes[2].set_yscale("log")
    for patch in axes[2].patches:
        patch.set_edgecolor(TEAL[4])
        patch.set_linewidth(1.2)
        patch.set_alpha(0.95)
    for spine in ("top", "right"):
        axes[2].spines[spine].set_visible(False)
    axes[2].legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=9.5,
    )

    plt.tight_layout()
    save_dual(fig, output_dir, "dist_full_vs_no_synth_proxies")


def plot_label_distributions(variants: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ap_all = pd.concat([df["ap"].dropna() for df in variants.values()], ignore_index=True)
    ap_bins = np.linspace(float(ap_all.min()), float(ap_all.max()), 25) if len(ap_all) else np.linspace(0.0, 1.0, 25)
    axes[0].hist(
        variants["train_no_synth"]["ap"].dropna(),
        bins=ap_bins,
        alpha=0.60,
        color=TEAL[2],
        label=f"Train only (n={variants['train_no_synth']['ap'].notna().sum()})",
    )
    axes[0].hist(
        variants["full_train"]["ap"].dropna(),
        bins=ap_bins,
        alpha=0.50,
        color=TEAL[4],
        label=f"Train + synth (n={variants['full_train']['ap'].notna().sum()})",
    )
    axes[0].set_title("Aggregation propensity")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False, fontsize=10)
    sns.despine(ax=axes[0])

    sa_counts = pd.DataFrame(
        {
            "variant": ["train_no_synth", "train_no_synth", "full_train", "full_train"],
            "value": [0, 1, 0, 1],
            "count": [
                int((variants["train_no_synth"]["is_assembled"] == 0).sum()),
                int((variants["train_no_synth"]["is_assembled"] == 1).sum()),
                int((variants["full_train"]["is_assembled"] == 0).sum()),
                int((variants["full_train"]["is_assembled"] == 1).sum()),
            ],
        }
    )
    sns.barplot(
        data=sa_counts,
        x="value",
        y="count",
        hue="variant",
        palette=[TEAL[2], TEAL[4]],
        ax=axes[1],
    )
    axes[1].set_title("SA / no-SA label")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False, title="")
    sns.despine(ax=axes[1])

    plt.tight_layout()
    save_dual(fig, output_dir, "dist_full_vs_no_synth_labels")


def main() -> int:
    args = parse_args()
    set_paper_style()

    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    splits_dir = Path(args.splits_dir).expanduser().resolve() if args.splits_dir else None
    train_df = load_train_split(Path(args.data_csv).expanduser().resolve(), splits_dir)
    train_df = annotate_random_sequences(train_df, Path(args.random_sequences_csv).expanduser().resolve())
    variants = build_variants(train_df)

    plot_aa_frequencies(variants, output_dir)
    plot_length_distribution(variants, output_dir)
    plot_proxy_marginals(variants, output_dir)
    plot_label_distributions(variants, output_dir)

    summary_df = summary_rows(variants)
    summary_path = output_dir / "train_variant_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "git_commit": get_git_commit(config_path.parents[2]),
        "seed": int(config.get("seed", 42)),
        "config_path": str(config_path),
        "model_checkpoint_path": None,
        "splits_dir": str(splits_dir) if splits_dir is not None else "data/splits",
        "random_sequences_csv": str(Path(args.random_sequences_csv).expanduser().resolve()),
        "output_dir": str(output_dir),
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print("Wrote", summary_path)
    print("Wrote", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
