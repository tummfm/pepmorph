#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from common import DATA_RAW, DATA_PROCESSED, ensure_dir, set_paper_style


AMINO_ACID_DICT = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe", "G": "Gly",
    "H": "Hse", "I": "Ile", "K": "Lys", "L": "Leu", "M": "Met", "N": "Asn",
    "P": "Pro", "Q": "Gln", "R": "Arg", "S": "Ser", "T": "Thr", "V": "Val",
    "W": "Trp", "Y": "Tyr",
}
AMINO_ACID_DICT_INV = {v.upper(): k for k, v in AMINO_ACID_DICT.items()}


def convert_to_1_letter(peptide: str) -> str:
    return "".join([AMINO_ACID_DICT_INV[aa] for aa in peptide.split("-")])


def load_raw_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experimental_df = pd.read_csv(DATA_RAW / "experimental.csv", sep=";", keep_default_na=False).set_index("peptide")
    classification_df = pd.read_csv(DATA_RAW / "merged_clas.csv", keep_default_na=False).set_index("peptide")
    regression_df = pd.read_csv(DATA_RAW / "merged_aps.csv", keep_default_na=False).set_index("peptide")

    metrics_path = DATA_RAW / "peptide_metrics.csv"
    if not metrics_path.exists():
        metrics_path = (
            DATA_RAW.parent.parent
            / "src"
            / "descriptor_calc"
            / "pepfold_pipeline"
            / "peptide_metrics.csv"
        )

    metrics_df = (
        pd.read_csv(metrics_path, keep_default_na=False)
        .rename(columns={"sequence": "peptide"})
        .set_index("peptide")
    )

    metrics_no_ap_df = (
        pd.read_csv(DATA_RAW / "peptide_metrics_no_ap.csv", keep_default_na=False)
        .rename(columns={"sequence": "peptide"})
        .set_index("peptide")
    )

    beyond_tri = pd.read_csv(DATA_RAW / "beyond_tri.txt", sep=" ", header=None, names=["peptide_tri", "ap_beyond"])
    beyond_tri["peptide"] = beyond_tri["peptide_tri"].apply(convert_to_1_letter)
    beyond_tri["length"] = beyond_tri["peptide"].str.len()
    beyond_tri = beyond_tri[beyond_tri["length"] > 2].drop(columns=["peptide_tri"])

    return experimental_df, classification_df, regression_df, metrics_df, metrics_no_ap_df, beyond_tri


def merge_ap_sources(regression_df: pd.DataFrame, beyond_tri: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(regression_df, beyond_tri[["peptide", "ap_beyond"]], on="peptide", how="outer")
    merged_df["ap"] = merged_df["ap"].combine_first(merged_df["ap_beyond"])
    merged_df.drop(columns=["ap_beyond"], inplace=True)
    merged_df["length"] = merged_df["peptide"].str.len()
    return merged_df


def add_classification_label(df: pd.DataFrame, min_ap: float, max_ap: float) -> pd.DataFrame:
    df = df.copy()
    df["label"] = np.where(
        (df["ap"] >= min_ap) & (df["ap"] <= max_ap),
        np.nan,
        np.where(df["ap"] > max_ap, 1, 0),
    )
    return df


def merge_metrics(df: pd.DataFrame, metrics_df: pd.DataFrame, metrics_no_ap_df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("peptide").join(metrics_df, how="outer")
    df = df.combine_first(metrics_no_ap_df)
    return df


def add_beta_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_beta_sheet_content"] = np.where(
        df["beta_sheet_fraction"] > 0,
        1,
        np.where(df["beta_sheet_fraction"].isna(), np.nan, 0),
    )
    return df


def filter_lengths(df: pd.DataFrame, min_len: int = 3, max_len: int = 10) -> pd.DataFrame:
    df = df.copy()
    df = df[df["length"] > min_len]
    df = df[df["length"] < max_len + 1]
    return df


def normalize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[columns] = (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min())
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PepMorph raw datasets into processed CSVs")
    parser.add_argument("--min-ap", type=float, default=1.65)
    parser.add_argument("--max-ap", type=float, default=1.8)
    parser.add_argument("--output-dir", type=Path, default=DATA_PROCESSED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    set_paper_style()

    experimental_df, classification_df, regression_df, metrics_df, metrics_no_ap_df, beyond_tri = load_raw_tables()

    peptides_in_regression_not_in_classification = regression_df.index.difference(classification_df.index)
    if not peptides_in_regression_not_in_classification.empty:
        print("Peptides in regression but not in classification:")
        print(peptides_in_regression_not_in_classification)
        print("Min AP:", regression_df.loc[peptides_in_regression_not_in_classification, "ap"].min())
        print("Max AP:", regression_df.loc[peptides_in_regression_not_in_classification, "ap"].max())

    merged_df = merge_ap_sources(regression_df, beyond_tri)
    merged_df = add_classification_label(merged_df, min_ap=args.min_ap, max_ap=args.max_ap)

    merged_df = merge_metrics(merged_df, metrics_df, metrics_no_ap_df)
    merged_df = merged_df.reset_index()
    if "peptide" not in merged_df.columns:
        merged_df.rename(columns={"index": "peptide"}, inplace=True)
    merged_df["length"] = merged_df["peptide"].str.len()
    merged_df.rename(columns={"label": "is_assembled"}, inplace=True)
    merged_df = merged_df.astype({"is_assembled": np.float64, "ap": np.float64})
    merged_df = filter_lengths(merged_df, min_len=2, max_len=10)
    merged_df = add_beta_flag(merged_df)

    norm_columns = [
        "ap",
        "is_assembled",
        "hydrophobic_moment",
        "has_beta_sheet_content",
        "net_charge",
    ]

    merged_df[["peptide"] + norm_columns].to_csv(output_dir / "merged_all_no_norm.csv", index=False)

    merged_norm = normalize_columns(merged_df, norm_columns)
    merged_norm[["peptide"] + norm_columns + ["length"]].to_csv(output_dir / "merged_all.csv", index=False)

    print("Wrote:")
    print(" -", output_dir / "merged_all_no_norm.csv")
    print(" -", output_dir / "merged_all.csv")


if __name__ == "__main__":
    main()
