#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AP/RMOI across negative control categories.")
    repo_root = Path(__file__).resolve().parents[4]
    parser.add_argument("--base-dir", type=str, default=str(repo_root / "artifacts" / "validation" / "gen_peptides"))
    parser.add_argument("--neg-control", type=str, default="negative_control_structures.csv")
    parser.add_argument("--rmoi-summary", type=str, default="rmoi_and_ap_summary_total.csv")
    parser.add_argument("--threshold", type=float, default=1.8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()

    neg_ctrl_df = pd.read_csv(base_dir / args.neg_control).rename(columns={"structure_id": "peptide"})
    rmoi_df = pd.read_csv(base_dir / args.rmoi_summary)

    merged_df = pd.merge(neg_ctrl_df, rmoi_df, on=["peptide"], how="left")

    threshold = args.threshold

    random_peptides_df = merged_df[merged_df["structure_type"] == "random"]
    high_ap_random_df = random_peptides_df[random_peptides_df["aggregation_propensity_mean"] > threshold]

    sphere_fiber_pre_ap_df = merged_df[(merged_df["structure_type"].isin(["sphere", "fiber"])) & (merged_df["ap"].isna())]
    high_rmoi_sphere_fiber_df = sphere_fiber_pre_ap_df[sphere_fiber_pre_ap_df["aggregation_propensity_mean"] > threshold]

    sphere_fiber_post_ap_df = merged_df[(merged_df["structure_type"].isin(["sphere", "fiber"])) & (~merged_df["ap"].isna())]
    high_ap_sphere_fiber_df = sphere_fiber_post_ap_df[sphere_fiber_post_ap_df["aggregation_propensity_mean"] > threshold]

    unsupervised_df = rmoi_df[rmoi_df["morphology"] == "unsupervised"]
    high_ap_unsupervised_df = unsupervised_df[unsupervised_df["aggregation_propensity_mean"] > threshold]

    print(f"Random peptides > {threshold}: {len(high_ap_random_df)} / {len(random_peptides_df)}")
    print(f"Sphere/fiber pre-AP > {threshold}: {len(high_rmoi_sphere_fiber_df)} / {len(sphere_fiber_pre_ap_df)}")
    print(f"Sphere/fiber post-AP > {threshold}: {len(high_ap_sphere_fiber_df)} / {len(sphere_fiber_post_ap_df)}")
    print(f"Unsupervised > {threshold}: {len(high_ap_unsupervised_df)} / {len(unsupervised_df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
