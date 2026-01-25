#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CG_DIR = Path(__file__).resolve().parents[1]
if str(CG_DIR) not in sys.path:
    sys.path.insert(0, str(CG_DIR))

from common import PEPMORPH_MAIN_ARTIFACTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize control MD simulations with PepMorph labels.")
    parser.add_argument("--base-dir", type=str, default=str(PEPMORPH_MAIN_ARTIFACTS_DIR))
    parser.add_argument("--input-dir", type=str, default=str(PEPMORPH_MAIN_ARTIFACTS_DIR / "outputs"))
    parser.add_argument("--output-dir", type=str, default=str(PEPMORPH_MAIN_ARTIFACTS_DIR / "outputs"))
    parser.add_argument("--ap-threshold", type=float, default=1.8)
    return parser.parse_args()


def classify_structure(row: pd.Series) -> str:
    if row["structure_type"] == "random":
        return "random"
    if row["structure_type"] == "unsupervised":
        return "unsupervised"
    if row["structure_type"] == "fiber":
        return "fiber_no_ap" if pd.isna(row["ap"]) else "fiber_no_desc"
    if row["structure_type"] == "sphere":
        return "sphere_no_ap" if pd.isna(row["ap"]) else "sphere_no_desc"
    return "unknown"


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(input_dir / "rmoi_and_ap_summary_total.csv")
    by_run = pd.read_csv(input_dir / "rmoi_and_ap_by_run_total.csv")

    runs_pepmorph = pd.read_csv(input_dir / "rmoi_and_ap_summary_pepmorph.csv")
    by_run_pepmorph = pd.read_csv(input_dir / "rmoi_and_ap_by_run_pepmorph.csv")

    runs_pepmorph = runs_pepmorph.dropna(axis=1, how="all")
    by_run_pepmorph = by_run_pepmorph.dropna(axis=1, how="all")

    runs_pepmorph = runs_pepmorph.rename(columns={"morphology": "class"})
    by_run_pepmorph = by_run_pepmorph.rename(columns={"morphology": "class"})

    runs_pepmorph["class"] = runs_pepmorph["class"].replace(
        {"fibers": "fiber_pepmorph", "spheres": "sphere_pepmorph"}
    )
    by_run_pepmorph["class"] = by_run_pepmorph["class"].replace(
        {"fibers": "fiber_pepmorph", "spheres": "sphere_pepmorph"}
    )

    original = pd.read_csv(input_dir / "negative_control_structures.csv")
    original = original.rename(columns={"structure_id": "peptide"})
    original["class"] = original.apply(classify_structure, axis=1)

    merged = pd.merge(original[["peptide", "class"]], runs, on="peptide", how="left")
    merged = pd.concat([merged, runs_pepmorph], ignore_index=True)

    merged_by_run = pd.merge(original[["peptide", "class"]], by_run, on="peptide", how="left")
    merged_by_run = pd.concat([merged_by_run, by_run_pepmorph], ignore_index=True)

    summary_path = output_dir / "merged_summary.csv"
    merged.to_csv(summary_path, index=False)

    by_run_path = output_dir / "merged_by_run.csv"
    merged_by_run.to_csv(by_run_path, index=False)

    threshold = args.ap_threshold
    merged_random = merged[merged["class"] == "random"]
    merged_unsupervised = merged[merged["class"] == "unsupervised"]

    print("Merged summary saved to", summary_path)
    print("Merged by-run saved to", by_run_path)
    print(f"Random peptides >= {threshold}: {len(merged_random[merged_random['aggregation_propensity_mean'] >= threshold])}")
    print(
        f"Unsupervised peptides >= {threshold}: {len(merged_unsupervised[merged_unsupervised['aggregation_propensity_mean'] >= threshold])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
