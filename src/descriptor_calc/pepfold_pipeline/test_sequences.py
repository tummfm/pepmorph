#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick checks for peptide descriptor outputs")
    parser.add_argument("--metrics", type=Path, default=Path("peptide_metrics.csv"))
    parser.add_argument("--structures-dir", type=Path, default=Path("peptide_structures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics
    if not metrics_path.is_absolute():
        metrics_path = Path(__file__).resolve().parent / metrics_path

    metrics = pd.read_csv(metrics_path)
    print(metrics[(abs(metrics["net_charge"]) < 2) & (metrics["hydrophobic_moment"] > 1.2)])

    structures_dir = args.structures_dir
    if not structures_dir.is_absolute():
        structures_dir = Path(__file__).resolve().parent / structures_dir

    peptide_folders = [p for p in structures_dir.glob("peptide_*") if p.is_dir()]
    print(f"Found {len(peptide_folders)} peptide folders.")

    count = 0
    for folder_path in peptide_folders:
        peptide_id = folder_path.name
        seq_files = list(folder_path.glob("*.seq"))
        if not seq_files:
            continue
        pdb_pattern = folder_path / f"{peptide_id}-bestmodel.pdb"
        if not pdb_pattern.exists():
            continue
        count += 1

    print(f"There are {len(peptide_folders) - count} yet to process peptide folders.")


if __name__ == "__main__":
    main()
