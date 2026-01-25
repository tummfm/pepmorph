#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import VALIDATION_ARTIFACTS

DEFAULT_MIN_MAX = {
    "has_beta_sheet_content": (0.0, 1.0),
    "hydrophobic_moment": (0.0, 1.998),
    "net_charge": (-6.0, 6.0),
    "ap": (0.959986, 2.89703),
}

DEFAULT_SPHERE_SAMPLE = {
    "hydrophobic_moment": [0.6, 1.00],
    "net_charge": [0.4, 0.6],
}

DEFAULT_FIBER_SAMPLE = {
    "has_beta_sheet_content": [0.1, 1.0],
    "net_charge": [0.4, 0.6],
}


def denormalize_ranges(ranges: dict[str, list[float]], min_max: dict[str, tuple[float, float]]):
    out = {}
    for key, (lo, hi) in ranges.items():
        if key not in min_max:
            raise KeyError(f"Missing min/max for {key}")
        mn, mx = min_max[key]
        out[key] = [lo * (mx - mn) + mn, hi * (mx - mn) + mn]
    return out


def load_metrics(paths: list[Path], beta_threshold: float) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "sequence" not in df.columns:
            raise ValueError(f"{path} must contain a 'sequence' column")
        df = df.copy()
        if "beta_sheet_fraction" in df.columns:
            df["has_beta_sheet_content"] = (df["beta_sheet_fraction"] > beta_threshold).astype(int)
        if "peptide_id" in df.columns:
            df = df.drop(columns=["peptide_id"])
        df = df.drop_duplicates(subset=["sequence"])
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["sequence"])
    return merged


def load_ap_files(paths: list[Path], min_max: dict[str, tuple[float, float]], ap_normalized: bool) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, header=None, names=["sequence", "sequence_3", "ap", "assembly"])
        if ap_normalized:
            mn, mx = min_max["ap"]
            df["ap"] = mn + df["ap"].astype(float) * (mx - mn)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged["sequence"] = merged["sequence"].astype(str)
    return merged


def filter_by_ranges(df: pd.DataFrame, ranges: dict[str, list[float]]) -> pd.DataFrame:
    filtered = df.copy()
    for key, (lo, hi) in ranges.items():
        filtered = filtered[(filtered[key] >= lo) & (filtered[key] <= hi)]
    return filtered


def process(
    sphere_metrics: list[Path],
    fiber_metrics: list[Path],
    sphere_ap_files: list[Path],
    fiber_ap_files: list[Path],
    output_spheres: Path,
    output_fibers: Path,
    ap_normalized: bool,
    beta_threshold: float,
    sphere_ranges: dict[str, list[float]],
    fiber_ranges: dict[str, list[float]],
):
    sphere_ranges = denormalize_ranges(sphere_ranges, DEFAULT_MIN_MAX)
    fiber_ranges = denormalize_ranges(fiber_ranges, DEFAULT_MIN_MAX)

    spheres_df = load_metrics(sphere_metrics, beta_threshold=beta_threshold)
    fibers_df = load_metrics(fiber_metrics, beta_threshold=beta_threshold)

    valid_spheres = filter_by_ranges(spheres_df, sphere_ranges)
    valid_fibers = filter_by_ranges(fibers_df, fiber_ranges)

    ap_spheres = load_ap_files(sphere_ap_files, DEFAULT_MIN_MAX, ap_normalized=ap_normalized)
    ap_fibers = load_ap_files(fiber_ap_files, DEFAULT_MIN_MAX, ap_normalized=ap_normalized)

    merged_spheres = valid_spheres.merge(ap_spheres, on="sequence", how="inner")
    merged_fibers = valid_fibers.merge(ap_fibers, on="sequence", how="inner")

    merged_spheres.to_csv(output_spheres, index=False)
    merged_fibers.to_csv(output_fibers, index=False)

    print("Wrote", output_spheres)
    print("Wrote", output_fibers)


PRESETS = {
    "targeted": dict(
        sphere_metrics=["gen_peptides/spheres_metrics.csv", "gen_peptides/spheres_metrics_extra.csv"],
        fiber_metrics=["gen_peptides/fibers_metrics.csv"],
        sphere_ap_files=["gen_peptides/filtered_ap_peptides_spheres.txt", "gen_peptides/filtered_ap_peptides_spheres_extra.txt"],
        fiber_ap_files=["gen_peptides/filtered_ap_peptides_fiber_final.txt"],
        output_spheres="gen_peptides/valid_spheres_new.csv",
        output_fibers="gen_peptides/valid_fibers_new.csv",
        ap_normalized=True,
    ),
    "unconditional": dict(
        sphere_metrics=["gen_peptides/spheres_metrics_unconditional.csv"],
        fiber_metrics=["gen_peptides/fibers_metrics_unconditional.csv"],
        sphere_ap_files=[
            "gen_peptides/filtered_ap_peptides_random_spheres.txt",
            "gen_peptides/filtered_ap_peptides_random_spheres_extra.txt",
        ],
        fiber_ap_files=["gen_peptides/filtered_ap_peptides_random_fiber.txt"],
        output_spheres="gen_peptides/valid_spheres_unconditional.csv",
        output_fibers="gen_peptides/valid_fibers_unconditional.csv",
        ap_normalized=True,
    ),
    "random": dict(
        sphere_metrics=["gen_peptides/random_metrics.csv"],
        fiber_metrics=["gen_peptides/random_metrics.csv"],
        sphere_ap_files=["gen_peptides/filtered_ap_peptides_random.txt"],
        fiber_ap_files=["gen_peptides/filtered_ap_peptides_random.txt"],
        output_spheres="gen_peptides/valid_spheres_unconditional.csv",
        output_fibers="gen_peptides/valid_fibers_unconditional.csv",
        ap_normalized=False,
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter generated peptides by descriptor windows.")
    parser.add_argument("mode", choices=sorted(PRESETS.keys()))
    parser.add_argument("--base-dir", type=str, default=str(VALIDATION_ARTIFACTS))
    parser.add_argument("--beta-threshold", type=float, default=0.1)
    parser.add_argument("--sphere-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--fiber-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--ap-normalized", action="store_true", help="Interpret AP values as normalized [0,1].")
    parser.add_argument("--output-spheres", type=str, default=None)
    parser.add_argument("--output-fibers", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir).expanduser().resolve()

    preset = PRESETS[args.mode]
    sphere_metrics = [base_dir / p for p in preset["sphere_metrics"]]
    fiber_metrics = [base_dir / p for p in preset["fiber_metrics"]]
    sphere_ap_files = [base_dir / p for p in preset["sphere_ap_files"]]
    fiber_ap_files = [base_dir / p for p in preset["fiber_ap_files"]]

    output_spheres = base_dir / (args.output_spheres or preset["output_spheres"])
    output_fibers = base_dir / (args.output_fibers or preset["output_fibers"])

    ap_normalized = args.ap_normalized or preset["ap_normalized"]

    sphere_ranges = DEFAULT_SPHERE_SAMPLE.copy()
    fiber_ranges = DEFAULT_FIBER_SAMPLE.copy()
    if args.sphere_range:
        sphere_ranges["hydrophobic_moment"] = list(args.sphere_range)
    if args.fiber_range:
        fiber_ranges["net_charge"] = list(args.fiber_range)

    process(
        sphere_metrics=sphere_metrics,
        fiber_metrics=fiber_metrics,
        sphere_ap_files=sphere_ap_files,
        fiber_ap_files=fiber_ap_files,
        output_spheres=output_spheres,
        output_fibers=output_fibers,
        ap_normalized=ap_normalized,
        beta_threshold=args.beta_threshold,
        sphere_ranges=sphere_ranges,
        fiber_ranges=fiber_ranges,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
