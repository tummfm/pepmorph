#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import GEN_PEPTIDES_DIR
from utils import enumerate_conditions, read_sequences_txt


FIBER_RANGES = {
    "length": (7, 11, 1),
    "is_assembled": (1, 2, 1),
    "has_beta_sheet_content": (1, 2, 1),
    "net_charge": (0.4, 0.6, 0.05),
}

SPHERE_RANGES = {
    "length": (5, 8, 1),
    "is_assembled": (1, 2, 1),
    "hydrophobic_moment": (0.6, 1.05, 0.1),
    "net_charge": (0.4, 0.6, 0.05),
}


def build_conditions_df(peptides: list[str], ranges: dict, samples_per_cond: int) -> pd.DataFrame:
    conds = enumerate_conditions(ranges, n_samples=samples_per_cond)
    if len(conds) < len(peptides):
        repeats = int(np.ceil(len(peptides) / len(conds)))
        conds = (conds * repeats)[: len(peptides)]
    else:
        conds = conds[: len(peptides)]
    df = pd.DataFrame({
        "peptide": peptides,
        "index": range(len(peptides)),
        **{k: [c[k] for c in conds] for k in conds[0].keys()},
    })
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate conditioning grids for generated peptides.")
    parser.add_argument("--base-dir", type=str, default=str(GEN_PEPTIDES_DIR))
    parser.add_argument("--fibers-init", type=str, default="gen_peptides/generated_fibers_init.txt")
    parser.add_argument("--spheres-init", type=str, default="gen_peptides/generated_spheres_init.txt")
    parser.add_argument("--output-fibers", type=str, default="gen_peptides/fibers_with_conditions.csv")
    parser.add_argument("--output-spheres", type=str, default="gen_peptides/spheres_with_conditions.csv")
    parser.add_argument("--fibers-samples-per-condition", type=int, default=300)
    parser.add_argument("--spheres-samples-per-condition", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()

    fibers_peptides = read_sequences_txt(base_dir / args.fibers_init)
    spheres_peptides = read_sequences_txt(base_dir / args.spheres_init)

    fibers_df = build_conditions_df(
        fibers_peptides, FIBER_RANGES, samples_per_cond=args.fibers_samples_per_condition
    )
    spheres_df = build_conditions_df(
        spheres_peptides, SPHERE_RANGES, samples_per_cond=args.spheres_samples_per_condition
    )

    fibers_out = base_dir / args.output_fibers
    spheres_out = base_dir / args.output_spheres

    fibers_df.to_csv(fibers_out, index=False)
    spheres_df.to_csv(spheres_out, index=False)

    print("Wrote", fibers_out)
    print("Wrote", spheres_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
