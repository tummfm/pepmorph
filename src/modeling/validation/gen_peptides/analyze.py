#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build negative-control structures from filtering stages.")
    repo_root = Path(__file__).resolve().parents[4]
    parser.add_argument("--base-dir", type=str, default=str(repo_root / "artifacts" / "validation" / "gen_peptides"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-count", type=int, default=15)
    parser.add_argument("--output-csv", type=str, default="negative_control_structures_new.csv")
    parser.add_argument("--output-fasta", type=str, default="negative_control_structures_new.fasta")
    return parser.parse_args()


def sample_negatives(init_path: Path, filtered_path: Path, valid_path: Path, prefix: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    init_df = pd.read_csv(init_path, header=None, names=[f"{prefix}_id"]).drop_duplicates()
    filtered_df = pd.read_csv(filtered_path, header=None, names=[f"{prefix}_id", f"{prefix}_id_3", "ap", "agg_prob"])
    valid_df = pd.read_csv(valid_path, header=0, names=[f"{prefix}_id"])

    not_passing_ap = init_df[~init_df[f"{prefix}_id"].isin(filtered_df[f"{prefix}_id"])]
    not_passing_valid = filtered_df[~filtered_df[f"{prefix}_id"].isin(valid_df[f"{prefix}_id"])]

    neg_ap = not_passing_ap.sample(n=min(15, len(not_passing_ap)), random_state=seed)
    neg_valid = not_passing_valid.sample(n=min(15, len(not_passing_valid)), random_state=seed)

    neg_ap = neg_ap.rename(columns={f"{prefix}_id": "structure_id", f"{prefix}_id_3": "structure_id_3"}).assign(
        structure_type=prefix
    )
    neg_valid = neg_valid.rename(columns={f"{prefix}_id": "structure_id", f"{prefix}_id_3": "structure_id_3"}).assign(
        structure_type=prefix
    )

    return neg_ap, neg_valid


def generate_random_peptide(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list(ALPHABET), size=length))


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    rng = np.random.default_rng(args.seed)

    fibers_neg_ap, fibers_neg_valid = sample_negatives(
        base_dir / "generated_fibers_init.txt",
        base_dir / "filtered_ap_peptides_fiber_final.txt",
        base_dir / "valid_fibers.csv",
        "fiber",
        args.seed,
    )
    spheres_neg_ap, spheres_neg_valid = sample_negatives(
        base_dir / "generated_spheres_init.txt",
        base_dir / "filtered_ap_peptides_spheres.txt",
        base_dir / "valid_spheres.csv",
        "sphere",
        args.seed,
    )

    neg_control = pd.concat([fibers_neg_ap, spheres_neg_ap, fibers_neg_valid, spheres_neg_valid])
    neg_control.reset_index(drop=True, inplace=True)

    neg_control["sequence_length"] = neg_control["structure_id"].astype(str).apply(len)
    short_ap = neg_control[(neg_control["sequence_length"] < 5) & (neg_control["ap"].isna())]
    short_valid = neg_control[(neg_control["sequence_length"] < 5) & (neg_control["ap"].notna())]

    spheres_not_passing_ap = pd.read_csv(base_dir / "generated_spheres_init.txt", header=None, names=["sphere_id"])
    spheres_filtered = pd.read_csv(
        base_dir / "filtered_ap_peptides_spheres.txt", header=None, names=["sphere_id", "sphere_id_3", "ap", "agg_prob"]
    )
    spheres_not_passing_ap = spheres_not_passing_ap[~spheres_not_passing_ap["sphere_id"].isin(spheres_filtered["sphere_id"])]
    spheres_not_passing_valid = spheres_filtered[~spheres_filtered["sphere_id"].isin(pd.read_csv(base_dir / "valid_spheres.csv", header=0, names=["sphere_id"])["sphere_id"])]

    ap_sub = spheres_not_passing_ap[spheres_not_passing_ap["sphere_id"].apply(len) >= 5].sample(
        n=len(short_ap), random_state=args.seed
    )
    valid_sub = spheres_not_passing_valid[spheres_not_passing_valid["sphere_id"].apply(len) >= 5].sample(
        n=len(short_valid), random_state=args.seed
    )

    neg_control_final = neg_control[neg_control["sequence_length"] >= 5].copy()

    ap_sub = ap_sub.rename(columns={"sphere_id": "structure_id", "sphere_id_3": "structure_id_3"}).assign(
        structure_type="sphere", ap=np.nan, agg_prob=np.nan, sequence_length=ap_sub["sphere_id"].apply(len)
    )
    valid_sub = valid_sub.rename(columns={"sphere_id": "structure_id", "sphere_id_3": "structure_id_3"}).assign(
        structure_type="sphere", sequence_length=valid_sub["sphere_id"].apply(len)
    )

    neg_control_final = pd.concat([neg_control_final, ap_sub, valid_sub])
    neg_control_final["structure_length"] = neg_control_final["structure_id"].apply(len)

    current_count = len(neg_control_final)
    target_count = args.target_count
    all_lengths = list(range(5, 11))

    while current_count < target_count:
        length = rng.choice(all_lengths)
        random_peptide = generate_random_peptide(int(length), rng)
        neg_control_final = pd.concat(
            [
                neg_control_final,
                pd.DataFrame(
                    {
                        "structure_id": [random_peptide],
                        "structure_id_3": [None],
                        "structure_type": ["random"],
                        "structure_length": [length],
                    }
                ),
            ],
            ignore_index=True,
        )
        current_count += 1

    neg_control_final.reset_index(drop=True, inplace=True)

    csv_path = base_dir / args.output_csv
    fasta_path = base_dir / args.output_fasta
    neg_control_final.to_csv(csv_path, index=False)

    with open(fasta_path, "w") as fasta_file:
        for idx, row in neg_control_final.iterrows():
            fasta_file.write(f">{row['structure_type']}_{idx}\n")
            fasta_file.write(f"{row['structure_id']}\n")

    print("Wrote", csv_path)
    print("Wrote", fasta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
