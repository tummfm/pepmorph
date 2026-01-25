#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA_PROCESSED, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test splits.")
    parser.add_argument("--data-csv", type=str, default=str(DATA_PROCESSED / "merged_all.csv"))
    parser.add_argument("--output-dir", type=str, default=str(Path(DATA_PROCESSED).parents[0] / "splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--stratify-col", type=str, default="length")
    return parser.parse_args()


def write_indices(path: Path, indices: list[int]) -> None:
    path.write_text("\n".join(str(i) for i in indices) + "\n")


def main() -> int:
    args = parse_args()
    data_csv = Path(args.data_csv).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())

    df = pd.read_csv(data_csv, keep_default_na=False, na_values=[""])
    if args.stratify_col not in df.columns:
        raise ValueError(f"Missing stratify column: {args.stratify_col}")

    train_val_idx, test_idx = train_test_split(
        df.index,
        test_size=args.test_size,
        stratify=df[args.stratify_col],
        random_state=args.seed,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=args.val_size,
        stratify=df.loc[train_val_idx, args.stratify_col],
        random_state=args.seed,
    )

    train_idx = sorted(int(i) for i in train_idx)
    val_idx = sorted(int(i) for i in val_idx)
    test_idx = sorted(int(i) for i in test_idx)

    write_indices(output_dir / "train_idx.txt", train_idx)
    write_indices(output_dir / "val_idx.txt", val_idx)
    write_indices(output_dir / "test_idx.txt", test_idx)

    meta = {
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "stratify_col": args.stratify_col,
        "n_total": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    (output_dir / "splits_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("Wrote", output_dir / "train_idx.txt")
    print("Wrote", output_dir / "val_idx.txt")
    print("Wrote", output_dir / "test_idx.txt")
    print("Wrote", output_dir / "splits_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
