#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from common import DATA_PROCESSED, DEFAULT_CVAE_CHECKPOINT, FEATURES, SPLITS_DIR, VALIDATION_ARTIFACTS
from cvae.models import CVAESimpleEnc
from cvae.utils import ALPHABET, CONDITION_LENGTH, MAX_FASTA_LENGTH, MAX_SEQ_LENGTH, PAD_TOKEN_ID, set_seed
from cvae_evaluation import even_counts_across_lengths, sample_from_prior_ar


DEFAULT_OUTPUT_DIR = VALIDATION_ARTIFACTS / "heldout_queries"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "masked_cvae" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export held-out test-set conditioning queries and generated samples.")
    parser.add_argument("--data-csv", type=str, default=str(DATA_PROCESSED / "merged_all.csv"))
    parser.add_argument("--raw-data-csv", type=str, default=str(DATA_PROCESSED / "merged_all_no_norm.csv"))
    parser.add_argument("--splits-dir", type=str, default=str(SPLITS_DIR))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CVAE_CHECKPOINT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-query", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument(
        "--source-selection",
        choices=["complete_case", "observed"],
        default="complete_case",
        help="`complete_case` enforces one query for each k=1..6; `observed` uses all test rows and skips infeasible k.",
    )
    parser.add_argument(
        "--max-source-peptides",
        type=int,
        default=0,
        help="Maximum number of source test peptides. Use 0 to keep all eligible rows.",
    )
    parser.add_argument(
        "--max-total-generated",
        type=int,
        default=15000,
        help="Maximum number of generated sequences to export. Use 0 to keep all possible query/sample combinations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
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


def _read_indices(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def load_test_frames(norm_path: Path, raw_path: Path, splits_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    norm_df = pd.read_csv(norm_path, keep_default_na=False, na_values=[""])
    raw_df = pd.read_csv(raw_path, keep_default_na=False, na_values=[""])

    if len(norm_df) != len(raw_df):
        raise ValueError("Normalized and raw processed tables must have the same number of rows.")
    if not norm_df["peptide"].equals(raw_df["peptide"]):
        raise ValueError("Normalized and raw processed tables are not row-aligned by peptide.")

    raw_df = raw_df.copy()
    raw_df["length"] = raw_df["peptide"].astype(str).str.len()

    test_idx = _read_indices(splits_dir / "test_idx.txt")
    test_norm = norm_df.iloc[test_idx].copy()
    test_raw = raw_df.iloc[test_idx].copy()
    test_norm["source_row_index"] = test_idx
    test_raw["source_row_index"] = test_idx
    return test_norm.reset_index(drop=True), test_raw.reset_index(drop=True)


def select_source_rows(df: pd.DataFrame, mode: str, max_source_peptides: int, seed: int) -> pd.DataFrame:
    observed_counts = df[FEATURES].notna().sum(axis=1)
    if mode == "complete_case":
        selected = df.loc[observed_counts == len(FEATURES)].copy()
    else:
        selected = df.loc[observed_counts >= 1].copy()

    selected = selected.loc[selected["length"].between(5, MAX_FASTA_LENGTH)].copy()
    if max_source_peptides <= 0 or len(selected) <= max_source_peptides:
        return selected.sort_values(["length", "source_row_index"]).reset_index(drop=True)

    return sample_balanced_by_length(selected, total=max_source_peptides, seed=seed)


def sample_balanced_by_length(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lengths = sorted(df["length"].dropna().astype(int).unique().tolist())
    target_counts = even_counts_across_lengths(lengths, total)

    chosen: list[int] = []
    for length in lengths:
        pool = df.index[df["length"] == length].to_numpy()
        take = min(len(pool), target_counts.get(length, 0))
        if take:
            chosen.extend(rng.choice(pool, size=take, replace=False).tolist())

    if len(chosen) < total:
        remaining = df.index.difference(chosen).to_numpy()
        need = min(len(remaining), total - len(chosen))
        if need:
            chosen.extend(rng.choice(remaining, size=need, replace=False).tolist())

    return df.loc[sorted(chosen)].sort_values(["length", "source_row_index"]).reset_index(drop=True)


def build_query_plan(
    selected: pd.DataFrame,
    k_min: int,
    k_max: int,
    n_per_query: int,
    max_total_generated: int,
    seed: int,
) -> pd.DataFrame:
    k_values = list(range(k_min, k_max + 1))
    if not k_values:
        raise ValueError("No valid k values requested.")
    if n_per_query < 1:
        raise ValueError("n_per_query must be at least 1.")

    all_queries = pd.DataFrame(
        [(int(row.source_row_index), int(k)) for row in selected.itertuples(index=False) for k in k_values],
        columns=["source_row_index", "k"],
    )
    if max_total_generated <= 0:
        return all_queries

    query_budget = max_total_generated // n_per_query
    if query_budget < 1:
        raise ValueError("max_total_generated is smaller than n_per_query; no queries can be exported.")
    if query_budget >= len(all_queries):
        return all_queries

    per_k_targets = even_counts_across_lengths(k_values, query_budget)
    query_chunks: list[pd.DataFrame] = []
    for k in k_values:
        target = per_k_targets.get(k, 0)
        if target <= 0:
            continue
        sampled = sample_balanced_by_length(selected, total=min(target, len(selected)), seed=seed + 1000 * k)
        query_chunks.append(
            sampled.loc[:, ["source_row_index"]].assign(k=int(k))
        )

    query_df = pd.concat(query_chunks, ignore_index=True)
    query_df = query_df.sort_values(["source_row_index", "k"]).reset_index(drop=True)
    if len(query_df) > query_budget:
        query_df = query_df.iloc[:query_budget].copy()
    return query_df


def build_model(checkpoint: Path, config: dict, device: torch.device) -> torch.nn.Module:
    model = CVAESimpleEnc(
        encoder_hidden_dim=int(config.get("encoder_hidden_dim", 256)),
        num_encoder_layers=int(config.get("num_encoder_layers", 2)),
        vocab_size=len(ALPHABET),
        latent_dim=int(config.get("latent_dim", 24)),
        cond_dim=CONDITION_LENGTH,
        max_seq_length=MAX_SEQ_LENGTH,
        decoder_hidden_dim=int(config.get("decoder_hidden_dim", 256)),
        num_decoder_layers=int(config.get("num_decoder_layers", 2)),
        nhead=int(config.get("nhead", 8)),
        dropout=float(config.get("dropout", 0.1)),
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    return model


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_condition_and_mask(row: pd.Series, used_features: Iterable[str], max_fasta_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    condition = torch.zeros(len(FEATURES), dtype=torch.float)
    mask = torch.zeros(len(FEATURES), dtype=torch.float)
    used = set(used_features)
    for idx, feature in enumerate(FEATURES):
        value = row.get(feature, np.nan)
        if feature not in used or pd.isna(value):
            continue
        condition[idx] = float(value)
        if feature == "length":
            condition[idx] = float(value) / max_fasta_length
        mask[idx] = 1.0
    return condition, mask


def json_value(value):
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def row_to_dict(row: pd.Series) -> dict[str, object]:
    return {feature: json_value(row.get(feature, np.nan)) for feature in FEATURES}


def row_to_model_input_dict(row: pd.Series) -> dict[str, object]:
    out = row_to_dict(row)
    if out["length"] is not None:
        out["length"] = float(out["length"]) / MAX_FASTA_LENGTH
    return out


def derive_used_features(source_row: pd.Series, k: int, mode: str, seed: int) -> list[str]:
    if mode == "complete_case":
        available = list(FEATURES)
    else:
        available = [feature for feature in FEATURES if pd.notna(source_row.get(feature, np.nan))]
    if not available:
        return []
    k_eff = min(k, len(available))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.array(available, dtype=object), size=k_eff, replace=False).tolist())


def build_generated_records(samples: list[tuple[str, int]]) -> list[dict[str, object]]:
    counts = Counter(seq for seq, _ in samples)
    seen = Counter()
    generated = []
    for sample_idx, (sequence, seq_length) in enumerate(samples, start=1):
        seen[sequence] += 1
        generated.append(
            {
                "sample_index": sample_idx,
                "sequence": sequence,
                "length": int(seq_length),
                "duplicate_within_query": counts[sequence] > 1,
                "occurrence_index": int(seen[sequence]),
            }
        )
    return generated


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    config_path = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    device = resolve_device(args.device)

    test_norm, test_raw = load_test_frames(
        Path(args.data_csv).expanduser().resolve(),
        Path(args.raw_data_csv).expanduser().resolve(),
        Path(args.splits_dir).expanduser().resolve(),
    )
    selected_norm = select_source_rows(test_norm, args.source_selection, args.max_source_peptides, args.seed)
    selected_raw = test_raw.set_index("source_row_index").loc[selected_norm["source_row_index"]].reset_index()
    query_plan = build_query_plan(
        selected_norm,
        k_min=args.k_min,
        k_max=args.k_max,
        n_per_query=args.n_per_query,
        max_total_generated=args.max_total_generated,
        seed=args.seed,
    )

    if selected_norm.empty:
        raise ValueError("No eligible held-out test peptides matched the selection criteria.")
    if query_plan.empty:
        raise ValueError("No held-out queries matched the requested generation budget.")

    model = build_model(checkpoint, config, device)

    jsonl_path = output_dir / "heldout_queries_generation.jsonl"
    index_path = output_dir / "heldout_queries_index.csv"
    pepfold_path = output_dir / "heldout_pepfold_inputs.csv"
    metadata_path = output_dir / "run_metadata.json"

    git_commit = get_git_commit(config_path.parents[3])
    run_metadata = {
        "git_commit": git_commit,
        "seed": args.seed,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint),
        "model_config": {
            "latent_dim": int(config.get("latent_dim", 24)),
            "encoder_hidden_dim": int(config.get("encoder_hidden_dim", 256)),
            "decoder_hidden_dim": int(config.get("decoder_hidden_dim", 256)),
            "num_encoder_layers": int(config.get("num_encoder_layers", 2)),
            "num_decoder_layers": int(config.get("num_decoder_layers", 2)),
            "nhead": int(config.get("nhead", 8)),
            "dropout": float(config.get("dropout", 0.1)),
        },
        "device": str(device),
        "splits_dir": str(Path(args.splits_dir).expanduser().resolve()),
        "source_selection": args.source_selection,
        "max_source_peptides": args.max_source_peptides,
        "n_source_peptides_selected": int(len(selected_norm)),
        "n_unique_source_peptides_used": int(query_plan["source_row_index"].nunique()),
        "n_queries_exported": int(len(query_plan)),
        "n_per_query": args.n_per_query,
        "max_total_generated": args.max_total_generated,
        "n_generated_total": int(len(query_plan) * args.n_per_query),
        "temperature": args.temperature,
        "top_p": None,
        "beam_size": None,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "max_fasta_length": MAX_FASTA_LENGTH,
        "features": FEATURES,
    }
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n")

    line_no = 0
    with open(jsonl_path, "w") as jsonl_handle, open(index_path, "w", newline="") as index_handle, open(
        pepfold_path, "w", newline=""
    ) as pepfold_handle:
        index_writer = csv.DictWriter(
            index_handle,
            fieldnames=[
                "query_id",
                "jsonl_path",
                "jsonl_line",
                "source_test_row_index",
                "source_test_sequence",
                "source_length",
                "k",
                "k_used",
                "conditioned_descriptor_names",
                "n_generated",
            ],
        )
        pepfold_writer = csv.DictWriter(pepfold_handle, fieldnames=["query_id", "generated_sequence"])
        index_writer.writeheader()
        pepfold_writer.writeheader()

        selected_norm_by_index = selected_norm.set_index("source_row_index")
        selected_raw_by_index = selected_raw.set_index("source_row_index")

        for query in query_plan.to_dict("records"):
            source_index = int(query["source_row_index"])
            k = int(query["k"])
            source_norm = selected_norm_by_index.loc[source_index]
            source_raw = selected_raw_by_index.loc[source_index]
            norm_row = pd.Series(source_norm)
            raw_row = pd.Series(source_raw)
            observed_mask = {
                feature: int(pd.notna(norm_row.get(feature, np.nan)))
                for feature in FEATURES
            }
            observed_features = [feature for feature, flag in observed_mask.items() if flag == 1]

            used_features = derive_used_features(
                norm_row,
                k=k,
                mode=args.source_selection,
                seed=args.seed + source_index * 17 + k,
            )
            if not used_features:
                continue

            condition, query_mask = make_condition_and_mask(norm_row, used_features, MAX_FASTA_LENGTH)
            generated_samples = sample_from_prior_ar(
                model,
                condition.to(device),
                query_mask.to(device),
                temperature=args.temperature,
                n=args.n_per_query,
            )
            generated_records = build_generated_records(generated_samples)

            query_id = f"heldout_{source_index}_k{k}"
            payload = {
                "query_id": query_id,
                "source_test_row_index": source_index,
                "source_test_sequence": norm_row["peptide"],
                "source_length": int(norm_row["length"]),
                "k": int(k),
                "k_used": int(len(used_features)),
                "source_observed_mask": observed_mask,
                "source_observed_feature_names": observed_features,
                "query_mask": {
                    feature: int(feature in used_features)
                    for feature in FEATURES
                },
                "conditioned_descriptor_names": used_features,
                "c_raw": row_to_dict(raw_row),
                "c_normalized": row_to_dict(norm_row),
                "c_model_input": row_to_model_input_dict(norm_row),
                "generation_hyperparams": {
                    "n_per_query": args.n_per_query,
                    "temperature": args.temperature,
                    "top_p": None,
                    "beam_size": None,
                },
                "generated_sequences": generated_records,
            }

            jsonl_handle.write(json.dumps(payload) + "\n")
            line_no += 1

            index_writer.writerow(
                {
                    "query_id": query_id,
                    "jsonl_path": jsonl_path.name,
                    "jsonl_line": line_no,
                    "source_test_row_index": source_index,
                    "source_test_sequence": norm_row["peptide"],
                    "source_length": int(norm_row["length"]),
                    "k": int(k),
                    "k_used": int(len(used_features)),
                    "conditioned_descriptor_names": ",".join(used_features),
                    "n_generated": len(generated_records),
                }
            )

            for generated in generated_records:
                pepfold_writer.writerow(
                    {
                        "query_id": query_id,
                        "generated_sequence": generated["sequence"],
                    }
                )

    print("Wrote", jsonl_path)
    print("Wrote", index_path)
    print("Wrote", pepfold_path)
    print("Wrote", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
