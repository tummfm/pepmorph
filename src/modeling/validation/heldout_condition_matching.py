#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from copy import deepcopy

from common import DEFAULT_AP_CHECKPOINT, TEAL, VALIDATION_ARTIFACTS, set_paper_style
from condition_matching_report import (
    FEATURES,
    add_condition_matches_from_params,
    compute_condition_matching_effectiveness,
    compute_condition_matching_effectiveness_by_k,
    plot_conditions_success,
    plot_samples_per_hit,
)
from classifier.models import PeptidePredictor
from cvae.utils import ALPHABET, esm_model_pretrained


HELDOUT_DIR = VALIDATION_ARTIFACTS / "heldout_queries"
DEFAULT_JSONL = HELDOUT_DIR / "heldout_queries_generation.jsonl"
DEFAULT_METRICS = HELDOUT_DIR / "peptide_metrics.csv"
DEFAULT_OUTPUT_DIR = HELDOUT_DIR / "condition_matching"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-style condition-matching plots for held-out query exports.")
    parser.add_argument("--queries-jsonl", type=str, default=str(DEFAULT_JSONL))
    parser.add_argument("--metrics-csv", type=str, default=str(DEFAULT_METRICS))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_AP_CHECKPOINT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tol-pct", type=float, default=0.10)
    parser.add_argument("--sa-threshold", type=float, default=0.5)
    parser.add_argument("--beta-threshold", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_heldout_samples(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open() as handle:
        for cond_idx, line in enumerate(handle):
            payload = json.loads(line)
            params = payload.get("c_normalized") or {}
            used_features = list(payload.get("conditioned_descriptor_names", []))
            for generated in payload.get("generated_sequences", []):
                rows.append(
                    {
                        "cond_idx": int(cond_idx),
                        "query_id": payload.get("query_id"),
                        "source_test_row_index": payload.get("source_test_row_index"),
                        "source_test_sequence": payload.get("source_test_sequence"),
                        "target_length": payload.get("source_length"),
                        "sequence": generated.get("sequence"),
                        "length": generated.get("length"),
                        "params": params,
                        "used_features": used_features,
                        "ood_type": "in_dist",
                    }
                )
    if not rows:
        raise ValueError(f"No generated sequences found in {path}.")
    return pd.DataFrame(rows)


def normalize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    min_max = {
        "hydrophobic_moment": (0.000000, 1.998000),
        "net_charge": (-6.000000, 6.000000),
    }
    for col, (mn, mx) in min_max.items():
        if col not in df.columns:
            raise ValueError(f"metrics CSV must contain '{col}'.")
        df[col] = (df[col] - mn) / (mx - mn)
    return df


def load_predictor(checkpoint: Path, device: torch.device) -> PeptidePredictor:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing AP checkpoint: {checkpoint}")
    predictor = PeptidePredictor(deepcopy(esm_model_pretrained), alphabet=ALPHABET)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    predictor.load_state_dict(state)
    predictor.to(device).eval()
    return predictor


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def pretty_feature_name(feature: str) -> str:
    mapping = {
        "length": "Length",
        "is_assembled": "SA/no-SA",
        "ap": "AP",
        "has_beta_sheet_content": r"Has $\beta$-strand",
        "hydrophobic_moment": "Hydrophobic moment",
        "net_charge": "Net charge",
    }
    return mapping.get(feature, feature)


def build_target_success_summary(samples_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        match_col = f"match_{feature}"
        used_mask = samples_df["used_features"].apply(lambda used: feature in set(used))
        vals = samples_df.loc[used_mask, match_col].dropna().astype(bool)
        n = int(vals.shape[0])
        k = int(vals.sum()) if n else 0
        lo, hi = wilson_ci(k, n)
        rows.append(
            {
                "feature": feature,
                "label": pretty_feature_name(feature),
                "k_match": k,
                "n_considered": n,
                "effectiveness": (k / n) if n else np.nan,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


def plot_morph_match_rate_vs_k(by_k: pd.DataFrame, output_path: Path) -> None:
    df = by_k.loc[by_k["metric"] == "Within 10% of target 3D descriptors"].copy()
    if df.empty:
        raise ValueError("No morphology matching rows found in by-k summary.")

    df["ci_low"], df["ci_high"] = zip(*df.apply(lambda r: wilson_ci(int(r["k_match"]), int(r["n_considered"])), axis=1))
    x = df["num_conditioned"].to_numpy(dtype=int)
    y = df["effectiveness"].to_numpy(dtype=float)
    lo = df["ci_low"].to_numpy(dtype=float)
    hi = df["ci_high"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(x, y, marker="o", color=TEAL[5], linewidth=2.2, markersize=6)
    ax.fill_between(x, lo, hi, color=TEAL[1], alpha=0.25, linewidth=0)
    ax.set_xticks(x)
    ax.set_xlim(x.min() - 0.3, x.max() + 0.3)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("# of conditioned descriptors $k$", labelpad=8)
    ax.set_ylabel("Morphology-proxy match rate", labelpad=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.20, bottom=0.20, right=0.98, top=0.97)
    fig.savefig(output_path, format=output_path.suffix.lstrip("."), dpi=300)
    plt.close(fig)


def plot_success_per_target_condition(target_summary: pd.DataFrame, output_path: Path) -> None:
    df = target_summary.copy()
    x = np.arange(len(df))
    y = df["effectiveness"].to_numpy(dtype=float)
    lo = df["ci_low"].to_numpy(dtype=float)
    hi = df["ci_high"].to_numpy(dtype=float)
    yerr = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(
        x,
        y,
        width=0.72,
        color=TEAL[5],
        alpha=0.95,
        edgecolor=TEAL[4],
        linewidth=1.2,
    )
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=TEAL[4], elinewidth=1.6, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist(), rotation=24, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Target-condition success rate", labelpad=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.15, bottom=0.30, right=0.98, top=0.97)
    fig.savefig(output_path, format=output_path.suffix.lstrip("."), dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_paper_style()

    queries_jsonl = Path(args.queries_jsonl).expanduser().resolve()
    metrics_csv = Path(args.metrics_csv).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_df = load_heldout_samples(queries_jsonl)
    metrics_df = normalize_metrics(pd.read_csv(metrics_csv))
    device = resolve_device(args.device)
    predictor = load_predictor(checkpoint, device)

    samples_df = add_condition_matches_from_params(
        samples_df=samples_df,
        morph_df=metrics_df,
        model=predictor,
        device=str(device),
        sa_threshold=args.sa_threshold,
        beta_fraction_threshold=args.beta_threshold,
        tol_pct=args.tol_pct,
    )

    summary = compute_condition_matching_effectiveness(samples_df)
    by_k = compute_condition_matching_effectiveness_by_k(samples_df)
    by_target = build_target_success_summary(samples_df)

    samples_path = output_dir / "heldout_samples_with_matches.csv"
    summary_path = output_dir / "condition_matching_summary.csv"
    by_k_path = output_dir / "condition_matching_by_k.csv"
    by_target_path = output_dir / "condition_matching_by_target.csv"
    success_plot_path = output_dir / "conditions_success_stacked.svg"
    sph_plot_path = output_dir / "samples_per_hit_vs_k.svg"
    morph_plot_path = output_dir / "morph_match_rate_vs_k.svg"
    target_plot_path = output_dir / "success_per_target_condition.svg"
    metadata_path = output_dir / "run_metadata.json"

    samples_df.to_csv(samples_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_k.to_csv(by_k_path, index=False)
    by_target.to_csv(by_target_path, index=False)
    plot_conditions_success(samples_df, success_plot_path)
    plot_samples_per_hit(samples_df, sph_plot_path)
    plot_morph_match_rate_vs_k(by_k, morph_plot_path)
    plot_success_per_target_condition(by_target, target_plot_path)

    metadata = {
        "queries_jsonl": str(queries_jsonl),
        "metrics_csv": str(metrics_csv),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "tol_pct": float(args.tol_pct),
        "sa_threshold": float(args.sa_threshold),
        "beta_threshold": float(args.beta_threshold),
        "n_queries": int(samples_df["cond_idx"].nunique()),
        "n_generated_sequences": int(len(samples_df)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print("Wrote", samples_path)
    print("Wrote", summary_path)
    print("Wrote", by_k_path)
    print("Wrote", by_target_path)
    print("Wrote", success_plot_path)
    print("Wrote", sph_plot_path)
    print("Wrote", morph_plot_path)
    print("Wrote", target_plot_path)
    print("Wrote", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
