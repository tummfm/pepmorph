#!/usr/bin/env python

from __future__ import annotations

import argparse
import ast
import math
import pickle
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from common import DEFAULT_AP_CHECKPOINT, FIGS_DIR, GEN_PEPTIDES_DIR, RESULTS_DIR, TEAL, set_paper_style
from classifier.models import PeptidePredictor
from cvae.utils import ALPHABET, MAX_SEQ_LENGTH, convert_and_pad, esm_model_pretrained

FEATURES = [
    "length",
    "is_assembled",
    "ap",
    "has_beta_sheet_content",
    "hydrophobic_moment",
    "net_charge",
]

MORPH_FEATURES = ["has_beta_sheet_content", "hydrophobic_moment", "net_charge"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Condition matching effectiveness report.")
    parser.add_argument(
        "--results-pkl",
        type=str,
        default=str(RESULTS_DIR / "cvae_evaluation_results.pkl"),
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=str(GEN_PEPTIDES_DIR / "peptide_metrics_validation.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_AP_CHECKPOINT),
    )
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--fig-dir", type=str, default=str(FIGS_DIR))
    parser.add_argument("--save-updated-pkl", action="store_true")
    parser.add_argument("--tol-pct", type=float, default=0.10)
    parser.add_argument("--sa-threshold", type=float, default=0.5)
    parser.add_argument("--beta-threshold", type=float, default=0.0)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def _normalize_sa_target(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, np.integer, float)):
        return 1 if int(round(float(x))) != 0 else 0
    s = str(x).strip().lower()
    if s in {"sa", "yes", "true", "1", "assembled", "self-assembly"}:
        return 1
    if s in {"no-sa", "no", "false", "0", "non-assembly", "not"}:
        return 0
    return np.nan


def _rel_match(pred, targ, tol_abs=0.1):
    pred = float(pred)
    targ = float(targ)
    return abs(pred - targ) <= tol_abs


def _as_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return list(v)
        except Exception:
            return [t.strip() for t in x.split(",") if t.strip()]
    return []


@torch.inference_mode()
def _predict_ap_sa_aligned(model, sequences, device="cuda", batch_size=2048, sa_threshold=0.5):
    model.to(device).eval()
    pred_ap, pred_prob, pred_lbl = [], [], []
    for start in tqdm(range(0, len(sequences), batch_size), desc="Predicting AP/SA"):
        chunk = sequences[start : start + batch_size]
        data = [(f"peptide_{start + i}", s) for i, s in enumerate(chunk)]
        tokens = convert_and_pad(data, seq_length=MAX_SEQ_LENGTH).to(device)
        ap_preds, sa_preds = model(tokens)
        ap = ap_preds.detach().float().cpu().numpy().ravel()
        prob = sa_preds.detach().float().cpu().numpy().ravel()
        lbl = (prob >= sa_threshold).astype(np.int32)
        pred_ap.extend(ap.tolist())
        pred_prob.extend(prob.tolist())
        pred_lbl.extend(lbl.tolist())
    return np.array(pred_ap), np.array(pred_prob), np.array(pred_lbl)


def add_condition_matches_from_params(
    samples_df: pd.DataFrame,
    morph_df: pd.DataFrame,
    model: torch.nn.Module,
    device: str = "cuda",
    sa_threshold: float = 0.5,
    beta_fraction_threshold: float = 0.0,
    tol_pct: float = 0.1,
) -> pd.DataFrame:
    df = samples_df.copy()

    seqs = df["sequence"].tolist()
    pred_ap, pred_sa_prob, pred_is_assembled = _predict_ap_sa_aligned(
        model, seqs, device=device, batch_size=2048, sa_threshold=sa_threshold
    )
    df["pred_ap"] = pred_ap
    df["pred_sa_prob"] = pred_sa_prob
    df["pred_is_assembled"] = pred_is_assembled

    need = ["sequence", "beta_sheet_fraction", "hydrophobic_moment", "net_charge"]
    missing = [c for c in need if c not in morph_df.columns]
    if missing:
        raise ValueError(f"Missing columns in metrics CSV: {missing}")

    morph = morph_df[need].drop_duplicates(subset=["sequence"], keep="first").rename(
        columns={
            "beta_sheet_fraction": "pred_beta_sheet_fraction",
            "hydrophobic_moment": "pred_hydrophobic_moment",
            "net_charge": "pred_net_charge",
        }
    )

    df = df.merge(morph, on="sequence", how="left", validate="many_to_one")

    df["pred_has_beta_sheet_content"] = df["pred_beta_sheet_fraction"] > float(beta_fraction_threshold)
    df["seq_length"] = df["sequence"].str.len().astype("Int64")

    def _match_row(row, feat):
        used = set(_as_list(row.get("used_features", [])))
        if feat not in used:
            return np.nan
        params = row.get("params", {}) or {}
        if feat not in params:
            return np.nan
        targ = params[feat]

        if feat == "length":
            if pd.isna(row["seq_length"]):
                return np.nan
            try:
                return int(row["seq_length"]) == int(targ)
            except Exception:
                return np.nan

        if feat == "is_assembled":
            targ_bin = _normalize_sa_target(targ)
            if pd.isna(targ_bin) or pd.isna(row["pred_is_assembled"]):
                return np.nan
            return int(row["pred_is_assembled"]) == int(targ_bin)

        if feat == "ap":
            if pd.isna(row["pred_ap"]) or pd.isna(targ):
                return np.nan
            return _rel_match(row["pred_ap"], targ, tol_abs=tol_pct)

        if feat == "has_beta_sheet_content":
            pred = row["pred_has_beta_sheet_content"]
            if pd.isna(pred) or pd.isna(targ):
                return np.nan
            try:
                targ_bin = bool(int(targ)) if not isinstance(targ, bool) else targ
            except Exception:
                return np.nan
            return bool(pred) == bool(targ_bin)

        if feat == "hydrophobic_moment":
            pred = row["pred_hydrophobic_moment"]
            if pd.isna(pred) or pd.isna(targ):
                return np.nan
            return _rel_match(pred, targ, tol_abs=tol_pct)

        if feat == "net_charge":
            pred = row["pred_net_charge"]
            if pd.isna(pred) or pd.isna(targ):
                return np.nan
            try:
                return int(round(float(pred))) == int(round(float(targ)))
            except Exception:
                return np.nan

        return np.nan

    for feat in FEATURES:
        df[f"match_{feat}"] = df.apply(lambda r, f=feat: _match_row(r, f), axis=1).astype("boolean")

    return df


def compute_condition_matching_effectiveness(samples_df: pd.DataFrame) -> pd.DataFrame:
    df = samples_df

    considered_morph = []
    matched_morph = []
    for _, r in df.iterrows():
        used = set(_as_list(r.get("used_features", [])))
        tgt = [f for f in MORPH_FEATURES if f in used]
        if not tgt:
            considered_morph.append(False)
            matched_morph.append(False)
            continue
        vals = []
        ok = True
        for f in tgt:
            v = r.get(f"match_{f}", pd.NA)
            if pd.isna(v):
                ok = False
                break
            vals.append(bool(v))
        considered_morph.append(ok)
        matched_morph.append(all(vals) if ok else False)

    n_morph = int(np.sum(considered_morph))
    k_morph = int(np.sum(np.array(matched_morph)[considered_morph])) if n_morph else 0
    eff_morph = (k_morph / n_morph) if n_morph else np.nan

    considered_ap = []
    matched_ap = []
    for _, r in df.iterrows():
        used = set(_as_list(r.get("used_features", [])))
        if "ap" in used:
            v = r.get("match_ap", pd.NA)
            if not pd.isna(v):
                considered_ap.append(True)
                matched_ap.append(bool(v))
            else:
                considered_ap.append(False)
                matched_ap.append(False)
        else:
            considered_ap.append(False)
            matched_ap.append(False)
    n_ap = int(np.sum(considered_ap))
    k_ap = int(np.sum(np.array(matched_ap)[considered_ap])) if n_ap else 0
    eff_ap = (k_ap / n_ap) if n_ap else np.nan

    considered_sa = []
    matched_sa = []
    for _, r in df.iterrows():
        used = set(_as_list(r.get("used_features", [])))
        if "is_assembled" in used:
            v = r.get("match_is_assembled", pd.NA)
            if not pd.isna(v):
                considered_sa.append(True)
                matched_sa.append(bool(v))
            else:
                considered_sa.append(False)
                matched_sa.append(False)
        else:
            considered_sa.append(False)
            matched_sa.append(False)
    n_sa = int(np.sum(considered_sa))
    k_sa = int(np.sum(np.array(matched_sa)[considered_sa])) if n_sa else 0
    eff_sa = (k_sa / n_sa) if n_sa else np.nan

    considered_all = []
    matched_all = []
    for _, r in df.iterrows():
        used = [f for f in FEATURES if f in set(_as_list(r.get("used_features", [])))]
        if not used:
            considered_all.append(False)
            matched_all.append(False)
            continue
        vals = []
        ok = True
        for f in used:
            v = r.get(f"match_{f}", pd.NA)
            if pd.isna(v):
                ok = False
                break
            vals.append(bool(v))
        considered_all.append(ok)
        matched_all.append(all(vals) if ok else False)

    n_all = int(np.sum(considered_all))
    k_all = int(np.sum(np.array(matched_all)[considered_all])) if n_all else 0
    eff_all = (k_all / n_all) if n_all else np.nan

    summary = pd.DataFrame(
        [
            {
                "metric": "Within 10% of target 3D descriptors",
                "k": k_morph,
                "n": n_morph,
                "effectiveness": eff_morph,
            },
            {"metric": "Within 10% of target AP", "k": k_ap, "n": n_ap, "effectiveness": eff_ap},
            {"metric": "SA/no-SA classification", "k": k_sa, "n": n_sa, "effectiveness": eff_sa},
            {
                "metric": "Within 10% of all target descriptors",
                "k": k_all,
                "n": n_all,
                "effectiveness": eff_all,
            },
        ]
    )
    return summary


def compute_condition_matching_effectiveness_by_k(samples_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in samples_df.iterrows():
        used = set(_as_list(r.get("used_features", [])))
        used_in_scope = [f for f in FEATURES if f in used]
        k_used = len(used_in_scope)
        if not used_in_scope:
            continue

        def _row_and_across(feats):
            tgt = [f for f in feats if f in used]
            if not tgt:
                return False, False
            vals = []
            for f in tgt:
                v = r.get(f"match_{f}", pd.NA)
                if pd.isna(v):
                    return False, False
                vals.append(bool(v))
            return True, all(vals)

        def _row_single(feat):
            if feat not in used:
                return False, False
            v = r.get(f"match_{feat}", pd.NA)
            if pd.isna(v):
                return False, False
            return True, bool(v)

        c_morph, m_morph = _row_and_across(MORPH_FEATURES)
        c_ap, m_ap = _row_single("ap")
        c_sa, m_sa = _row_single("is_assembled")
        if not used_in_scope:
            c_all = False
            m_all = False
        else:
            vals = []
            ok = True
            for f in used_in_scope:
                v = r.get(f"match_{f}", pd.NA)
                if pd.isna(v):
                    ok = False
                    break
                vals.append(bool(v))
            c_all = ok
            m_all = all(vals) if ok else False

        rows.extend(
            [
                {
                    "num_conditioned": k_used,
                    "metric": "Within 10% of target 3D descriptors",
                    "k_match": int(m_morph),
                    "n_considered": int(c_morph),
                },
                {
                    "num_conditioned": k_used,
                    "metric": "Within 10% of target AP",
                    "k_match": int(m_ap),
                    "n_considered": int(c_ap),
                },
                {
                    "num_conditioned": k_used,
                    "metric": "SA/no-SA classification",
                    "k_match": int(m_sa),
                    "n_considered": int(c_sa),
                },
                {
                    "num_conditioned": k_used,
                    "metric": "Within 10% of all target descriptors",
                    "k_match": int(m_all),
                    "n_considered": int(c_all),
                },
            ]
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = (
        out.groupby(["num_conditioned", "metric"], as_index=False)
        .agg(k_match=("k_match", "sum"), n_considered=("n_considered", "sum"))
        .assign(effectiveness=lambda d: d["k_match"] / d["n_considered"].replace(0, np.nan))
    )

    return out.sort_values(["num_conditioned", "metric"]).reset_index(drop=True)


def plot_conditions_success(samples_df: pd.DataFrame, output_path: Path) -> None:
    tmp = samples_df.copy()
    tmp = tmp[tmp["ood_type"] == "in_dist"].copy()

    def _count_k(u):
        used = set(_as_list(u))
        return sum(f in used for f in FEATURES)

    def _row_success(row):
        used = [f for f in FEATURES if f in set(_as_list(row.get("used_features", [])))]
        if not used:
            return False
        for feat in used:
            v = row.get("match_" + feat, pd.NA)
            if pd.isna(v) or not bool(v):
                return False
        return True

    tmp["k"] = tmp["used_features"].apply(_count_k)
    tmp["success"] = tmp.apply(_row_success, axis=1)

    plot_df = (
        tmp[tmp["k"] > 0]
        .groupby("k", as_index=False)
        .agg(total=("success", "size"), successes=("success", "sum"))
    )
    plot_df["remainder"] = plot_df["total"] - plot_df["successes"]

    max_k = len(FEATURES)
    full_k = np.arange(1, max_k + 1, dtype=int)
    plot_df = plot_df.set_index("k").reindex(full_k, fill_value=0).reset_index()

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    x = plot_df["k"].to_numpy()
    succ = plot_df["successes"].to_numpy()
    rem = plot_df["remainder"].to_numpy()

    ax.bar(x, succ, width=0.7, label="Matching peptides", color=TEAL[4], alpha=0.9, linewidth=0)
    ax.bar(x, rem, width=0.7, bottom=succ, label="All peptides", color=TEAL[0], alpha=0.9, linewidth=0)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_xticks(full_k)
    ax.set_xlim(full_k.min() - 0.6, full_k.max() + 0.6)

    ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.9, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_samples_per_hit(samples_df: pd.DataFrame, output_path: Path) -> None:
    tmp = samples_df.copy()
    tmp = tmp[tmp["ood_type"] == "in_dist"].copy()

    def _count_k(u):
        used = set(_as_list(u))
        return sum(f in used for f in FEATURES)

    def _row_success(row):
        used = [f for f in FEATURES if f in set(_as_list(row.get("used_features", [])))]
        if not used:
            return False
        for feat in used:
            v = row.get("match_" + feat, pd.NA)
            if pd.isna(v) or not bool(v):
                return False
        return True

    tmp["k"] = tmp["used_features"].apply(_count_k)
    tmp["success"] = tmp.apply(_row_success, axis=1)

    plot_df = (
        tmp[tmp["k"] > 0]
        .groupby("k", as_index=False)
        .agg(total=("success", "size"), successes=("success", "sum"))
    )
    max_k = len(FEATURES)
    full_k = np.arange(1, max_k + 1, dtype=int)
    plot_df = plot_df.set_index("k").reindex(full_k, fill_value=0).reset_index()

    def wilson_ci(k, n, z=1.959963984540054):
        if n == 0:
            return (np.nan, np.nan)
        p = k / n
        denom = 1 + (z**2) / n
        center = (p + (z**2) / (2 * n)) / denom
        half = (z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom
        lo = max(0.0, center - half)
        hi = min(1.0, center + half)
        return lo, hi

    plot_df["p_hat"] = np.where(plot_df["total"] > 0, plot_df["successes"] / plot_df["total"], np.nan)
    cis = plot_df.apply(
        lambda r: wilson_ci(int(r["successes"]), int(r["total"])) if r["total"] > 0 else (np.nan, np.nan), axis=1
    )
    plot_df["p_lo"] = [c[0] for c in cis]
    plot_df["p_hi"] = [c[1] for c in cis]

    eps = 1e-12
    plot_df["E_1hit"] = 1.0 / np.maximum(plot_df["p_hat"], eps)
    plot_df["E_1hit_lo"] = 1.0 / np.maximum(plot_df["p_hi"], eps)
    plot_df["E_1hit_hi"] = 1.0 / np.maximum(plot_df["p_lo"], eps)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    x = plot_df["k"].to_numpy()
    y = plot_df["E_1hit"].to_numpy()

    ax.plot(x, y, marker="o", color=TEAL[4], linewidth=2.2, markersize=6, label=r"Expected samples per hit (1/p_k)")
    ax.fill_between(x, plot_df["E_1hit_lo"], plot_df["E_1hit_hi"], color=TEAL[1], alpha=0.25, linewidth=0)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_xticks(full_k)
    ax.set_xlim(full_k.min() - 0.6, full_k.max() + 0.6)
    ax.set_xlabel("# of conditioned descriptors k")
    ax.set_ylabel("Samples per hit (1/p_k)")

    ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.9, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_paper_style()

    output_dir = Path(args.output_dir).expanduser().resolve()
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.results_pkl).expanduser(), "rb") as f:
        results = pickle.load(f)

    samples_df = results["samples_df"]

    metrics_df = pd.read_csv(args.metrics_csv)

    min_max = {
        "hydrophobic_moment": (0.000000, 1.998000),
        "net_charge": (-6.000000, 6.000000),
    }

    for col, (mn, mx) in min_max.items():
        if col not in metrics_df.columns:
            raise ValueError(f"metrics CSV must contain '{col}'.")
        rng = mx - mn
        metrics_df[col] = (metrics_df[col] - mn) / rng

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = PeptidePredictor(deepcopy(esm_model_pretrained), alphabet=ALPHABET)
    state = torch.load(Path(args.checkpoint).expanduser(), map_location=device, weights_only=True)
    predictor.load_state_dict(state)
    predictor.to(device).eval()

    samples_df = add_condition_matches_from_params(
        samples_df=samples_df,
        morph_df=metrics_df,
        model=predictor,
        device=device,
        sa_threshold=args.sa_threshold,
        beta_fraction_threshold=args.beta_threshold,
        tol_pct=args.tol_pct,
    )

    summary = compute_condition_matching_effectiveness(samples_df)
    summary.to_csv(output_dir / "condition_matching_summary.csv", index=False)

    by_k = compute_condition_matching_effectiveness_by_k(samples_df)
    by_k.to_csv(output_dir / "condition_matching_by_k.csv", index=False)

    if not args.skip_plots:
        plot_conditions_success(samples_df, fig_dir / "conditions_success_stacked.svg")
        plot_samples_per_hit(samples_df, fig_dir / "samples_per_hit_vs_k.pdf")

    if args.save_updated_pkl:
        results["samples_df"] = samples_df
        out_pkl = output_dir / "cvae_evaluation_results_with_matches.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(results, f)
        print("Wrote", out_pkl)

    print("Wrote", output_dir / "condition_matching_summary.csv")
    print("Wrote", output_dir / "condition_matching_by_k.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
