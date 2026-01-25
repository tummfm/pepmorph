#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CG_DIR = Path(__file__).resolve().parents[1]
if str(CG_DIR) not in sys.path:
    sys.path.insert(0, str(CG_DIR))

from common import ROBUSTNESS_ARTIFACTS_DIR, TEAL, set_paper_style


def parse_args() -> argparse.Namespace:
    base_dir = ROBUSTNESS_ARTIFACTS_DIR / "outputs"
    parser = argparse.ArgumentParser(description="Summarize robustness MD differences with CIs.")
    parser.add_argument("--input-csv", type=str, default=str(base_dir / "rmoi_and_ap_by_run_total.csv"))
    parser.add_argument("--output-dir", type=str, default=str(base_dir))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    z = 1.959963984540054
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(values: np.ndarray, stat_fn=np.mean, n_boot: int = 10000, alpha: float = 0.05, rng=None):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(0) if rng is None else rng

    n = values.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(values, size=n, replace=True)
        boots[i] = stat_fn(samp)

    stat = stat_fn(values)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return stat, lo, hi


def parse_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return True
    if s in {"no", "n", "false", "0"}:
        return False
    return np.nan


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    df = pd.read_csv(input_path)
    expected_cols = {
        "type",
        "target",
        "peptide",
        "run",
        "RMOI",
        "aggregation_propensity",
        "matches_morphology_visual",
        "notes_visual",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df["vis_ok"] = df["matches_morphology_visual"].apply(parse_bool)
    df["RMOI"] = pd.to_numeric(df["RMOI"], errors="coerce")
    df["AP"] = pd.to_numeric(df["aggregation_propensity"], errors="coerce")

    df["type"] = df["type"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip().str.lower()
    df["peptide"] = df["peptide"].astype(str).str.strip()
    df["run"] = df["run"].astype(str).str.strip()

    type_order = ["original", "bigger_box_1230", "martini2"]
    target_order = ["sphere", "fiber"]
    df["type"] = pd.Categorical(df["type"], categories=type_order, ordered=True)
    df["target"] = pd.Categorical(df["target"], categories=target_order, ordered=True)

    run_rows = []
    for (t, g), sub in df.groupby(["type", "target"], observed=True):
        n = int(sub["vis_ok"].notna().sum())
        k = int(sub["vis_ok"].fillna(False).sum())
        phat, lo, hi = wilson_ci(k, n)
        run_rows.append(
            {
                "type": t,
                "target": g,
                "n_runs": n,
                "k_success": k,
                "run_success_rate": phat,
                "run_CI_lo": lo,
                "run_CI_hi": hi,
                "RMOI_mean": sub["RMOI"].mean(),
                "RMOI_std": sub["RMOI"].std(ddof=1),
                "AP_mean": sub["AP"].mean(),
                "AP_std": sub["AP"].std(ddof=1),
            }
        )

    run_summary = pd.DataFrame(run_rows).sort_values(["target", "type"])
    run_summary_path = output_dir / "run_summary.csv"
    run_summary.to_csv(run_summary_path, index=False)

    pep = (
        df.groupby(["type", "target", "peptide"], observed=True)
        .agg(
            n_runs=("vis_ok", "count"),
            k_runs_ok=("vis_ok", "sum"),
            RMOI_mean=("RMOI", "mean"),
            AP_mean=("AP", "mean"),
        )
        .reset_index()
    )
    pep["majority_ok"] = pep["k_runs_ok"] >= 2

    pep_rows = []
    for (t, g), sub in pep.groupby(["type", "target"], observed=True):
        n = int(sub.shape[0])
        k = int(sub["majority_ok"].sum())
        phat, lo, hi = wilson_ci(k, n)
        pep_rows.append(
            {
                "type": t,
                "target": g,
                "n_peptides": n,
                "k_success_peptides": k,
                "peptide_success_rate": phat,
                "pep_CI_lo": lo,
                "pep_CI_hi": hi,
                "RMOI_mean_over_peptides": sub["RMOI_mean"].mean(),
                "AP_mean_over_peptides": sub["AP_mean"].mean(),
            }
        )

    pep_summary = pd.DataFrame(pep_rows).sort_values(["target", "type"])
    pep_summary_path = output_dir / "peptide_summary.csv"
    pep_summary.to_csv(pep_summary_path, index=False)

    print("Wrote", run_summary_path)
    print("Wrote", pep_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
