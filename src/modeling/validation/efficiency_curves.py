#!/usr/bin/env python

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS_DIR, RESULTS_DIR, TEAL, VALIDATION_ARTIFACTS, resolve_path, set_paper_style


SPHERE_WINDOW = dict(
    length_min=5, length_max=7,
    hydro_min=1.1988, hydro_max=2.0,
    q_min=-1.0, q_max=1.0,
    require_beta=False,
)

FIBER_WINDOW = dict(
    length_min=7, length_max=10,
    hydro_min=None, hydro_max=None,
    q_min=-1.0, q_max=1.0,
    require_beta=True,
)

DEFAULT_FILE_MAP = [
    dict(
        name="PepMorph (sphere-targeted)",
        n_generated=4800,
        ap_file="gen_peptides/filtered_ap_peptides_spheres_final.txt",
        metrics_file="gen_peptides/spheres_metrics_merged.csv",
        target="spheres",
    ),
    dict(
        name="PepMorph (fiber-targeted)",
        n_generated=4800,
        ap_file="gen_peptides/filtered_ap_peptides_fiber_final.txt",
        metrics_file="gen_peptides/fibers_metrics.csv",
        target="fibers",
    ),
    dict(
        name="Random sampling (5-10 aa)",
        n_generated=4800,
        ap_file="gen_peptides/filtered_ap_peptides_random.txt",
        metrics_file="gen_peptides/random_metrics.csv",
        target="spheres",
    ),
    dict(
        name="Random sampling (5-10 aa)",
        n_generated=4800,
        ap_file="gen_peptides/filtered_ap_peptides_random.txt",
        metrics_file="gen_peptides/random_metrics.csv",
        target="fibers",
    ),
    dict(
        name="Random sampling (5-10 aa)",
        n_generated=4800,
        ap_file="gen_peptides/filtered_ap_peptides_random.txt",
        metrics_file="gen_peptides/random_metrics.csv",
        target="untargeted",
    ),
    dict(
        name="Length-cond. unconditional (sphere lengths)",
        n_generated=4800,
        ap_file="gen_peptides/filtered_ap_peptides_random_spheres_final.txt",
        metrics_file="gen_peptides/spheres_metrics_unconditional.csv",
        target="spheres",
    ),
    dict(
        name="Length-cond. unconditional (fiber lengths)",
        n_generated=4800,
        ap_file="gen_peptides/filtered_ap_peptides_random_fiber_final.txt",
        metrics_file="gen_peptides/fibers_metrics_unconditional.csv",
        target="fibers",
    ),
]


def beta_flag(beta_sheet_fraction: float) -> bool:
    return pd.notna(beta_sheet_fraction) and float(beta_sheet_fraction) > 0.0


def load_ap_pass_txt(path: Path) -> pd.DataFrame:
    """
    Expected format:
    SEQ,LENGTH,AP,SA_prob
    """
    df = pd.read_csv(path, header=None, names=["sequence", "3-seq", "AP", "SA_prob"])
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df["length"] = df["sequence"].str.len().astype(int)
    df["AP"] = df["AP"].astype(float)
    df["SA_prob"] = df["SA_prob"].astype(float)
    return df


def load_metrics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sequence" not in df.columns:
        raise ValueError(f"{path} must contain a 'sequence' column")
    df["sequence"] = df["sequence"].astype(str).str.strip()
    df["length"] = df["sequence"].str.len().astype(int)
    for col in ["beta_sheet_fraction", "hydrophobic_moment", "net_charge"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def merge_ap_and_metrics(ap_df: pd.DataFrame, met_df: pd.DataFrame) -> pd.DataFrame:
    merged = ap_df.merge(
        met_df[["sequence", "beta_sheet_fraction", "hydrophobic_moment", "net_charge", "length"]],
        on="sequence",
        how="left",
        suffixes=("", "_met"),
    )
    merged["has_beta"] = merged["beta_sheet_fraction"].apply(beta_flag)
    return merged


def passes_window(row: pd.Series, window: dict) -> bool:
    if "length_min" in window and row["length"] < window["length_min"]:
        return False
    if "length_max" in window and row["length"] > window["length_max"]:
        return False

    if pd.isna(row["net_charge"]) or pd.isna(row["hydrophobic_moment"]) or pd.isna(row["beta_sheet_fraction"]):
        return False

    if window.get("require_beta", False) and not bool(row["has_beta"]):
        return False

    if window["q_min"] is not None and not (window["q_min"] <= float(row["net_charge"]) <= window["q_max"]):
        return False

    if window["hydro_min"] is not None and not (
        window["hydro_min"] <= float(row["hydrophobic_moment"]) <= window["hydro_max"]
    ):
        return False

    return True


def bootstrap_permutation_curves(
    hit_flags: np.ndarray,
    score: np.ndarray | None = None,
    n_boot: int = 800,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n = len(hit_flags)
    budgets = np.arange(1, n + 1)

    hits_mat = np.zeros((n_boot, n), dtype=float)
    best_mat = np.zeros((n_boot, n), dtype=float) if score is not None else None

    for b in range(n_boot):
        perm = rng.permutation(n)
        h = hit_flags[perm].astype(int)
        hits_mat[b] = np.cumsum(h)

        if score is not None:
            s = score[perm].astype(float)
            best_mat[b] = np.maximum.accumulate(s)

    mean_hits = hits_mat.mean(axis=0)
    lo_hits = np.percentile(hits_mat, 2.5, axis=0)
    hi_hits = np.percentile(hits_mat, 97.5, axis=0)

    if best_mat is None:
        return budgets, mean_hits, lo_hits, hi_hits, None

    if hit_flags.sum() == 0:
        return budgets, mean_hits, lo_hits, hi_hits, None

    best_mat = np.where(np.isfinite(best_mat), best_mat, np.nan)
    if np.all(np.isnan(best_mat)):
        return budgets, mean_hits, lo_hits, hi_hits, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_best = np.nanmean(best_mat, axis=0)
        lo_best = np.nanpercentile(best_mat, 2.5, axis=0)
        hi_best = np.nanpercentile(best_mat, 97.5, axis=0)
    return budgets, mean_hits, lo_hits, hi_hits, (mean_best, lo_best, hi_best)


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def prob_at_least_one_hit(p: float, n: np.ndarray) -> np.ndarray:
    return 1.0 - np.power(1.0 - p, n)


def pepfold_efficiency_for_entry(entry: dict, base_dir: Path, seed: int) -> dict:
    ap_path = (base_dir / entry["ap_file"]).resolve()
    metrics_path = (base_dir / entry["metrics_file"]).resolve()
    ap = load_ap_pass_txt(ap_path)
    met = load_metrics_csv(metrics_path)
    merged = merge_ap_and_metrics(ap, met)

    target = entry["target"]
    if target == "spheres":
        window = SPHERE_WINDOW
    elif target == "fibers":
        window = FIBER_WINDOW
    elif target == "untargeted":
        window = None
    else:
        raise ValueError(f"Unknown target: {target}")

    n_oracle_calls = len(merged)
    pepfold_success = merged["beta_sheet_fraction"].notna().sum()

    if window is None:
        merged["proxy_hit"] = False
    else:
        merged["proxy_hit"] = merged.apply(lambda r: passes_window(r, window), axis=1)

    n_hits = int(merged["proxy_hit"].sum())

    score = merged["AP"].to_numpy().astype(float)
    score_best_among_hits = np.where(merged["proxy_hit"].to_numpy(), score, -np.inf)

    return dict(
        name=entry["name"],
        target=target,
        n_generated=entry["n_generated"],
        n_ap_pass=n_oracle_calls,
        n_pepfold_success=int(pepfold_success),
        n_proxy_hits=n_hits,
        data=merged,
        score_best_among_hits=score_best_among_hits,
        seed=seed,
    )


def summarize_pepfold_results(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        n_gen = r["n_generated"]
        n_ap = r["n_ap_pass"]
        n_pf = r["n_pepfold_success"]
        n_hit = r["n_proxy_hits"]

        p = (n_hit / n_ap) if n_ap > 0 else np.nan
        lo, hi = wilson_ci(n_hit, n_ap) if n_ap > 0 else (np.nan, np.nan)

        calls_per_hit = (1 / p) if (p and p > 0) else np.inf
        calls_per_hit_lo = (1 / hi) if (hi and hi > 0) else np.inf
        calls_per_hit_hi = (1 / lo) if (lo and lo > 0) else np.inf

        rows.append(
            dict(
                cohort=r["name"],
                target=r["target"],
                generated=n_gen,
                ap_pass=n_ap,
                pepfold_success=n_pf,
                proxy_hits=n_hit,
                hit_rate=p,
                hit_rate_ci_lo=lo,
                hit_rate_ci_hi=hi,
                calls_per_hit=calls_per_hit,
                calls_per_hit_ci_lo=calls_per_hit_lo,
                calls_per_hit_ci_hi=calls_per_hit_hi,
            )
        )

    return pd.DataFrame(rows)


def augment_summary_generated_budget(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hit_rate_gen"] = df["proxy_hits"] / df["generated"]

    ci_gen = df.apply(lambda r: wilson_ci(int(r["proxy_hits"]), int(r["generated"])), axis=1)
    df["hit_rate_gen_ci_lo"] = [c[0] for c in ci_gen]
    df["hit_rate_gen_ci_hi"] = [c[1] for c in ci_gen]

    df["calls_per_hit_gen"] = np.where(df["hit_rate_gen"] > 0, 1.0 / df["hit_rate_gen"], np.inf)
    df["calls_per_hit_gen_ci_lo"] = np.where(
        df["hit_rate_gen_ci_hi"] > 0, 1.0 / df["hit_rate_gen_ci_hi"], np.inf
    )
    df["calls_per_hit_gen_ci_hi"] = np.where(
        df["hit_rate_gen_ci_lo"] > 0, 1.0 / df["hit_rate_gen_ci_lo"], np.inf
    )

    return df


def plot_pepfold_curves(
    results: list[dict],
    target: str,
    fig_dir: Path,
    n_boot: int,
    seed: int,
    include_names: list[str] | None = None,
):
    subset = [r for r in results if r["target"] == target]
    if include_names is not None:
        subset = [r for r in subset if r["name"] in include_names]

    if len(subset) == 0:
        print("No cohorts for target:", target)
        return

    plt.figure(figsize=(7.2, 4.8))
    for r in subset:
        hit = r["data"]["proxy_hit"].to_numpy().astype(int)
        budgets, m, lo, hi, _ = bootstrap_permutation_curves(hit, score=None, n_boot=n_boot, seed=seed)
        plt.plot(budgets, m, label=r["name"])
        plt.fill_between(budgets, lo, hi, alpha=0.2)

    plt.xlabel("PEP-FOLD calls (AP-pass peptides attempted)")
    plt.ylabel("Cumulative # proxy-feasible hits")
    plt.title(f"PEP-FOLD oracle efficiency: {target}")
    plt.legend(frameon=True)
    plt.tight_layout()
    out1 = fig_dir / f"pepfold_hits_vs_calls_{target}.pdf"
    plt.savefig(out1)
    plt.close()
    print("Wrote", out1)

    plt.figure(figsize=(7.2, 4.8))
    plotted = False
    for r in subset:
        hit = r["data"]["proxy_hit"].to_numpy().astype(int)
        score = r["score_best_among_hits"]
        budgets, _, _, _, best = bootstrap_permutation_curves(hit, score=score, n_boot=n_boot, seed=seed)
        if best is None:
            print(f"Skipping best-AP curve (no proxy hits): {r['name']}")
            continue
        mean_best, lo_best, hi_best = best

        mean_best = np.where(np.isfinite(mean_best), mean_best, np.nan)
        lo_best = np.where(np.isfinite(lo_best), lo_best, np.nan)
        hi_best = np.where(np.isfinite(hi_best), hi_best, np.nan)

        plt.plot(budgets, mean_best, label=r["name"])
        plt.fill_between(budgets, lo_best, hi_best, alpha=0.2)
        plotted = True

    if not plotted:
        print("No best-AP curves to plot for target:", target)
        plt.close()
        return

    plt.xlabel("PEP-FOLD calls (AP-pass peptides attempted)")
    plt.ylabel("Best predicted AP among proxy-feasible hits")
    plt.title(f"PEP-FOLD oracle efficiency (best-AP): {target}")
    plt.legend(frameon=True)
    plt.tight_layout()
    out2 = fig_dir / f"pepfold_bestAP_vs_calls_{target}.pdf"
    plt.savefig(out2)
    plt.close()
    print("Wrote", out2)


def plot_calls_per_hit_bar(
    df: pd.DataFrame,
    target: str,
    metric: str,
    title: str,
    outfile: Path,
    cohort_order: list[str] | None = None,
    logy: bool = True,
):
    d = df[df["target"] == target].copy()
    if cohort_order is not None:
        d["cohort"] = pd.Categorical(d["cohort"], categories=cohort_order, ordered=True)
        d = d.sort_values("cohort")

    y = d[metric].to_numpy()
    lo = d[f"{metric}_ci_lo"].to_numpy()
    hi = d[f"{metric}_ci_hi"].to_numpy()

    finite = np.isfinite(y)
    if not finite.any():
        print("No finite calls-per-hit values for target:", target)
        return

    yerr = np.vstack([y - lo, hi - y])
    yerr[:, ~finite] = 0.0
    yerr = np.nan_to_num(yerr, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(d))

    colors = [TEAL[4]] * len(d)
    y_plot = y.copy()
    y_plot[~finite] = np.nan
    ax.bar(x, y_plot, color=colors, alpha=0.9, linewidth=0)

    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#3F3F3F", elinewidth=1.2, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(d["cohort"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("Calls per hit")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Wrote", outfile)


def plot_prob_hit_vs_budget(
    df: pd.DataFrame,
    target: str,
    budget_kind: str,
    outfile: Path,
    cohort_order: list[str] | None = None,
    max_budget: int | None = None,
):
    d = df[df["target"] == target].copy()
    if cohort_order is not None:
        d["cohort"] = pd.Categorical(d["cohort"], categories=cohort_order, ordered=True)
        d = d.sort_values("cohort")

    if budget_kind == "generated":
        p = d["hit_rate_gen"].to_numpy()
        plo = d["hit_rate_gen_ci_lo"].to_numpy()
        phi = d["hit_rate_gen_ci_hi"].to_numpy()
        budget_max = int(d["generated"].max()) if max_budget is None else int(max_budget)
        xlabel = "# generated samples (filter calls)"
    elif budget_kind == "pepfold":
        p = d["hit_rate"].to_numpy()
        plo = d["hit_rate_ci_lo"].to_numpy()
        phi = d["hit_rate_ci_hi"].to_numpy()
        budget_max = int(d["ap_pass"].max()) if max_budget is None else int(max_budget)
        xlabel = "PEP-FOLD calls (AP-pass sequences attempted)"
    else:
        raise ValueError("budget_kind must be 'generated' or 'pepfold'")

    budgets = np.arange(0, budget_max + 1, dtype=int)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, row in d.reset_index(drop=True).iterrows():
        name = row["cohort"]
        if not np.isfinite(p[i]):
            print(f"Skipping probability curve (non-finite hit rate): {name}")
            continue
        y = prob_at_least_one_hit(p[i], budgets)
        ylo = prob_at_least_one_hit(plo[i], budgets) if np.isfinite(plo[i]) else y * np.nan
        yhi = prob_at_least_one_hit(phi[i], budgets) if np.isfinite(phi[i]) else y * np.nan

        color = TEAL[min(5, i + 1)]
        ax.plot(budgets, y, label=name, linewidth=2.0, color=color)
        ax.fill_between(budgets, ylo, yhi, alpha=0.18, linewidth=0, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(at least one proxy-hit)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Probability of success vs budget ({target}, {budget_kind})")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.9, fontsize=10)
    plt.tight_layout()
    fig.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Wrote", outfile)


def make_si_efficiency_table(df: pd.DataFrame, rows: list[tuple[str, str]]) -> pd.DataFrame:
    def format_ci(lo, hi, fmt="{:.4f}"):
        return f"[{fmt.format(lo)}, {fmt.format(hi)}]"

    out_rows = []
    for cohort, target in rows:
        r = df[(df["cohort"] == cohort) & (df["target"] == target)].iloc[0]
        out_rows.append(
            dict(
                Cohort=cohort,
                Target=target,
                Generated=int(r["generated"]),
                Hits=int(r["proxy_hits"]),
                HitRate=f'{r["hit_rate_gen"]:.4f} {format_ci(r["hit_rate_gen_ci_lo"], r["hit_rate_gen_ci_hi"])}',
                CallsPerHit=(
                    f'{r["calls_per_hit_gen"]:.1f} '
                    f'{format_ci(r["calls_per_hit_gen_ci_lo"], r["calls_per_hit_gen_ci_hi"], fmt="{:.1f}")}'
                ),
            )
        )
    return pd.DataFrame(out_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PEP-FOLD efficiency curves and summary tables")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(VALIDATION_ARTIFACTS),
        help="Base directory for gen_peptides inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory to write summary tables.",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=str(FIGS_DIR),
        help="Directory to write plots.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["spheres", "fibers"],
        help="Targets to plot (spheres, fibers).",
    )
    parser.add_argument(
        "--include-names",
        nargs="*",
        default=None,
        help="Optional cohort names to include.",
    )
    parser.add_argument("--n-boot", type=int, default=600, help="Bootstrap iterations for curves.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap permutations.")
    parser.add_argument("--skip-plots", action="store_true", help="Only write summary tables.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = resolve_path(args.base_dir, VALIDATION_ARTIFACTS)
    output_dir = resolve_path(args.output_dir, RESULTS_DIR)
    fig_dir = resolve_path(args.fig_dir, FIGS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    if not args.skip_plots:
        set_paper_style()

    results = []
    for entry in DEFAULT_FILE_MAP:
        ap_path = (base_dir / entry["ap_file"]).resolve()
        metrics_path = (base_dir / entry["metrics_file"]).resolve()
        if not (ap_path.exists() and metrics_path.exists()):
            print(f"Skipping (missing files): {entry['name']}")
            continue
        results.append(pepfold_efficiency_for_entry(entry, base_dir=base_dir, seed=args.seed))

    if len(results) == 0:
        print("No cohorts to process. Check --base-dir and input files.")
        return 0

    summary = summarize_pepfold_results(results)
    summary_path = output_dir / "pepfold_efficiency_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Wrote", summary_path)

    summary_aug = augment_summary_generated_budget(summary)
    summary_aug_path = output_dir / "pepfold_efficiency_summary_augmented.csv"
    summary_aug.to_csv(summary_aug_path, index=False)
    print("Wrote", summary_aug_path)

    if args.skip_plots:
        return 0

    order_sph = [
        "PepMorph (sphere-targeted)",
        "Length-cond. unconditional (sphere lengths)",
        "Random sampling (5-10 aa)",
    ]
    order_fib = [
        "PepMorph (fiber-targeted)",
        "Length-cond. unconditional (fiber lengths)",
        "Random sampling (5-10 aa)",
    ]

    for target in args.targets:
        plot_pepfold_curves(
            results,
            target=target,
            fig_dir=fig_dir,
            n_boot=args.n_boot,
            seed=args.seed,
            include_names=args.include_names,
        )

    plot_calls_per_hit_bar(
        df=summary_aug,
        target="spheres",
        metric="calls_per_hit_gen",
        title="Sample efficiency (spheres): generated calls per proxy-hit",
        outfile=fig_dir / "calls_per_hit_gen_spheres.pdf",
        cohort_order=order_sph,
        logy=True,
    )

    plot_calls_per_hit_bar(
        df=summary_aug,
        target="fibers",
        metric="calls_per_hit_gen",
        title="Sample efficiency (fibers): generated calls per proxy-hit",
        outfile=fig_dir / "calls_per_hit_gen_fibers.pdf",
        cohort_order=order_fib,
        logy=True,
    )

    plot_prob_hit_vs_budget(
        summary_aug,
        "spheres",
        "generated",
        fig_dir / "prob_hit_vs_budget_generated_spheres.svg",
        cohort_order=order_sph,
    )
    plot_prob_hit_vs_budget(
        summary_aug,
        "fibers",
        "generated",
        fig_dir / "prob_hit_vs_budget_generated_fibers.svg",
        cohort_order=order_fib,
    )

    plot_prob_hit_vs_budget(
        summary_aug,
        "spheres",
        "pepfold",
        fig_dir / "prob_hit_vs_budget_pepfold_spheres.svg",
        cohort_order=order_sph,
    )
    plot_prob_hit_vs_budget(
        summary_aug,
        "fibers",
        "pepfold",
        fig_dir / "prob_hit_vs_budget_pepfold_fibers.svg",
        cohort_order=order_fib,
    )

    targeted = summary_aug[summary_aug["target"].isin(["spheres", "fibers"])].copy()
    keep = set(order_sph + order_fib)
    targeted = targeted[targeted["cohort"].isin(keep)].copy()

    cols = [
        "cohort",
        "target",
        "generated",
        "ap_pass",
        "pepfold_success",
        "proxy_hits",
        "hit_rate_gen",
        "hit_rate_gen_ci_lo",
        "hit_rate_gen_ci_hi",
        "calls_per_hit_gen",
        "calls_per_hit_gen_ci_lo",
        "calls_per_hit_gen_ci_hi",
        "hit_rate",
        "hit_rate_ci_lo",
        "hit_rate_ci_hi",
        "calls_per_hit",
        "calls_per_hit_ci_lo",
        "calls_per_hit_ci_hi",
    ]
    targeted = targeted[cols]
    targeted_path = output_dir / "efficiency_table_targeted.csv"
    targeted.to_csv(targeted_path, index=False)
    print("Wrote", targeted_path)

    si_rows = [
        ("PepMorph (sphere-targeted)", "spheres"),
        ("Length-cond. unconditional (sphere lengths)", "spheres"),
        ("Random sampling (5-10 aa)", "spheres"),
        ("PepMorph (fiber-targeted)", "fibers"),
        ("Length-cond. unconditional (fiber lengths)", "fibers"),
        ("Random sampling (5-10 aa)", "fibers"),
    ]
    si_tbl = make_si_efficiency_table(summary_aug, si_rows)
    si_path = output_dir / "si_efficiency_table_generated_budget.csv"
    si_tbl.to_csv(si_path, index=False)
    print("Wrote", si_path)

    _ = rng  # keep seed usage explicit
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
