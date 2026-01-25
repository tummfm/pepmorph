#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import fisher_exact, norm
from statsmodels.stats.contingency_tables import Table2x2


@dataclass(frozen=True)
class BinomResult:
    x: int
    n: int


def wilson_ci(x: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return np.nan, np.nan, np.nan
    if x < 0 or x > n:
        raise ValueError(f"Invalid x={x} for n={n}")

    z = norm.ppf(1 - alpha / 2)
    p = x / n
    denom = 1 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return p, lo, hi


def fmt_pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def summarize_rate(name: str, res: BinomResult, alpha: float = 0.05) -> str:
    p, lo, hi = wilson_ci(res.x, res.n, alpha=alpha)
    return (
        f"{name}: {res.x}/{res.n} = {fmt_pct(p)} "
        f"(Wilson {(1 - alpha) * 100:.0f}% CI: {fmt_pct(lo)}-{fmt_pct(hi)})"
    )


def fisher_or_p(table, alternative: str = "two-sided"):
    or_, p = fisher_exact(table, alternative=alternative)
    return or_, p


def make_table(a: BinomResult, b: BinomResult):
    return [[a.x, a.n - a.x], [b.x, b.n - b.x]]


def or_ci_exact(table, alpha=0.05):
    t = Table2x2(np.asarray(table, dtype=float))
    ci_low, ci_high = t.oddsratio_confint(alpha=alpha, method="exact")
    return float(t.oddsratio), float(ci_low), float(ci_high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descriptor outcome Fisher tests and Wilson CIs.")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alpha = args.alpha

    spheres_pepmorph = BinomResult(x=12, n=15)
    spheres_descfail = BinomResult(x=5, n=11)
    fibrils_pepmorph = BinomResult(x=13, n=15)
    fibrils_descfail = BinomResult(x=5, n=12)

    all_pepmorph = BinomResult(
        x=spheres_pepmorph.x + fibrils_pepmorph.x,
        n=spheres_pepmorph.n + fibrils_pepmorph.n,
    )
    all_descfail = BinomResult(
        x=spheres_descfail.x + fibrils_descfail.x,
        n=spheres_descfail.n + fibrils_descfail.n,
    )

    print("== Success rates with Wilson CIs (visual, conditional on aggregation) ==")
    print(summarize_rate("Spheres - PepMorph", spheres_pepmorph, alpha))
    print(summarize_rate("Spheres - Descriptor-fail", spheres_descfail, alpha))
    print(summarize_rate("Fibrils - PepMorph", fibrils_pepmorph, alpha))
    print(summarize_rate("Fibrils - Descriptor-fail", fibrils_descfail, alpha))
    print(summarize_rate("All - PepMorph", all_pepmorph, alpha))
    print(summarize_rate("All - Descriptor-fail", all_descfail, alpha))

    print("== Fisher exact tests (PepMorph vs Descriptor-fail) ==")
    tab_s = make_table(spheres_pepmorph, spheres_descfail)
    or_s, p_s = fisher_or_p(tab_s)
    print("Spheres table:", tab_s, "(rows: PepMorph, Desc-fail; cols: success, failure)")
    print(f"Spheres OR = {or_s:.4g}, p = {p_s:.4g}")

    tab_f = make_table(fibrils_pepmorph, fibrils_descfail)
    or_f, p_f = fisher_or_p(tab_f)
    print("\nFibrils table:", tab_f, "(rows: PepMorph, Desc-fail; cols: success, failure)")
    print(f"Fibrils OR = {or_f:.4g}, p = {p_f:.4g}")

    tab_a = make_table(all_pepmorph, all_descfail)
    or_a, p_a = fisher_or_p(tab_a)
    print("\nAll table:", tab_a, "(rows: PepMorph, Desc-fail; cols: success, failure)")
    print(f"All OR = {or_a:.4g}, p = {p_a:.4g}")

    or_s_hat, or_s_lo, or_s_hi = or_ci_exact(tab_s, alpha=alpha)
    or_f_hat, or_f_lo, or_f_hi = or_ci_exact(tab_f, alpha=alpha)
    or_a_hat, or_a_lo, or_a_hi = or_ci_exact(tab_a, alpha=alpha)

    print(
        f"Spheres OR (Table2x2) = {or_s_hat:.4g}, "
        f"{(1-alpha)*100:.0f}% exact CI = [{or_s_lo:.4g}, {or_s_hi:.4g}]"
    )
    print(
        f"Fibrils OR (Table2x2) = {or_f_hat:.4g}, "
        f"{(1-alpha)*100:.0f}% exact CI = [{or_f_lo:.4g}, {or_f_hi:.4g}]"
    )
    print(
        f"All OR (Table2x2) = {or_a_hat:.4g}, "
        f"{(1-alpha)*100:.0f}% exact CI = [{or_a_lo:.4g}, {or_a_hi:.4g}]"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
