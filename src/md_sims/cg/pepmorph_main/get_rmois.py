#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

CG_DIR = Path(__file__).resolve().parents[1]
if str(CG_DIR) not in sys.path:
    sys.path.insert(0, str(CG_DIR))

from common import PEPMORPH_MAIN_ARTIFACTS_DIR

DEFAULT_RUNS = ["run_1", "run_2", "run_3"]


def parse_args() -> argparse.Namespace:
    base_dir = PEPMORPH_MAIN_ARTIFACTS_DIR
    parser = argparse.ArgumentParser(description="Aggregate RMOI and SASA metrics across PepMorph control runs.")
    parser.add_argument("--base-dir", type=str, default=str(base_dir))
    parser.add_argument(
        "--compute-script",
        type=str,
        default=str(Path(__file__).resolve().parent / "compute_rmoi.py"),
    )
    parser.add_argument("--cutoff-nm", type=float, default=0.6)
    parser.add_argument("--output-dir", type=str, default=str(base_dir / "outputs"))
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--max-peptides", type=int, default=None)
    parser.add_argument("--include-class-output", action="store_true")
    return parser.parse_args()


def parse_sasa(xvg_path: Path) -> float | None:
    """
    Read area.xvg/sasa.xvg, skip header lines starting with '#' or '@'.
    Parse rows as [time, area], then return (mean of first 2 areas) / (mean of last 2 areas).
    """
    data = []
    with xvg_path.open("r") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith(("#", "@")):
                continue
            parts = raw.split()
            if len(parts) >= 2:
                try:
                    _, area = float(parts[0]), float(parts[1])
                    data.append(area)
                except ValueError:
                    continue
    if not data:
        return None
    if len(data) == 1:
        return 1.0 if data[0] != 0 else None
    first = sum(data[:2]) / min(2, len(data))
    last = sum(data[-2:]) / min(2, len(data))
    if last == 0:
        return None
    return first / last


def find_run_dirs(runs_dir: Path) -> list[Path]:
    runs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]

    def sort_key(path: Path):
        match = re.search(r"(\d+)$", path.name)
        return (0, int(match.group(1))) if match else (1, path.name)

    return sorted(runs, key=sort_key)


def run_compute_rmoi(script: Path, gro: Path, xtc: Path, cutoff_nm: float) -> float | None:
    cmd = [sys.executable, str(script), "-g", str(gro), "-x", str(xtc), "-c", str(cutoff_nm)]
    out = subprocess.check_output(cmd, text=True).splitlines()
    for line in out:
        if line.startswith("Ratio of principal moments"):
            try:
                return float(line.split()[-1])
            except ValueError:
                return None
    return None


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    compute_script = Path(args.compute_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = base_dir / "runs"
    if not runs_dir.is_dir():
        raise SystemExit(f"Missing runs directory: {runs_dir}")

    run_dirs = find_run_dirs(runs_dir)
    if args.runs:
        run_dirs = [p for p in run_dirs if p.name in set(args.runs)]

    records: list[dict] = []
    for run_dir in run_dirs:
        class_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
        if args.classes:
            class_dirs = [p for p in class_dirs if p.name in set(args.classes)]
        class_dirs = sorted(class_dirs, key=lambda p: p.name)

        for class_dir in class_dirs:
            pep_dirs = sorted([p for p in class_dir.iterdir() if p.is_dir()])
            if args.max_peptides:
                pep_dirs = pep_dirs[: args.max_peptides]

            for pep_dir in pep_dirs:
                gro = pep_dir / "peptide-cg.gro"
                xtc = pep_dir / "trajout.xtc"
                if not (gro.is_file() and xtc.is_file()):
                    continue

                ratio = run_compute_rmoi(compute_script, gro, xtc, args.cutoff_nm)
                if ratio is None:
                    raise RuntimeError(f"[{run_dir.name}/{class_dir.name}/{pep_dir.name}] could not parse RMOI")

                area_xvg = pep_dir / "area.xvg"
                if not area_xvg.is_file():
                    area_xvg = pep_dir / "sasa.xvg"
                agg_prop = parse_sasa(area_xvg) if area_xvg.is_file() else None

                records.append(
                    {
                        "peptide": pep_dir.name,
                        "class": class_dir.name,
                        "run": run_dir.name,
                        "RMOI": ratio,
                        "aggregation_propensity": agg_prop,
                    }
                )

    df_long = pd.DataFrame.from_records(records)
    if df_long.empty:
        raise SystemExit("No records found. Check folder structure and file names.")

    out_by_run = output_dir / "rmoi_and_ap_by_run_total.csv"
    df_long[["peptide", "run", "RMOI", "aggregation_propensity"]].sort_values(
        ["peptide", "run"]
    ).to_csv(out_by_run, float_format="%.6f", index=False)

    if args.include_class_output:
        out_by_run_class = output_dir / "rmoi_and_ap_by_run_with_class.csv"
        df_long.sort_values(["class", "peptide", "run"]).to_csv(
            out_by_run_class, float_format="%.6f", index=False
        )
        print("Wrote:", out_by_run_class)

    stats = df_long.groupby("peptide").agg(
        RMOI_mean=("RMOI", "mean"),
        RMOI_std=("RMOI", "std"),
        aggregation_propensity_mean=("aggregation_propensity", "mean"),
        aggregation_propensity_std=("aggregation_propensity", "std"),
        n_runs_RMOI=("RMOI", "count"),
        n_runs_AP=("aggregation_propensity", "count"),
    )

    def pivot_metric(metric: str) -> pd.DataFrame:
        pvt = df_long.pivot_table(index="peptide", columns="run", values=metric, aggfunc="first")
        pvt.columns = [f"{metric}_{c}" for c in pvt.columns]
        return pvt

    wide_rmoi = pivot_metric("RMOI")
    wide_ap = pivot_metric("aggregation_propensity")

    summary = stats.join(wide_rmoi, how="left").join(wide_ap, how="left").sort_index()

    out_summary = output_dir / "rmoi_and_ap_summary_total.csv"
    summary.to_csv(out_summary, float_format="%.6f", index=True)

    print("Wrote:", out_summary)
    print("Wrote:", out_by_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
