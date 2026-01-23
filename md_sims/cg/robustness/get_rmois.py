#!/usr/bin/env python3
import os, re, subprocess
import pandas as pd

# === CONFIGURE THESE ===
BASE_DIR  = "./"
SCRIPT    = "compute_rmoi.py"  # script that prints a line starting with "Ratio of principal moments"
CUTOFF_NM = 0.6

def parse_sasa(xvg_path):
    """
    Read area.xvg, skip header lines starting with '#' or '@'.
    Parse rows as [time, area], then return (mean of first 2 areas) / (mean of last 2 areas).
    Falls back gracefully if fewer than 2 rows exist.
    """
    data = []
    with open(xvg_path, "r") as f:
        for L in f:
            L = L.strip()
            if not L or L.startswith(("#", "@")):
                continue
            parts = L.split()
            if len(parts) >= 2:
                try:
                    t, a = float(parts[0]), float(parts[1])
                    data.append((t, a))
                except ValueError:
                    continue
    if not data:
        return None
    if len(data) == 1:
        return data[0][1] / data[0][1] if data[0][1] != 0 else None
    # average first 2 and last 2
    first = sum(a for _, a in data[:2]) / min(2, len(data))
    last  = sum(a for _, a in data[-2:]) / min(2, len(data))
    if last == 0:
        return None
    return first / last

def find_run_dirs(morph_dir):
    """Return sorted list of subdirs matching run_* (numeric sort if possible)."""
    runs = []
    for name in os.listdir(morph_dir):
        full = os.path.join(morph_dir, name)
        if os.path.isdir(full) and name.startswith("run_"):
            runs.append(name)
    # sort by trailing integer if present, else lexicographically
    def sort_key(s):
        m = re.search(r"(\d+)$", s)
        return (0, int(m.group(1))) if m else (1, s)
    return sorted(runs, key=sort_key)

def run_compute_rmoi(gro, xtc, cutoff_nm):
    """Call external script and parse 'Ratio of principal moments ... <value>' from stdout."""
    cmd = ["python3", SCRIPT, "-g", gro, "-x", xtc, "-c", str(cutoff_nm)]
    out = subprocess.check_output(cmd, text=True).splitlines()
    for L in out:
        if L.startswith("Ratio of principal moments"):
            try:
                return float(L.split()[-1])
            except ValueError:
                pass
    return None

records = []
for type_ in ["original", "martini2", "bigger_box_1230"]:
    base_dir = os.path.join(BASE_DIR, type_)
    print(f"Processing folder: {base_dir}")
    if not os.path.isdir(base_dir):
        continue

    for run_name in find_run_dirs(base_dir):
        run_dir = os.path.join(base_dir, run_name)
        for peptide in sorted(os.listdir(run_dir)):
            pep_dir = os.path.join(run_dir, peptide)
            if not os.path.isdir(pep_dir):
                continue

            gro = os.path.join(pep_dir, "peptide-cg.gro")
            xtc = os.path.join(pep_dir, "trajout.xtc")
            if not (os.path.isfile(gro) and os.path.isfile(xtc)):
                continue

            ratio = run_compute_rmoi(gro, xtc, CUTOFF_NM)
            if ratio is None:
                raise RuntimeError(f"[{type_}/{run_name}/{peptide}] could not parse RMOI")

            area_xvg = os.path.join(pep_dir, "area.xvg")
            area_xvg = os.path.join(pep_dir, "sasa.xvg") if not os.path.isfile(area_xvg) else area_xvg
            agg_prop = parse_sasa(area_xvg) if os.path.isfile(area_xvg) else None

            records.append({
                #"morphology":             morphology,
                "type":                   type_,        # e.g., bigger_box
                "peptide":                peptide,
                "run":                    run_name,     # e.g., run_1
                "RMOI":                   ratio,
                "aggregation_propensity": agg_prop,
            })

# --- build per-run long dataframe ---
df_long = pd.DataFrame.from_records(records)
if df_long.empty:
    raise SystemExit("No records found. Check folder structure and file names.")

# --- pivot to wide columns per run and add mean/std per peptide ---
index_cols = ["peptide", "type"]

def pivot_metric(metric):
    pvt = df_long.pivot_table(index=index_cols, columns="run", values=metric, aggfunc="first")
    # flatten columns: metric_run_1, ...
    pvt.columns = [f"{metric}_{c}" for c in pvt.columns]
    return pvt

wide_rmoi = pivot_metric("RMOI")
wide_ap   = pivot_metric("aggregation_propensity")

stats = df_long.groupby(index_cols).agg(
    RMOI_mean=("RMOI", "mean"),
    RMOI_std =("RMOI", "std"),
    aggregation_propensity_mean=("aggregation_propensity", "mean"),
    aggregation_propensity_std =("aggregation_propensity", "std"),
    n_runs_RMOI=("RMOI", "count"),
    n_runs_AP=("aggregation_propensity", "count"),
)

summary = (
    stats
    .join(wide_rmoi, how="left")
    .join(wide_ap,   how="left")
    .sort_index()
)

# write a single CSV that includes per-run columns + mean/std
out_path = os.path.join(BASE_DIR, "rmoi_and_ap_summary_total.csv")
summary.to_csv(out_path, float_format="%.6f", index=True)

# (Optional) also keep the by-run long table for debugging/traceability:
out_by_run = os.path.join(BASE_DIR, "rmoi_and_ap_by_run_total.csv")
df_long.sort_values(index_cols + ["run"]).to_csv(out_by_run, float_format="%.6f", index=False)

print(f"Wrote:\n  {out_path}\n  {out_by_run}")
