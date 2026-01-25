from __future__ import annotations

import ast
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def parse_list_field(value) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            pass
        return [v.strip() for v in value.split(",") if v.strip()]
    return [value]


def read_sequences_txt(path: Path) -> list[str]:
    seqs: list[str] = []
    with open(path, "r") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            token = raw.split(",", 1)[0].split()[0]
            if token:
                seqs.append(token.upper())
    return seqs


def read_sequences_csv(path: Path, column: str | None = "sequence") -> list[str]:
    df = pd.read_csv(path)
    if column not in df.columns:
        column = df.columns[0]
    return df[column].dropna().astype(str).str.upper().tolist()


def aa_freqs(seq_list: Iterable[str], alphabet: str = AA_ALPHABET) -> np.ndarray:
    counts = {aa: 0 for aa in alphabet}
    total = 0
    for seq in seq_list:
        for ch in str(seq).upper():
            if ch in counts:
                counts[ch] += 1
                total += 1
    if total == 0:
        return np.zeros(len(alphabet), dtype=float)
    return np.array([counts[a] / total for a in alphabet], dtype=float)


def expand_ranges(ranges: dict[str, tuple[float, float, float]]) -> dict[str, list[float]]:
    return {k: np.arange(v[0], v[1], v[2]).tolist() for k, v in ranges.items()}


def enumerate_conditions(ranges: dict[str, tuple[float, float, float]], n_samples: int) -> list[dict]:
    feature_ranges = expand_ranges(ranges)
    combos = list(product(*feature_ranges.values()))
    keys = list(feature_ranges.keys())
    conds: list[dict] = []
    for combo in combos:
        params = dict(zip(keys, combo))
        for _ in range(n_samples):
            conds.append(params)
    return conds
