#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from common import DATA_PROCESSED, DEFAULT_CVAE_CHECKPOINT, RESULTS_DIR, load_split_indices
from cvae.models import CVAESimpleEnc
from cvae.utils import (
    CONDITION_LENGTH,
    MAX_SEQ_LENGTH,
    MAX_FASTA_LENGTH,
    ALPHABET,
    PAD_TOKEN_ID,
    EOS_ID,
    BOS_ID,
    idx_to_fasta,
    set_seed,
)

try:
    import editdistance
except Exception:
    editdistance = None

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

FEATURES_ALL = [
    "length",
    "is_assembled",
    "ap",
    "has_beta_sheet_content",
    "hydrophobic_moment",
    "net_charge",
]

COND_FEATURES = ["is_assembled", "ap", "has_beta_sheet_content", "hydrophobic_moment", "net_charge"]

AMINO_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ALPHABET)}
PAD_VAL = -1


def to_fasta(tokens: List[int]) -> str:
    fasta = ""
    for token in tokens:
        if token == EOS_ID:
            break
        if token in [0, 1, 3, 29, 30, 31, 32]:
            continue
        fasta += idx_to_fasta[token]
    return fasta


def encode_seq(s: str) -> np.ndarray:
    return np.array([AA_TO_INT.get(ch, len(AMINO_ALPHABET) - 1) for ch in s], dtype=np.int8)


def encode_pad_sequences(seqs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    lens = np.array([len(s) for s in seqs], dtype=np.int16)
    lmax = int(lens.max()) if len(lens) else 0
    codes = np.full((len(seqs), lmax), PAD_VAL, dtype=np.int16)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            codes[i, j] = AA_TO_INT.get(ch, len(AMINO_ALPHABET) - 1)
    return codes, lens


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def nw_identity_counts(a_codes: np.ndarray, b_codes: np.ndarray) -> Tuple[int, int]:
        na, nb = a_codes.size, b_codes.size
        score = np.empty((na + 1, nb + 1), dtype=np.int16)
        match = np.zeros((na + 1, nb + 1), dtype=np.int16)
        alen = np.zeros((na + 1, nb + 1), dtype=np.int16)

        gap = -1
        score[0, 0] = 0
        for i in range(1, na + 1):
            score[i, 0] = score[i - 1, 0] + gap
            alen[i, 0] = i
        for j in range(1, nb + 1):
            score[0, j] = score[0, j - 1] + gap
            alen[0, j] = j

        for i in range(1, na + 1):
            ai = a_codes[i - 1]
            for j in range(1, nb + 1):
                bj = b_codes[j - 1]
                sc_diag = score[i - 1, j - 1] + (1 if ai == bj else 0)
                mt_diag = match[i - 1, j - 1] + (1 if ai == bj else 0)
                ln_diag = alen[i - 1, j - 1] + 1

                sc_up = score[i - 1, j] + gap
                mt_up = match[i - 1, j]
                ln_up = alen[i - 1, j] + 1

                sc_left = score[i, j - 1] + gap
                mt_left = match[i, j - 1]
                ln_left = alen[i, j - 1] + 1

                sc = sc_diag
                mt = mt_diag
                ln = ln_diag
                if (sc_up > sc) or (sc_up == sc and (mt_up > mt or (mt_up == mt and ln_up < ln))):
                    sc = sc_up
                    mt = mt_up
                    ln = ln_up
                if (sc_left > sc) or (sc_left == sc and (mt_left > mt or (mt_left == mt and ln_left < ln))):
                    sc = sc_left
                    mt = mt_left
                    ln = ln_left

                score[i, j] = sc
                match[i, j] = mt
                alen[i, j] = ln

        return match[na, nb], alen[na, nb]

else:

    def nw_identity_counts(a_codes: np.ndarray, b_codes: np.ndarray) -> Tuple[int, int]:
        na, nb = a_codes.size, b_codes.size
        score = np.zeros((na + 1, nb + 1), dtype=int)
        match = np.zeros((na + 1, nb + 1), dtype=int)
        alen = np.zeros((na + 1, nb + 1), dtype=int)

        gap = -1
        for i in range(1, na + 1):
            score[i, 0] = score[i - 1, 0] + gap
            alen[i, 0] = i
        for j in range(1, nb + 1):
            score[0, j] = score[0, j - 1] + gap
            alen[0, j] = j

        for i in range(1, na + 1):
            ai = a_codes[i - 1]
            for j in range(1, nb + 1):
                bj = b_codes[j - 1]
                sc_diag = score[i - 1, j - 1] + (1 if ai == bj else 0)
                mt_diag = match[i - 1, j - 1] + (1 if ai == bj else 0)
                ln_diag = alen[i - 1, j - 1] + 1

                sc_up = score[i - 1, j] + gap
                mt_up = match[i - 1, j]
                ln_up = alen[i - 1, j] + 1

                sc_left = score[i, j - 1] + gap
                mt_left = match[i, j - 1]
                ln_left = alen[i, j - 1] + 1

                sc = sc_diag
                mt = mt_diag
                ln = ln_diag
                if (sc_up > sc) or (sc_up == sc and (mt_up > mt or (mt_up == mt and ln_up < ln))):
                    sc = sc_up
                    mt = mt_up
                    ln = ln_up
                if (sc_left > sc) or (sc_left == sc and (mt_left > mt or (mt_left == mt and ln_left < ln))):
                    sc = sc_left
                    mt = mt_left
                    ln = ln_left

                score[i, j] = sc
                match[i, j] = mt
                alen[i, j] = ln

        return match[na, nb], alen[na, nb]


def nw_percent_identity(a: str, b: str) -> float:
    a_codes = encode_seq(a)
    b_codes = encode_seq(b)
    matches, aln_len = nw_identity_counts(a_codes, b_codes)
    if aln_len == 0:
        return 0.0
    return float(matches) / float(aln_len)


def normalized_edit_dist(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if editdistance is not None:
        d = editdistance.eval(a, b)
    else:
        # basic Levenshtein fallback
        na, nb = len(a), len(b)
        dp = list(range(nb + 1))
        for i in range(1, na + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, nb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
        d = dp[nb]
    return d / max(len(a), len(b), 1)


def compute_exact_novelty_and_nn_edit(
    generated: List[str],
    training: List[str],
    length_window: int = 1,
) -> pd.DataFrame:
    train_by_len: dict[int, List[str]] = defaultdict(list)
    for t in training:
        train_by_len[len(t)].append(t)

    rows = []
    print("Calculating exact novelty + nearest-train normalized edit distance...")
    for seq in tqdm(generated, desc="Novelty/NN-edit"):
        length = len(seq)
        exact = seq in train_by_len[length]
        pool: list[str] = []
        for dlen in range(-length_window, length_window + 1):
            pool.extend(train_by_len.get(length + dlen, []))
        if not pool:
            nn = min((normalized_edit_dist(seq, t) for t in training))
        else:
            nn = min((normalized_edit_dist(seq, t) for t in pool))
        rows.append((seq, exact, nn))

    return pd.DataFrame(rows, columns=["sequence", "exact_match", "nn_dist"])


def compute_diversity_and_uniqueness(samples_df: pd.DataFrame, across_pairs_sample: int = 100000) -> dict:
    all_seqs = samples_df["sequence"].tolist()
    uniq_overall = 100.0 * (len(set(all_seqs)) / len(all_seqs))

    within_stats = {}
    print("Computing within-condition diversity...")
    for cond_idx, grp in tqdm(samples_df.groupby("cond_idx"), total=samples_df["cond_idx"].nunique()):
        seqs = grp["sequence"].tolist()
        dists = []
        for i in range(len(seqs)):
            si = seqs[i]
            for j in range(i + 1, len(seqs)):
                dists.append(normalized_edit_dist(si, seqs[j]))
        d = np.array(dists, dtype=float) if dists else np.array([np.nan])
        within_stats[cond_idx] = {
            "mean": float(np.nanmean(d)),
            "std": float(np.nanstd(d)),
            "median": float(np.nanmedian(d)),
            "uniq_pct": 100.0 * (len(set(seqs)) / len(seqs)),
        }

    print(f"Computing across-condition diversity on {across_pairs_sample} sampled pairs...")
    rng = np.random.default_rng(0)
    idxs = samples_df.index.values
    d_across = []
    tries = 0
    while len(d_across) < across_pairs_sample and tries < across_pairs_sample * 10:
        i, j = rng.integers(0, len(idxs), size=2)
        if i == j:
            tries += 1
            continue
        a = samples_df.iloc[i]
        b = samples_df.iloc[j]
        if a["cond_idx"] == b["cond_idx"]:
            tries += 1
            continue
        d_across.append(normalized_edit_dist(a["sequence"], b["sequence"]))
    d_across = np.array(d_across, dtype=float)
    across_stats = {
        "mean": float(np.mean(d_across)) if len(d_across) else float("nan"),
        "std": float(np.std(d_across)) if len(d_across) else float("nan"),
        "median": float(np.median(d_across)) if len(d_across) else float("nan"),
        "n_pairs": int(len(d_across)),
    }

    return {
        "uniqueness_overall_pct": uniq_overall,
        "within_condition": within_stats,
        "across_condition": across_stats,
    }


def _even_counts_over_k(n_total: int, k_min: int = 1, k_max: int = 6, rng: np.random.RandomState = np.random.RandomState(0)) -> Dict[int, int]:
    ks = list(range(k_min, k_max + 1))
    base = n_total // len(ks)
    rem = n_total % len(ks)
    counts = {k: base for k in ks}
    if rem > 0:
        bump = rng.choice(ks, size=rem, replace=False)
        for k in bump:
            counts[k] += 1
    return counts


def _random_k_subsets(features: List[str], k_list: List[int], rng: np.random.RandomState) -> List[List[str]]:
    out = []
    for k in k_list:
        out.append(rng.choice(features, size=k, replace=False).tolist())
    return out


def fit_gmm_per_length(
    train_df: pd.DataFrame,
    lengths: Iterable[int],
    n_components: int = 5,
    random_state: int = 42,
):
    models = {}
    for length in lengths:
        df_l = train_df.loc[train_df["length"] == length, COND_FEATURES].copy()
        if df_l.empty:
            raise ValueError(f"No training rows for length={length}")
        num_cols = ["ap", "hydrophobic_moment", "net_charge"]
        bin_cols = ["is_assembled", "has_beta_sheet_content"]
        ct = ColumnTransformer(
            [("num", SimpleImputer(strategy="median"), num_cols), ("bin", SimpleImputer(strategy="most_frequent"), bin_cols)]
        )
        x_imp = ct.fit_transform(df_l)
        scaler = MinMaxScaler((0, 1))
        x_scaled = scaler.fit_transform(x_imp)
        gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
        gmm.fit(x_scaled)
        models[length] = {
            "imputer": ct,
            "scaler": scaler,
            "gmm": gmm,
            "num_cols": num_cols,
            "bin_cols": bin_cols,
        }
    return models


def sample_in_distribution_conditions(
    models: Dict[int, dict],
    per_length_counts: Dict[int, int],
    rng: np.random.RandomState = np.random.RandomState(0),
) -> List[dict]:
    conds = []
    per_l_params = {}
    for length, count in per_length_counts.items():
        if count <= 0:
            continue
        pack = models[length]
        gmm, scaler = pack["gmm"], pack["scaler"]
        num, binf = pack["num_cols"], pack["bin_cols"]

        xs, _ = gmm.sample(count)
        x_imp = scaler.inverse_transform(np.clip(xs, 0, 1))

        params_list = []
        for row in x_imp:
            row = np.clip(row, 0, None)
            vals = {}
            off = 0
            for i, col in enumerate(num):
                vals[col] = float(row[off + i])
            off += len(num)
            for i, col in enumerate(binf):
                vals[col] = float(1.0 if row[off + i] >= 0.5 else 0.0)
            vals_full = {"length": int(length)}
            vals_full.update(vals)
            params_list.append(vals_full)
        per_l_params[length] = params_list

    ks = list(range(1, 7))
    total = sum(len(v) for v in per_l_params.values())
    counts_by_k = _even_counts_over_k(total)
    k_list = []
    for k in ks:
        k_list += [k] * counts_by_k[k]
    rng.shuffle(k_list)

    ptr = 0
    for length in sorted(per_l_params.keys()):
        params_list = per_l_params[length]
        k_slice = k_list[ptr : ptr + len(params_list)]
        ptr += len(params_list)
        used_subsets = _random_k_subsets(FEATURES_ALL, k_slice, rng)
        for row_params, used_feats in zip(params_list, used_subsets):
            conds.append(
                {
                    "params": row_params,
                    "length": length,
                    "used_features": used_feats,
                    "ood_type": "in_dist",
                }
            )
    return conds


def compute_rare_values(train_df: pd.DataFrame, length: int) -> dict:
    df_l = train_df.loc[train_df["length"] == length]
    freq = df_l["has_beta_sheet_content"].value_counts(dropna=True)
    rare_beta = float(freq.idxmin()) if len(freq) else 1.0

    q05, q95 = df_l["hydrophobic_moment"].quantile(0.05), df_l["hydrophobic_moment"].quantile(0.95)
    tail_low = (df_l["hydrophobic_moment"] <= q05).sum()
    tail_high = (df_l["hydrophobic_moment"] >= q95).sum()
    rare_hm = float(q05 if tail_low < tail_high else q95)

    return {"has_beta_sheet_content": rare_beta, "hydrophobic_moment": rare_hm}


def sample_ood_conditions(
    train_df: pd.DataFrame,
    lengths: Iterable[int],
    total_ood: int = 10,
    rng: np.random.RandomState = np.random.RandomState(1),
) -> List[dict]:
    lengths = list(lengths)
    total_ood = max(2, total_ood)
    half = total_ood // 2
    idx_beta = rng.choice(lengths, size=half, replace=True)
    idx_hm = rng.choice(lengths, size=total_ood - half, replace=True)

    conds = []
    for length in idx_beta:
        rare = compute_rare_values(train_df, length)
        params = {
            "length": int(length),
            "has_beta_sheet_content": rare["has_beta_sheet_content"],
            "is_assembled": float(train_df.loc[train_df.length == length, "is_assembled"].median()),
            "ap": float(train_df.loc[train_df.length == length, "ap"].median()),
            "hydrophobic_moment": float(train_df.loc[train_df.length == length, "hydrophobic_moment"].median()),
            "net_charge": float(train_df.loc[train_df.length == length, "net_charge"].median()),
        }
        k = int(rng.randint(1, 7))
        used = set(["has_beta_sheet_content"])
        if k > 1:
            pool = [f for f in FEATURES_ALL if f not in used]
            used.update(rng.choice(pool, size=k - 1, replace=False).tolist())
        conds.append(
            {"params": params, "length": length, "used_features": sorted(list(used)), "ood_type": "rare_beta"}
        )

    for length in idx_hm:
        rare = compute_rare_values(train_df, length)
        params = {
            "length": int(length),
            "hydrophobic_moment": rare["hydrophobic_moment"],
            "is_assembled": float(train_df.loc[train_df.length == length, "is_assembled"].median()),
            "ap": float(train_df.loc[train_df.length == length, "ap"].median()),
            "has_beta_sheet_content": float(train_df.loc[train_df.length == length, "has_beta_sheet_content"].median()),
            "net_charge": float(train_df.loc[train_df.length == length, "net_charge"].median()),
        }
        k = int(rng.randint(1, 7))
        used = set(["hydrophobic_moment"])
        if k > 1:
            pool = [f for f in FEATURES_ALL if f not in used]
            used.update(rng.choice(pool, size=k - 1, replace=False).tolist())
        conds.append(
            {"params": params, "length": length, "used_features": sorted(list(used)), "ood_type": "rare_hm"}
        )

    return conds


def generate_cond_mask_vectors(device, **kwargs):
    condition = torch.zeros(len(FEATURES_ALL), dtype=torch.float, device=device)
    mask = torch.zeros(len(FEATURES_ALL), dtype=torch.float, device=device)

    for idx, feature in enumerate(FEATURES_ALL):
        value = kwargs.get(feature, None)
        if value is not None:
            condition[idx] = value if feature != "length" else value / MAX_FASTA_LENGTH
            mask[idx] = 1.0
    return condition, mask


def sample_from_prior_ar(model, condition, mask, temperature: float = 1.0, n: int = 1):
    m = model.module if hasattr(model, "module") else model
    m.eval()

    if n == 1:
        condition = condition.unsqueeze(0)
        mask = mask.unsqueeze(0)
    else:
        condition = condition.repeat(n, 1)
        mask = mask.repeat(n, 1)

    device = next(m.parameters()).device
    condition, mask = condition.to(device), mask.to(device)
    summary = m.compute_summary(condition, mask)
    prior_mu, prior_logvar = m.compute_prior(summary)
    z = m.reparameterize(prior_mu, prior_logvar)

    seq = torch.full((condition.size(0), 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(condition.size(0), dtype=torch.bool, device=device)
    outputs = torch.full((condition.size(0), MAX_SEQ_LENGTH), PAD_TOKEN_ID, dtype=torch.long, device=device)

    for t in range(MAX_SEQ_LENGTH):
        logits = m.decode(z, seq, summary)
        next_logits = logits[:, -1, :]
        if temperature == 1.0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.nn.functional.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        outputs[:, t] = next_token.squeeze(1)
        seq = torch.cat([seq, next_token], dim=1)
        finished = finished | (next_token.squeeze(1) == EOS_ID)
        if finished.all():
            break

    outs = []
    for row in outputs.tolist():
        if EOS_ID in row:
            cut = row.index(EOS_ID)
            toks = row[:cut]
        else:
            toks = row
        fasta = to_fasta(toks)
        outs.append((fasta, len(fasta)))
    return outs


def generate_for_conditions(
    cvae_model,
    device,
    cond_list: List[dict],
    n_per_cond: int = 100,
    temperature: float = 1.0,
) -> List[dict]:
    all_samples = []
    for idx, rec in enumerate(tqdm(cond_list, desc="Generating")):
        params_full = rec["params"]
        used = set(rec.get("used_features", FEATURES_ALL))
        kwargs = {k: v for k, v in params_full.items() if k in used}

        cond_vec, mask_vec = generate_cond_mask_vectors(device=device, **kwargs)
        outs = sample_from_prior_ar(cvae_model, cond_vec, mask_vec, temperature=temperature, n=n_per_cond)
        for fasta, length in outs:
            all_samples.append(
                {
                    "cond_idx": idx,
                    "sequence": fasta,
                    "length": length,
                    "target_length": params_full["length"],
                    "params": params_full,
                    "used_features": sorted(list(used)),
                    "ood_type": rec.get("ood_type", "in_dist"),
                }
            )
    return all_samples


def annotate_similarity_columns_with_progress(
    samples_df: pd.DataFrame,
    train_sequences: List[str],
    block_size_sim_train: int = 512,
    block_size_sim_gen: int = 256,
    compute_sim_gen_all: bool = True,
    compute_sim_gen_within: bool = True,
) -> pd.DataFrame:
    df = samples_df.copy().reset_index(drop=True)

    train_codes, train_lens = encode_pad_sequences(train_sequences)

    sim_train = []
    for i in tqdm(range(0, len(df), block_size_sim_train), desc="sim_train"):
        chunk = df["sequence"].iloc[i : i + block_size_sim_train].tolist()
        for seq in chunk:
            best = 0.0
            for t in train_sequences:
                best = max(best, nw_percent_identity(seq, t))
            sim_train.append(best)
    df["sim_train"] = sim_train

    if compute_sim_gen_all or compute_sim_gen_within:
        sim_gen_all = []
        sim_gen_within = []
        for cond_idx, grp in tqdm(df.groupby("cond_idx"), desc="sim_gen"):
            seqs = grp["sequence"].tolist()
            for i, seq in enumerate(seqs):
                best_all = 0.0
                best_within = 0.0
                for j, seq2 in enumerate(seqs):
                    if i == j:
                        continue
                    score = nw_percent_identity(seq, seq2)
                    best_all = max(best_all, score)
                    best_within = max(best_within, score)
                sim_gen_all.append(best_all)
                sim_gen_within.append(best_within)
        if compute_sim_gen_all:
            df["sim_gen_all"] = sim_gen_all
        if compute_sim_gen_within:
            df["sim_gen_within"] = sim_gen_within

    return df


def even_counts_across_lengths(lengths: List[int], total: int) -> Dict[int, int]:
    base = total // len(lengths)
    rem = total % len(lengths)
    counts = {length: base for length in lengths}
    for length in sorted(lengths)[:rem]:
        counts[length] += 1
    return counts


def run_full_evaluation(
    cvae_model,
    device,
    train_df: pd.DataFrame,
    train_sequences: List[str],
    target_total_in_dist_conditions: int = 100,
    per_length_counts: Optional[Dict[int, int]] = None,
    total_ood_conditions: int = 10,
    n_per_condition: int = 100,
    gmm_components: int = 5,
    random_state: int = 42,
    compute_sim_gen: bool = True,
):
    lengths = list(range(5, MAX_FASTA_LENGTH + 1))

    print("Fitting per-length GMMs on training conditions...")
    models = fit_gmm_per_length(train_df, lengths, n_components=gmm_components, random_state=random_state)

    if per_length_counts is None:
        per_length_counts = even_counts_across_lengths(lengths, target_total_in_dist_conditions)
    print("In-distribution per-length condition counts:", per_length_counts, " Total =", sum(per_length_counts.values()))

    print("Sampling in-distribution conditions...")
    in_dist_conds = sample_in_distribution_conditions(models, per_length_counts)

    print(f"Sampling OOD conditions (total={total_ood_conditions})...")
    ood_conds = sample_ood_conditions(train_df, lengths, total_ood=total_ood_conditions)

    print("Generating in-distribution sequences...")
    gen_in = generate_for_conditions(cvae_model, device, in_dist_conds, n_per_cond=n_per_condition)
    print("Generating OOD sequences...")
    gen_ood = generate_for_conditions(cvae_model, device, ood_conds, n_per_cond=n_per_condition)

    samples_df = pd.DataFrame(gen_in + gen_ood)
    print("Generation complete. Shape:", samples_df.shape)

    if compute_sim_gen:
        print("Computing similarity columns...")
        samples_df = annotate_similarity_columns_with_progress(
            samples_df,
            train_sequences=list(train_sequences),
            block_size_sim_train=512,
            block_size_sim_gen=256,
            compute_sim_gen_all=True,
            compute_sim_gen_within=True,
        )

    print("Computing diversity and uniqueness...")
    div_stats = compute_diversity_and_uniqueness(samples_df[samples_df["ood_type"] == "in_dist"], across_pairs_sample=100000)

    print("Computing novelty and nearest-train edit distance...")
    novelty_df = compute_exact_novelty_and_nn_edit(
        samples_df["sequence"].tolist(),
        list(train_sequences),
        length_window=1,
    )

    samples_df = samples_df.reset_index(drop=True)
    samples_df["exact_match"] = novelty_df["exact_match"].values
    samples_df["nn_dist"] = novelty_df["nn_dist"].values

    within_rows = []
    for cid, st in div_stats["within_condition"].items():
        within_rows.append({"cond_idx": cid, **st})
    within_df = pd.DataFrame(within_rows).sort_values("cond_idx")

    cond_meta = []
    for i, rec in enumerate(in_dist_conds):
        r = {"cond_idx": i, "ood_type": "in_dist"}
        r.update(rec["params"])
        cond_meta.append(r)
    off = len(in_dist_conds)
    for j, rec in enumerate(ood_conds):
        r = {"cond_idx": off + j, "ood_type": "OOD"}
        r.update(rec["params"])
        cond_meta.append(r)
    cond_meta_df = pd.DataFrame(cond_meta)

    results = {
        "samples_df": samples_df,
        "novelty_df": novelty_df,
        "diversity_stats": div_stats,
        "within_df": within_df,
        "cond_meta_df": cond_meta_df,
    }
    return results


def parse_per_length_counts(text: str) -> Dict[int, int]:
    counts = {}
    if not text:
        return counts
    for part in text.split(","):
        if not part:
            continue
        length_str, count_str = part.split(":")
        counts[int(length_str.strip())] = int(count_str.strip())
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CVAE evaluation and save results.")
    parser.add_argument("--data-csv", type=str, default=str(DATA_PROCESSED / "merged_all.csv"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CVAE_CHECKPOINT),
    )
    parser.add_argument(
        "--output-pkl",
        type=str,
        default=str(RESULTS_DIR / "cvae_evaluation_results.pkl"),
    )
    parser.add_argument(
        "--output-fasta",
        type=str,
        default=str(RESULTS_DIR / "validate_conditioning_peptides.fst"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-total-in-dist", type=int, default=100)
    parser.add_argument("--per-length-counts", type=str, default="")
    parser.add_argument("--total-ood", type=int, default=10)
    parser.add_argument("--n-per-condition", type=int, default=100)
    parser.add_argument("--gmm-components", type=int, default=5)
    parser.add_argument("--skip-sim-gen", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_csv = Path(args.data_csv).expanduser().resolve()
    df = pd.read_csv(data_csv, keep_default_na=False, na_values=[""])
    split_idx = load_split_indices()
    if split_idx:
        train_df = df.iloc[split_idx["train"]].copy()
        print("Using precomputed splits from data/splits")
    else:
        train_val_df, _ = train_test_split(df, test_size=0.1, stratify=df["length"], random_state=42)
        train_df, _ = train_test_split(train_val_df, test_size=0.1, stratify=train_val_df["length"], random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.device_count())

    cvae_model = CVAESimpleEnc(
        encoder_hidden_dim=256,
        num_encoder_layers=2,
        vocab_size=len(ALPHABET),
        latent_dim=24,
        cond_dim=CONDITION_LENGTH,
        max_seq_length=MAX_SEQ_LENGTH,
        decoder_hidden_dim=256,
        num_decoder_layers=2,
        nhead=8,
        dropout=0.1,
    )

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    cvae_model.load_state_dict(state)

    cvae_model.eval()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        cvae_model = nn.DataParallel(cvae_model)
    cvae_model.to(device)

    per_length_counts = parse_per_length_counts(args.per_length_counts)
    per_length_counts = per_length_counts or None

    train_seqs = set(train_df["peptide"].tolist())

    results = run_full_evaluation(
        cvae_model=cvae_model,
        device=device,
        train_df=train_df,
        train_sequences=list(train_seqs),
        target_total_in_dist_conditions=args.target_total_in_dist,
        per_length_counts=per_length_counts,
        total_ood_conditions=args.total_ood,
        n_per_condition=args.n_per_condition,
        gmm_components=args.gmm_components,
        random_state=args.seed,
        compute_sim_gen=not args.skip_sim_gen,
    )

    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        import pickle

        pickle.dump(results, f)
    print("Wrote", output_pkl)

    output_fasta = Path(args.output_fasta).expanduser().resolve()
    with open(output_fasta, "w") as f:
        for i, seq in enumerate(results["samples_df"]["sequence"], start=1):
            f.write(f">peptide_{i}\n{seq}\n")
    print("Wrote", output_fasta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
