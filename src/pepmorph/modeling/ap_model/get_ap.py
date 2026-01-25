#!/usr/bin/env python

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from copy import deepcopy

from common import REPO_ROOT  # adds repo root + model dir to sys.path
from classifier.models import PeptidePredictor
from cvae.utils import (
    MAX_SEQ_LENGTH,
    ALPHABET,
    esm_model_pretrained,
    convert_and_pad,
)


ARTIFACTS_DIR = REPO_ROOT / "artifacts"
VALIDATION_DIR = ARTIFACTS_DIR / "validation"
DEFAULT_CHECKPOINT = ARTIFACTS_DIR / "models" / "ap_model" / "peptide_predictor.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random peptides and score AP/SA.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )
    parser.add_argument("--output-dir", type=str, default=str(VALIDATION_DIR / "gen_peptides"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=4800)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--min-len-final", type=int, default=5)
    parser.add_argument("--ap-threshold", type=float, default=1.8)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_random_sequences(count: int, min_len: int, max_len: int, rng: np.random.Generator) -> list[str]:
    seqs = []
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    for _ in range(count):
        length = rng.integers(min_len, max_len + 1)
        seq = "".join(rng.choice(alphabet, size=length))
        seqs.append(seq)
    return seqs


def score_sequences(model, sequences: list[str], batch_size: int, device: torch.device):
    results = []
    min_ap, max_ap = 0.959986, 2.89703

    for start_idx in tqdm(range(0, len(sequences), batch_size), desc="Scoring batches"):
        chunk = sequences[start_idx : start_idx + batch_size]
        data = [(f"peptide_{start_idx + i}", seq) for i, seq in enumerate(chunk)]
        tokens = convert_and_pad(data, seq_length=MAX_SEQ_LENGTH).to(device)

        with torch.no_grad():
            ap_preds, cls_preds = model(tokens)

        for seq, ap_pred, cls_pred in zip(chunk, ap_preds, cls_preds):
            ap_val = ap_pred.item() * (max_ap - min_ap) + min_ap
            cls_prob = torch.sigmoid(cls_pred).item()
            results.append((seq, len(seq), ap_val, cls_prob))

    return pd.DataFrame(results, columns=["sequence", "length", "ap", "clf"])


def main() -> int:
    args = parse_args()
    set_random_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.device_count())

    model = PeptidePredictor(deepcopy(esm_model_pretrained), alphabet=ALPHABET)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    rng = np.random.default_rng(args.seed)

    initial_seqs = generate_random_sequences(args.n_samples, args.min_len, args.max_len, rng)
    results_df = score_sequences(model, initial_seqs, args.batch_size, device)

    results_df = results_df[results_df["length"] >= args.min_len_final]

    remaining = args.n_samples - len(results_df)
    extra_df = pd.DataFrame(columns=results_df.columns)
    if remaining > 0:
        extra_seqs = generate_random_sequences(remaining, args.min_len_final, args.max_len, rng)
        extra_df = score_sequences(model, extra_seqs, args.batch_size, device)

    generated_path = output_dir / "generated_random_init.txt"
    with open(generated_path, "w") as f:
        for seq in results_df["sequence"]:
            f.write(f"{seq}\n")
        for seq in extra_df["sequence"]:
            f.write(f"{seq}\n")

    filtered_df = results_df[results_df["ap"] > args.ap_threshold]
    filtered_extra_df = extra_df[extra_df["ap"] > args.ap_threshold]

    filtered_path = output_dir / "filtered_ap_peptides_random.txt"
    with open(filtered_path, "w") as f:
        for _, row in filtered_df.iterrows():
            f.write(f"{row['sequence']},{row['length']},{row['ap']},{row['clf']}\n")
        for _, row in filtered_extra_df.iterrows():
            f.write(f"{row['sequence']},{row['length']},{row['ap']},{row['clf']}\n")

    extra_fst = output_dir / "random_peptides_filtered_extra.fst"
    new_filtered_seqs = filtered_extra_df["sequence"].tolist()
    with open(extra_fst, "w") as f:
        for i, seq in enumerate(new_filtered_seqs):
            f.write(f">random_peptide_{i}\n{seq}\n")

    print("Wrote", generated_path)
    print("Wrote", filtered_path)
    print("Wrote", extra_fst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
