#!/usr/bin/env python

from __future__ import annotations

import argparse
import random
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from common import DATA_PROCESSED, DATA_RAW, REPO_ROOT
from ap_model.datasets import PeptidePredictorDataset
from ap_model.models import PeptidePredictor

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_CHECKPOINT = ARTIFACTS_DIR / "models" / "ap_model" / "peptide_predictor.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory analysis for AP/SA predictor.")
    parser.add_argument(
        "--data-csv",
        type=str,
        default=str(DATA_PROCESSED / "merged_all.csv"),
    )
    parser.add_argument(
        "--experimental-csv",
        type=str,
        default=str(DATA_RAW / "experimental.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-common", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_stratification_key(length: int) -> str:
    return str(length)


def split_dataset(df, stratify_col, test_size=0.1, val_size=0.1, random_state=42):
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[stratify_col], random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, stratify=train_val_df[stratify_col], random_state=random_state
    )
    return train_df, val_df, test_df


def convert_and_pad(data, seq_length, batch_converter, padding_idx):
    _, _, tokens = batch_converter(data)
    current_len = tokens.size(1)
    if current_len < seq_length:
        pad_length = seq_length - current_len
        padding = torch.full((tokens.size(0), pad_length), padding_idx, dtype=tokens.dtype)
        tokens = torch.cat([tokens, padding], dim=1)
    return tokens


def esm_collate_fn(batch, batch_converter, seq_length, padding_idx):
    data = [(f"peptide_{i}", seq) for i, (seq, _) in enumerate(batch)]
    tokens = convert_and_pad(data, seq_length=seq_length, batch_converter=batch_converter, padding_idx=padding_idx)
    targets = torch.tensor([target for _, target in batch])
    return tokens, targets


def validate_model(model, cls_loader, reg_loader, device):
    model.eval()
    all_cls_preds = []
    all_cls_labels = []
    all_reg_preds = []
    all_reg_labels = []

    accuracy = f1 = auc = mse = mae = r2 = float("nan")

    with torch.no_grad():
        for tokens, targets in cls_loader:
            tokens = tokens.to(device)
            targets = targets.to(device)
            _, cls_pred = model(tokens)
            all_cls_preds.append(cls_pred.cpu())
            all_cls_labels.append(targets.cpu())

        for tokens, targets in reg_loader:
            tokens = tokens.to(device)
            targets = targets.to(device)
            ap_pred, _ = model(tokens)
            all_reg_preds.append(ap_pred.cpu())
            all_reg_labels.append(targets.cpu())

    if cls_loader:
        all_cls_preds = torch.cat(all_cls_preds, dim=0).numpy().squeeze()
        all_cls_labels = torch.cat(all_cls_labels, dim=0).numpy().squeeze()

        cls_pred_labels = (all_cls_preds > 0.5).astype(int)
        accuracy = accuracy_score(all_cls_labels, cls_pred_labels)
        f1 = f1_score(all_cls_labels, cls_pred_labels)
        try:
            auc = roc_auc_score(all_cls_labels, all_cls_preds)
        except Exception:
            auc = float("nan")

    if reg_loader:
        all_reg_preds = torch.cat(all_reg_preds, dim=0).numpy().squeeze()
        all_reg_labels = torch.cat(all_reg_labels, dim=0).numpy().squeeze()

        mse = np.mean((all_reg_preds - all_reg_labels) ** 2)
        mae = np.mean(np.abs(all_reg_preds - all_reg_labels))
        ss_res = np.sum((all_reg_preds - all_reg_labels) ** 2)
        ss_tot = np.sum((all_reg_labels - np.mean(all_reg_labels)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return accuracy, f1, auc, mse, mae, r2, all_cls_preds, all_cls_labels, all_reg_preds, all_reg_labels


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.device_count())

    max_seq_length = 12
    esm_model_pretrained, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=max_seq_length - 2)

    df = pd.read_csv(args.data_csv, keep_default_na=False, na_values=[""])
    regression_df = df[["peptide", "length", "ap"]].dropna(subset=["ap"])
    classification_df = df[["peptide", "length", "is_assembled"]].dropna(subset=["is_assembled"])

    classification_df["stratify_key"] = classification_df["length"].apply(get_stratification_key)
    regression_df["stratify_key"] = regression_df["length"].apply(get_stratification_key)

    train_cls, val_cls, test_cls = split_dataset(classification_df, "stratify_key", random_state=args.seed)
    train_reg, val_reg, test_reg = split_dataset(regression_df, "stratify_key", random_state=args.seed)

    collate = lambda batch: esm_collate_fn(batch, batch_converter, max_seq_length, alphabet.padding_idx)

    val_cls_loader = DataLoader(PeptidePredictorDataset(val_cls, task="classification"), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_cls_loader = DataLoader(PeptidePredictorDataset(test_cls, task="classification"), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    val_reg_loader = DataLoader(PeptidePredictorDataset(val_reg, task="regression"), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_reg_loader = DataLoader(PeptidePredictorDataset(test_reg, task="regression"), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = PeptidePredictor(esm_model_pretrained, alphabet, hidden_dim=128, dropout=0.10)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    accuracy, f1, auc, mse, mae, r2, *_ = validate_model(model, val_cls_loader, val_reg_loader, device)
    print("Validation Metrics:")
    print(f"Classification -> Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print(f"Regression     -> MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

    accuracy, f1, auc, mse, mae, r2, *_ = validate_model(model, test_cls_loader, test_reg_loader, device)
    print("Test Metrics:")
    print(f"Classification -> Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print(f"Regression     -> MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

    experimental_df = pd.read_csv(args.experimental_csv, keep_default_na=False, na_values=[""], sep=";")
    experimental_df["len"] = experimental_df["peptide"].apply(len)
    experimental_df = experimental_df.rename(columns={"label": "is_assembled"})
    experimental_df["stratify_key"] = experimental_df["len"].apply(get_stratification_key)
    experimental_df = experimental_df[experimental_df["len"] <= max_seq_length - 2]

    experimental_loader = DataLoader(
        PeptidePredictorDataset(experimental_df, task="classification"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    accuracy, f1, auc, *_ = validate_model(model, experimental_loader, [], device)
    print("Experimental Data Metrics:")
    print(f"Classification -> Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    _, _, _, _, _, _, all_cls_preds, _, _, _ = validate_model(model, experimental_loader, [], device)
    experimental_df["predicted_is_assembled"] = all_cls_preds

    peptides_in_experimental = set(experimental_df["peptide"])
    peptides_in_regression = set(regression_df["peptide"])
    common_peptides = peptides_in_experimental.intersection(peptides_in_regression)
    print(f"Number of common peptides in experimental and regression data: {len(common_peptides)}")

    common_peptides_df = regression_df[regression_df["peptide"].isin(common_peptides)]
    common_peptides_df = common_peptides_df[["peptide", "ap"]]
    common_peptides_df = common_peptides_df.merge(
        experimental_df[["peptide", "is_assembled", "predicted_is_assembled"]],
        on="peptide",
    )
    common_peptides_df = common_peptides_df[["peptide", "is_assembled", "predicted_is_assembled", "ap"]]

    if args.output_common:
        common_peptides_df.to_csv(args.output_common, index=False)
        print("Wrote", args.output_common)
    else:
        print("Common peptides in experimental and classification data:")
        print(common_peptides_df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
