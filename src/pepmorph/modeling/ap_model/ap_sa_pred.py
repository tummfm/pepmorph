#!/usr/bin/env python

from __future__ import annotations

import argparse
import random
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import DATA_PROCESSED, REPO_ROOT
from ap_model.datasets import PeptidePredictorDataset
from ap_model.models import PeptidePredictor

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_SAVE_PATH = ARTIFACTS_DIR / "models" / "ap_model" / "peptide_predictor.pt"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULTS = {
    "data_csv": str(DATA_PROCESSED / "merged_all.csv"),
    "batch_size": 1024,
    "epochs": 5,
    "lr": 1e-3,
    "max_seq_length": 12,
    "hidden_dim": 128,
    "dropout": 0.1,
    "save_path": str(DEFAULT_SAVE_PATH),
    "seed": 42,
    "splits_dir": str(REPO_ROOT / "data" / "splits"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AP/SA predictor.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--data-csv",
        type=str,
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text())
    return payload or {}


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    overrides = {
        "data_csv": args.data_csv,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_seq_length": args.max_seq_length,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "save_path": args.save_path,
        "seed": args.seed,
        "splits_dir": args.splits_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_indices(splits_dir: Path) -> dict[str, list[int]] | None:
    train_path = splits_dir / "train_idx.txt"
    val_path = splits_dir / "val_idx.txt"
    test_path = splits_dir / "test_idx.txt"
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        return None

    def _read(path: Path) -> list[int]:
        return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]

    return {"train": _read(train_path), "val": _read(val_path), "test": _read(test_path)}


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


def main() -> int:
    args = parse_args()
    cfg = DEFAULTS.copy()
    cfg.update(load_config(Path(args.config)))
    cfg = apply_overrides(cfg, args)

    data_csv = resolve_path(cfg["data_csv"], REPO_ROOT)
    splits_dir = resolve_path(cfg["splits_dir"], REPO_ROOT)
    save_path = resolve_path(cfg["save_path"], REPO_ROOT)
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    lr = float(cfg["lr"])
    max_seq_length = int(cfg["max_seq_length"])
    hidden_dim = int(cfg["hidden_dim"])
    dropout = float(cfg["dropout"])
    seed = int(cfg["seed"])

    set_seed(seed)

    esm_model_pretrained, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=max_seq_length - 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.device_count())

    df = pd.read_csv(data_csv, keep_default_na=False, na_values=[""])
    regression_df = df[["peptide", "length", "ap"]].dropna(subset=["ap"])
    classification_df = df[["peptide", "length", "is_assembled"]].dropna(subset=["is_assembled"])

    classification_df["stratify_key"] = classification_df["length"].apply(get_stratification_key)
    regression_df["stratify_key"] = regression_df["length"].apply(get_stratification_key)

    split_idx = load_split_indices(splits_dir)
    if split_idx:
        train_idx = set(split_idx["train"])
        val_idx = set(split_idx["val"])
        test_idx = set(split_idx["test"])

        train_cls = classification_df.loc[classification_df.index.intersection(train_idx)]
        val_cls = classification_df.loc[classification_df.index.intersection(val_idx)]
        test_cls = classification_df.loc[classification_df.index.intersection(test_idx)]

        train_reg = regression_df.loc[regression_df.index.intersection(train_idx)]
        val_reg = regression_df.loc[regression_df.index.intersection(val_idx)]
        test_reg = regression_df.loc[regression_df.index.intersection(test_idx)]
        print("Using precomputed splits from", splits_dir)
    else:
        train_cls, val_cls, test_cls = split_dataset(classification_df, "stratify_key", random_state=seed)
        train_reg, val_reg, test_reg = split_dataset(regression_df, "stratify_key", random_state=seed)

    print("Classification splits:", len(train_cls), len(val_cls), len(test_cls))
    print("Regression splits:", len(train_reg), len(val_reg), len(test_reg))

    train_cls_dataset = PeptidePredictorDataset(train_cls, task="classification")
    val_cls_dataset = PeptidePredictorDataset(val_cls, task="classification")
    test_cls_dataset = PeptidePredictorDataset(test_cls, task="classification")

    train_reg_dataset = PeptidePredictorDataset(train_reg, task="regression")
    val_reg_dataset = PeptidePredictorDataset(val_reg, task="regression")
    test_reg_dataset = PeptidePredictorDataset(test_reg, task="regression")

    collate = lambda batch: esm_collate_fn(batch, batch_converter, max_seq_length, alphabet.padding_idx)

    loader_gen = torch.Generator().manual_seed(seed)
    train_cls_loader = DataLoader(
        train_cls_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=loader_gen,
    )
    val_cls_loader = DataLoader(val_cls_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_cls_loader = DataLoader(test_cls_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    train_reg_loader = DataLoader(
        train_reg_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=loader_gen,
    )
    val_reg_loader = DataLoader(val_reg_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_reg_loader = DataLoader(test_reg_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = PeptidePredictor(esm_model_pretrained, alphabet, hidden_dim=hidden_dim, dropout=dropout)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    model.freeze_encoder()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.shared.parameters())
                + list(model.ap_head.parameters())
                + list(model.cls_head.parameters()),
                "lr": lr,
            }
        ]
    )
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.SmoothL1Loss(beta=0.5)

    reg_iterator = iter(train_reg_loader)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_cls_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for tokens_cls, cls_labels in pbar:
            tokens_cls = tokens_cls.to(device)
            cls_labels = cls_labels.to(device)
            ap_pred_cls, cls_pred = model(tokens_cls)
            loss_cls = criterion_cls(cls_pred.squeeze(), cls_labels.float())

            try:
                tokens_reg, reg_labels = next(reg_iterator)
            except StopIteration:
                reg_iterator = iter(train_reg_loader)
                tokens_reg, reg_labels = next(reg_iterator)

            tokens_reg = tokens_reg.to(device)
            reg_labels = reg_labels.to(device)
            ap_pred_reg, _ = model(tokens_reg)
            loss_reg = criterion_reg(ap_pred_reg.squeeze(), reg_labels.float())

            total_loss = loss_cls + loss_reg
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=total_loss.item())

        print(f"Epoch {epoch + 1}: mean loss {epoch_loss / max(1, len(train_cls_loader)):.4f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_out = save_path.parent / "config_used.yaml"
    config_payload = {
        "data_csv": str(data_csv),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "max_seq_length": max_seq_length,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "save_path": str(save_path),
        "seed": seed,
        "splits_dir": str(splits_dir),
    }
    config_out.write_text(yaml.safe_dump(config_payload, sort_keys=False))
    torch.save(model.state_dict(), save_path)
    print("Saved", save_path)

    _ = val_cls_loader, test_cls_loader, val_reg_loader, test_reg_loader
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
