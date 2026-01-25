#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models import CVAESimpleEnc
from datasets import CVAEAllDataset
from utils import (
    CONDITION_LENGTH,
    MAX_FASTA_LENGTH,
    MAX_SEQ_LENGTH,
    PAD_TOKEN_ID,
    ALPHABET,
    finetune_collate_fn,
    kl_schedule,
    set_seed,
    vae_loss_fn_with_cond,
)

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_PROCESSED = ROOT / "data" / "processed"
DEFAULT_SPLITS = ROOT / "data" / "splits"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "models" / "masked_cvae"


def resolve_path(path: Path | str, base: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train masked conditional CVAE.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-csv", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--n-cycles", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--mask-prob", type=float, default=None)
    parser.add_argument("--random-augmentation", type=int, default=None)

    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--encoder-hidden-dim", type=int, default=None)
    parser.add_argument("--decoder-hidden-dim", type=int, default=None)
    parser.add_argument("--num-encoder-layers", type=int, default=None)
    parser.add_argument("--num-decoder-layers", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lambda-bin", type=float, default=None)
    parser.add_argument("--lambda-cont", type=float, default=None)
    parser.add_argument("--sampler-beta-weight", type=float, default=None)
    parser.add_argument("--sampler-sa-weight", type=float, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    payload = yaml.safe_load(path.read_text())
    return payload or {}


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    overrides = {
        "seed": args.seed,
        "split_seed": args.split_seed,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "kl_weight": args.kl_weight,
        "n_cycles": args.n_cycles,
        "warmup_epochs": args.warmup_epochs,
        "mask_prob": args.mask_prob,
        "random_augmentation": args.random_augmentation,
        "latent_dim": args.latent_dim,
        "encoder_hidden_dim": args.encoder_hidden_dim,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "nhead": args.nhead,
        "dropout": args.dropout,
        "lambda_bin": args.lambda_bin,
        "lambda_cont": args.lambda_cont,
        "sampler_beta_weight": args.sampler_beta_weight,
        "sampler_sa_weight": args.sampler_sa_weight,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    if args.data_csv is not None:
        cfg["data_csv"] = args.data_csv
    if args.splits_dir is not None:
        cfg["splits_dir"] = args.splits_dir
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.checkpoint_name is not None:
        cfg["checkpoint_name"] = args.checkpoint_name
    return cfg


def load_split_indices(splits_dir: Path) -> dict[str, list[int]] | None:
    train_path = splits_dir / "train_idx.txt"
    val_path = splits_dir / "val_idx.txt"
    test_path = splits_dir / "test_idx.txt"
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        return None

    def _read(path: Path) -> list[int]:
        return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]

    return {"train": _read(train_path), "val": _read(val_path), "test": _read(test_path)}


def main() -> int:
    args = parse_args()

    cfg = load_config(Path(args.config).expanduser().resolve())
    cfg = apply_overrides(cfg, args)

    data_csv = resolve_path(cfg.get("data_csv", DATA_PROCESSED / "merged_all.csv"), ROOT)
    splits_dir = resolve_path(cfg.get("splits_dir", DEFAULT_SPLITS), ROOT)
    output_dir = resolve_path(cfg.get("output_dir", DEFAULT_OUTPUT_DIR), ROOT)
    checkpoint_name = cfg.get("checkpoint_name", "finetuned_cvae.pt")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    split_seed = int(cfg.get("split_seed", seed))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.device_count())

    df = pd.read_csv(data_csv, keep_default_na=False, na_values=[""])

    split_idx = load_split_indices(splits_dir)
    if split_idx:
        train_df = df.iloc[split_idx["train"]].copy()
        val_df = df.iloc[split_idx["val"]].copy()
        test_df = df.iloc[split_idx["test"]].copy()
        print("Using precomputed splits from", splits_dir)
    else:
        train_val_df, test_df = train_test_split(
            df, test_size=0.1, stratify=df["length"], random_state=split_seed
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.1, stratify=train_val_df["length"], random_state=split_seed
        )
        print("Using stratified splits with seed", split_seed)

    sampler_beta_weight = float(cfg.get("sampler_beta_weight", 10.0))
    sampler_sa_weight = float(cfg.get("sampler_sa_weight", 2.0))

    w = np.ones(len(train_df), dtype=np.float32)
    w *= np.where(train_df["has_beta_sheet_content"] == 1, sampler_beta_weight, 1.0)
    w *= np.where(train_df["is_assembled"] == 0, sampler_sa_weight, 1.0)

    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(seed)
    sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True, generator=sampler_gen)

    train_dataset = CVAEAllDataset(
        train_df,
        max_fasta_length=MAX_FASTA_LENGTH,
        random_mask=True,
        random_size=int(cfg.get("random_augmentation", 15000)),
        mask_prob=float(cfg.get("mask_prob", 0.5)),
        seed=seed,
    )
    val_dataset = CVAEAllDataset(val_df, max_fasta_length=MAX_FASTA_LENGTH)
    test_dataset = CVAEAllDataset(test_df, max_fasta_length=MAX_FASTA_LENGTH)

    batch_size = int(cfg.get("batch_size", 2048))
    train_loader_ft = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=finetune_collate_fn)
    val_loader_ft = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=finetune_collate_fn)
    test_loader_ft = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=finetune_collate_fn)

    cvae_model = CVAESimpleEnc(
        encoder_hidden_dim=int(cfg.get("encoder_hidden_dim", 256)),
        num_encoder_layers=int(cfg.get("num_encoder_layers", 2)),
        vocab_size=len(ALPHABET),
        latent_dim=int(cfg.get("latent_dim", 24)),
        cond_dim=CONDITION_LENGTH,
        max_seq_length=MAX_SEQ_LENGTH,
        decoder_hidden_dim=int(cfg.get("decoder_hidden_dim", 256)),
        num_decoder_layers=int(cfg.get("num_decoder_layers", 2)),
        nhead=int(cfg.get("nhead", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        cvae_model = nn.DataParallel(cvae_model)

    cvae_model.to(device)
    print("CVAE initialized")

    num_epochs = int(cfg.get("num_epochs", 250))
    kl_weight = float(cfg.get("kl_weight", 0.05))
    lambda_bin = float(cfg.get("lambda_bin", 2.0))
    lambda_cont = float(cfg.get("lambda_cont", 0.5))
    warmup_epochs = int(cfg.get("warmup_epochs", 100))
    n_cycles = int(cfg.get("n_cycles", 10))

    learning_rate = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(cvae_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    config_out = output_dir / "config_used.yaml"
    config_out.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("Starting fine-tuning on conditioned dataset (full condition)...")
    for epoch in range(1, num_epochs + 1):
        cvae_model.train()
        running_loss = running_recon = running_kl = running_cond = 0.0
        num_batches = 0

        current_kl_weight = kl_weight * kl_schedule(epoch - 1, num_epochs, n_cycles=n_cycles)
        pbar = tqdm(train_loader_ft, desc=f"FT Epoch {epoch}/{num_epochs}")
        for tokens, tgt_tokens, conds, mask in pbar:
            tokens = tokens.to(device)
            tgt_tokens = tgt_tokens.to(device)
            conds = conds.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits, mu, logvar, p_mu, p_logvar, bc_logit, cc_pred, mask_logit = cvae_model(tokens, conds, mask)
            loss, recon, kl, cond_loss = vae_loss_fn_with_cond(
                logits=logits.view(-1, logits.size(-1)),
                tgt=tgt_tokens.view(-1),
                mu=mu,
                logvar=logvar,
                prior_mu=p_mu,
                prior_logvar=p_logvar,
                bc_logit=bc_logit,
                cc_pred=cc_pred,
                mask_logit=mask_logit,
                cond=conds,
                mask=mask,
                pad_idx=PAD_TOKEN_ID,
                kl_weight=current_kl_weight,
                lambda_bin=lambda_bin,
                lambda_cont=lambda_cont,
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon.item()
            running_kl += kl.item()
            running_cond += cond_loss.item()
            num_batches += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon.item():.4f}",
                kl=f"{kl.item():.4f}",
                cond=f"{cond_loss.item():.4f}",
            )

        avg_train_loss = running_loss / num_batches
        avg_train_recon = running_recon / num_batches
        avg_train_kl = running_kl / num_batches
        avg_train_cond = running_cond / num_batches

        cvae_model.eval()
        total_loss = total_recon = total_kl = total_cond = 0.0
        val_batches = 0
        with torch.no_grad():
            for tokens, tgt_tokens, conds, mask in val_loader_ft:
                tokens = tokens.to(device)
                tgt_tokens = tgt_tokens.to(device)
                conds = conds.to(device)
                mask = mask.to(device)

                logits, mu, logvar, p_mu, p_logvar, bc_logit, cc_pred, mask_logit = cvae_model(tokens, conds, mask)
                loss, recon, kl, cond_loss = vae_loss_fn_with_cond(
                    logits=logits.view(-1, logits.size(-1)),
                    tgt=tgt_tokens.view(-1),
                    mu=mu,
                    logvar=logvar,
                    prior_mu=p_mu,
                    prior_logvar=p_logvar,
                    bc_logit=bc_logit,
                    cc_pred=cc_pred,
                    mask_logit=mask_logit,
                    cond=conds,
                    mask=mask,
                    pad_idx=PAD_TOKEN_ID,
                    kl_weight=current_kl_weight,
                    lambda_bin=lambda_bin,
                    lambda_cont=lambda_cont,
                )
                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()
                total_cond += cond_loss.item()
                val_batches += 1

        avg_val_loss = total_loss / val_batches
        avg_val_recon = total_recon / val_batches
        avg_val_kl = total_kl / val_batches
        avg_val_cond = total_cond / val_batches
        print(
            f"FT Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f} "
            f"(Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}, Cond: {avg_train_cond:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} "
            f"(Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}, Cond: {avg_val_cond:.4f})"
        )

        if epoch > warmup_epochs:
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(avg_val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                print(f"LR reduced from {old_lr:.2e} to {new_lr:.2e} at epoch {epoch}")

    model_to_save = cvae_model.module if hasattr(cvae_model, "module") else cvae_model
    checkpoint_path = output_dir / checkpoint_name
    torch.save(model_to_save.state_dict(), checkpoint_path)
    print("Model saved to", checkpoint_path)
    _ = test_loader_ft
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
