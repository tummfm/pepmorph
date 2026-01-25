#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from common import DATA_PROCESSED, DEFAULT_CVAE_CHECKPOINT, load_split_indices
from cvae.models import CVAESimpleEnc
from cvae.datasets import CVAEAllDataset
from cvae.utils import (
    CONDITION_LENGTH,
    MAX_FASTA_LENGTH,
    MAX_SEQ_LENGTH,
    ALPHABET,
    PAD_TOKEN_ID,
    finetune_collate_fn,
    set_seed,
    vae_loss_fn_with_cond,
)

FEATURES = [
    "length",
    "is_assembled",
    "ap",
    "has_beta_sheet_content",
    "hydrophobic_moment",
    "net_charge",
]

IDX_BIN = [1, 3]
IDX_CONT = [0, 2, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CVAE reconstruction and auxiliary heads.")
    parser.add_argument(
        "--data-csv",
        type=str,
        default=str(DATA_PROCESSED / "merged_all.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CVAE_CHECKPOINT),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--mask-keep", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataloaders(data_csv: Path, batch_size: int):
    df = pd.read_csv(data_csv, keep_default_na=False, na_values=[""])
    split_idx = load_split_indices()
    if split_idx:
        train_df = df.iloc[split_idx["train"]].copy()
        val_df = df.iloc[split_idx["val"]].copy()
        test_df = df.iloc[split_idx["test"]].copy()
        print("Using precomputed splits from data/splits")
    else:
        train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df["length"], random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.1, stratify=train_val_df["length"], random_state=42)

    train_dataset = CVAEAllDataset(train_df, max_fasta_length=MAX_FASTA_LENGTH, random_mask=True)
    val_dataset = CVAEAllDataset(val_df, max_fasta_length=MAX_FASTA_LENGTH)
    test_dataset = CVAEAllDataset(test_df, max_fasta_length=MAX_FASTA_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=finetune_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=finetune_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=finetune_collate_fn)

    return train_loader, val_loader, test_loader, test_dataset


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = CVAESimpleEnc(
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
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)

    model.eval()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    print("CVAE initialized with finetuned architecture")
    return model


def evaluate(model, dataloader, pad_idx, device, kl_weight):
    model.eval()
    total = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "tokens": 0}

    with torch.no_grad():
        for tokens, tgt_tokens, conds, mask in dataloader:
            bsz = tokens.size(0)
            tokens = tokens.to(device)
            tgt_tokens = tgt_tokens.to(device)
            conds = conds.to(device)
            mask = mask.to(device)

            logits, mu, logvar, prior_mu, prior_logvar, bc_logit, cc_pred, mask_logit = model(tokens, conds, mask)

            loss, recon, kl, _ = vae_loss_fn_with_cond(
                logits=logits.view(-1, logits.size(-1)),
                tgt=tgt_tokens.view(-1),
                mu=mu,
                logvar=logvar,
                prior_mu=prior_mu,
                prior_logvar=prior_logvar,
                bc_logit=bc_logit,
                cc_pred=cc_pred,
                mask_logit=mask_logit,
                cond=conds,
                mask=mask,
                pad_idx=pad_idx,
                kl_weight=kl_weight,
                lambda_bin=1.0,
                lambda_cont=1.0,
            )

            total["loss"] += loss.item() * bsz
            total["recon"] += recon.item() * bsz
            total["kl"] += kl.item() * bsz
            total["tokens"] += (tgt_tokens != pad_idx).sum().item()

    n = len(dataloader.dataset)
    total["loss"] /= n
    total["recon"] /= n
    total["kl"] /= n
    total["ppl"] = float(torch.exp(torch.tensor(total["recon"] * total["tokens"] / total["tokens"])))

    return total


def make_complete_case_subset(dataset, cond_dim=CONDITION_LENGTH) -> Subset:
    idxs = []
    for i in range(len(dataset)):
        item = dataset[i]
        mask = item[2]
        if mask.sum().item() == cond_dim:
            idxs.append(i)
    return Subset(dataset, idxs)


def apply_random_mask_on_top(mask, p_keep=0.5):
    keep = (torch.rand_like(mask) < p_keep).float()
    return mask * keep


def evaluate_ppl(model, dataloader, pad_idx, device, kl_weight, mask_mode="observed", p_keep=0.5):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0

    for tokens, tgt_tokens, conds, mask in dataloader:
        bsz = tokens.size(0)
        tokens = tokens.to(device)
        tgt_tokens = tgt_tokens.to(device)
        conds = conds.to(device)
        mask = mask.to(device)

        if mask_mode == "observed":
            mask_eff = mask
        elif mask_mode == "full":
            mask_eff = torch.ones_like(mask)
        elif mask_mode == "random_on_top":
            mask_eff = apply_random_mask_on_top(mask, p_keep=p_keep)
        else:
            raise ValueError(f"Unknown mask_mode={mask_mode}")

        logits, mu, logvar, prior_mu, prior_logvar, bc_logit, cc_pred, mask_logit = model(tokens, conds, mask_eff)

        loss, recon, kl, _ = vae_loss_fn_with_cond(
            logits=logits.view(-1, logits.size(-1)),
            tgt=tgt_tokens.view(-1),
            mu=mu,
            logvar=logvar,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            bc_logit=bc_logit,
            cc_pred=cc_pred,
            mask_logit=mask_logit,
            cond=conds,
            mask=mask_eff,
            pad_idx=pad_idx,
            kl_weight=kl_weight,
            lambda_bin=1.0,
            lambda_cont=1.0,
            label_smoothing=0.0,
        )

        total_loss += loss.item() * bsz
        total_recon += recon.item() * bsz
        total_kl += kl.item() * bsz
        n_samples += bsz

    mean_loss = total_loss / n_samples
    mean_recon = total_recon / n_samples
    mean_kl = total_kl / n_samples
    ppl = float(torch.exp(torch.tensor(mean_recon)))

    return {"loss": mean_loss, "recon": mean_recon, "kl": mean_kl, "ppl": ppl}


def evaluate_aux_heads_per_dim(model, dataloader, device, max_pos_w=50.0, thresh=0.5):
    model.eval()

    mask_stats = {f: {"bce_sum": 0.0, "acc_sum": 0.0, "pos_sum": 0.0, "n_sum": 0.0} for f in FEATURES}
    bin_stats = {
        FEATURES[i]: {
            "bce_sum": 0.0,
            "acc_sum": 0.0,
            "bal_acc_sum": 0.0,
            "pos_sum": 0.0,
            "n_obs": 0.0,
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }
        for i in IDX_BIN
    }
    cont_stats = {
        FEATURES[i]: {"mae_sum": 0.0, "mse_sum": 0.0, "n_obs": 0.0, "obs_rate_sum": 0.0}
        for i in IDX_CONT
    }

    total_n = 0

    for tokens, _, conds, mask in dataloader:
        bsz = tokens.size(0)
        total_n += bsz

        tokens = tokens.to(device)
        conds = conds.to(device)
        mask = mask.to(device)

        logits, mu, logvar, prior_mu, prior_logvar, bc_logit, cc_pred, mask_logit = model(tokens, conds, mask)

        pos_frac = mask.mean(dim=0).clamp(1e-4, 1 - 1e-4)
        pos_w = ((1 - pos_frac) / pos_frac).clamp(1.0, max_pos_w)

        bce_per = F.binary_cross_entropy_with_logits(mask_logit, mask, pos_weight=pos_w, reduction="none")
        bce_dim = bce_per.mean(dim=0)

        mask_pred = (torch.sigmoid(mask_logit) >= thresh).float()
        acc_dim = (mask_pred == mask).float().mean(dim=0)
        pos_dim = mask.float().mean(dim=0)

        for j, feat in enumerate(FEATURES):
            mask_stats[feat]["bce_sum"] += float(bce_dim[j]) * bsz
            mask_stats[feat]["acc_sum"] += float(acc_dim[j]) * bsz
            mask_stats[feat]["pos_sum"] += float(pos_dim[j]) * bsz
            mask_stats[feat]["n_sum"] += bsz

        for j, idx in enumerate(IDX_BIN):
            feat = FEATURES[idx]
            logits_bin = bc_logit[:, j]
            mask_bin = mask[:, idx]
            target = conds[:, idx]

            obs = mask_bin.bool()
            if obs.sum().item() == 0:
                continue

            y_true = target[obs]
            y_logit = logits_bin[obs]

            bce = F.binary_cross_entropy_with_logits(y_logit, y_true, reduction="mean")
            pred = (torch.sigmoid(y_logit) >= thresh).float()

            tp = float(((pred == 1) & (y_true == 1)).sum().item())
            tn = float(((pred == 0) & (y_true == 0)).sum().item())
            fp = float(((pred == 1) & (y_true == 0)).sum().item())
            fn = float(((pred == 0) & (y_true == 1)).sum().item())

            n_obs = float(obs.sum().item())
            acc = float((pred == y_true).float().mean().item())
            pos_rate = float(y_true.float().mean().item())

            bin_stats[feat]["bce_sum"] += float(bce.item()) * n_obs
            bin_stats[feat]["acc_sum"] += acc * n_obs
            bin_stats[feat]["pos_sum"] += pos_rate * n_obs
            bin_stats[feat]["n_obs"] += n_obs
            bin_stats[feat]["tp"] += tp
            bin_stats[feat]["tn"] += tn
            bin_stats[feat]["fp"] += fp
            bin_stats[feat]["fn"] += fn

        for j, idx in enumerate(IDX_CONT):
            feat = FEATURES[idx]
            pred_vals = cc_pred[:, j]
            mask_cont = mask[:, idx]
            target_vals = conds[:, idx]

            obs = mask_cont.bool()
            if obs.sum().item() == 0:
                continue

            y_true = target_vals[obs]
            y_pred = pred_vals[obs]

            mae = torch.abs(y_pred - y_true).mean()
            mse = ((y_pred - y_true) ** 2).mean()

            cont_stats[feat]["mae_sum"] += float(mae.item()) * obs.sum().item()
            cont_stats[feat]["mse_sum"] += float(mse.item()) * obs.sum().item()
            cont_stats[feat]["n_obs"] += float(obs.sum().item())
            cont_stats[feat]["obs_rate_sum"] += float(obs.float().mean().item()) * bsz

    mask_out = {}
    for feat, st in mask_stats.items():
        n = st["n_sum"]
        if n == 0:
            continue
        mask_out[feat] = {
            "bce": st["bce_sum"] / n,
            "acc": st["acc_sum"] / n,
            "pos_rate": st["pos_sum"] / n,
        }

    bin_out = {}
    for feat, st in bin_stats.items():
        n = st["n_obs"]
        if n == 0:
            continue
        tp, tn, fp, fn = st["tp"], st["tn"], st["fp"], st["fn"]
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        bal_acc_total = np.nanmean([tpr, tnr])

        bin_out[feat] = {
            "obs_count": int(n),
            "pos_rate": st["pos_sum"] / n,
            "bce": st["bce_sum"] / n,
            "acc": st["acc_sum"] / n,
            "bal_acc": float(bal_acc_total),
            "tpr": float(tpr),
            "tnr": float(tnr),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    cont_out = {}
    for feat, st in cont_stats.items():
        n = st["n_obs"]
        if n == 0:
            continue
        cont_out[feat] = {
            "obs_count": int(n),
            "obs_rate": st["obs_rate_sum"] / total_n,
            "mae": st["mae_sum"] / n,
            "rmse": float(np.sqrt(st["mse_sum"] / n)),
        }

    return {"mask": mask_out, "binary": bin_out, "continuous": cont_out}


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.device_count())

    _, val_loader, test_loader, test_dataset = build_dataloaders(Path(args.data_csv), args.batch_size)
    model = load_model(Path(args.checkpoint), device)

    metrics_val = evaluate(model, val_loader, PAD_TOKEN_ID, device, kl_weight=args.kl_weight)
    print("==== Validation metrics ====")
    print(f"Loss: {metrics_val['loss']:.4f} | Recon: {metrics_val['recon']:.4f} | KL: {metrics_val['kl']:.4f}")
    print(f"Perplexity: {metrics_val['ppl']:.2f}")

    metrics_test = evaluate(model, test_loader, PAD_TOKEN_ID, device, kl_weight=args.kl_weight)
    print("==== Test metrics ====")
    print(f"Loss: {metrics_test['loss']:.4f} | Recon: {metrics_test['recon']:.4f} | KL: {metrics_test['kl']:.4f}")
    print(f"Perplexity: {metrics_test['ppl']:.2f}")

    test_complete = make_complete_case_subset(test_dataset, cond_dim=CONDITION_LENGTH)
    test_loader_complete = DataLoader(test_complete, batch_size=args.batch_size, shuffle=False, collate_fn=finetune_collate_fn)

    m_obs = evaluate_ppl(model, test_loader, PAD_TOKEN_ID, device, kl_weight=args.kl_weight, mask_mode="observed")
    m_full = evaluate_ppl(model, test_loader_complete, PAD_TOKEN_ID, device, kl_weight=args.kl_weight, mask_mode="full")
    m_rand = evaluate_ppl(
        model,
        test_loader_complete,
        PAD_TOKEN_ID,
        device,
        kl_weight=args.kl_weight,
        mask_mode="random_on_top",
        p_keep=args.mask_keep,
    )

    print("TEST (all) observed-mask PPL:", m_obs["ppl"])
    print("TEST (complete-case) full-mask PPL:", m_full["ppl"])
    print("TEST (complete-case) random-on-top PPL:", m_rand["ppl"])

    aux_detail = evaluate_aux_heads_per_dim(model, test_loader, device)
    print("MASK HEAD PER DIM:", aux_detail["mask"])
    print("BINARY HEADS:", aux_detail["binary"])
    print("CONT HEADS:", aux_detail["continuous"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
