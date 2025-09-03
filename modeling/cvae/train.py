import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models import  CVAESimpleEnc
from datasets import CVAEAllDataset
from utils import (
    CONDITION_LENGTH, MAX_FASTA_LENGTH, MAX_SEQ_LENGTH, PAD_TOKEN_ID, ALPHABET,
    finetune_collate_fn, vae_loss_fn_with_cond, kl_schedule
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.device_count())

batch_size = 2048
df = pd.read_csv("../../clean_data/merged_all.csv", keep_default_na=False, na_values=[''])
train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df['length'], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, stratify=train_val_df['length'], random_state=42)

w = np.ones(len(train_df), dtype=np.float32)
w *= np.where(train_df['has_beta_sheet_content'] == 1, 10.0, 1.0)
w *= np.where(train_df['is_assembled'] == 0, 2.0, 1.0)

sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)

train_dataset = CVAEAllDataset(train_df, max_fasta_length=MAX_FASTA_LENGTH, random_mask=True, random_size=5000)
val_dataset   = CVAEAllDataset(val_df, max_fasta_length=MAX_FASTA_LENGTH)
test_dataset  = CVAEAllDataset(test_df, max_fasta_length=MAX_FASTA_LENGTH)

train_loader_ft = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=finetune_collate_fn)
val_loader_ft   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=finetune_collate_fn)
test_loader_ft  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=finetune_collate_fn)

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
    dropout=0.1)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    cvae_model = nn.DataParallel(cvae_model)

cvae_model.to(device)
print("CVAE initialized")

num_epochs = 250
kl_weight = 0.05
lambda_bin, lambda_cont = 2.0, 0.5
warmup_epochs = 100

learning_rate = 1e-3
optimizer = torch.optim.AdamW(cvae_model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

print("Starting fine-tuning on conditioned dataset (full condition)...")
for epoch in range(1, num_epochs + 1):
    cvae_model.train()
    running_loss = running_recon = running_kl = running_cond = 0.0
    num_batches = 0

    current_kl_weight = kl_weight * kl_schedule(epoch - 1, num_epochs)
    pbar = tqdm(train_loader_ft, desc=f"FT Epoch {epoch}/{num_epochs}")
    for tokens, tgt_tokens, conds, mask in pbar:
        tokens     = tokens.to(device)
        tgt_tokens = tgt_tokens.to(device)
        conds      = conds.to(device)
        mask       = mask.to(device)

        optimizer.zero_grad()
        logits, mu, logvar, p_mu, p_logvar, bc_logit, cc_pred, mask_logit = cvae_model(tokens, conds, mask)
        loss, recon, kl, cond_loss = vae_loss_fn_with_cond(
            logits=logits.view(-1, logits.size(-1)),
            tgt=tgt_tokens.view(-1),
            mu=mu, logvar=logvar, prior_mu=p_mu, prior_logvar=p_logvar,
            bc_logit=bc_logit, cc_pred=cc_pred, mask_logit=mask_logit,
            cond=conds, mask=mask,
            pad_idx=PAD_TOKEN_ID, kl_weight=current_kl_weight, lambda_bin=lambda_bin, lambda_cont=lambda_cont)
        loss.backward()
        optimizer.step()

        running_loss  += loss.item()
        running_recon += recon.item()
        running_kl    += kl.item()
        running_cond  += cond_loss.item()
        num_batches   += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon.item():.4f}", kl=f"{kl.item():.4f}", cond=f"{cond_loss.item():.4f}")

    avg_train_loss  = running_loss / num_batches
    avg_train_recon = running_recon / num_batches
    avg_train_kl    = running_kl / num_batches
    avg_train_cond  = running_cond / num_batches

    cvae_model.eval()
    total_loss = total_recon = total_kl = total_cond = 0.0
    val_batches = 0
    with torch.no_grad():
        for tokens, tgt_tokens, conds, mask in val_loader_ft:
            tokens     = tokens.to(device)
            tgt_tokens = tgt_tokens.to(device)
            conds      = conds.to(device)
            mask       = mask.to(device)

            logits, mu, logvar, p_mu, p_logvar, bc_logit, cc_pred, mask_logit = cvae_model(tokens, conds, mask)
            loss, recon, kl, cond_loss = vae_loss_fn_with_cond(
                logits=logits.view(-1, logits.size(-1)),
                tgt=tgt_tokens.view(-1),
                mu=mu, logvar=logvar, prior_mu=p_mu, prior_logvar=p_logvar,
                bc_logit=bc_logit, cc_pred=cc_pred, mask_logit=mask_logit,
                cond=conds, mask=mask,
                pad_idx=PAD_TOKEN_ID, kl_weight=current_kl_weight, lambda_bin=lambda_bin, lambda_cont=lambda_cont)
            total_loss  += loss.item()
            total_recon += recon.item()
            total_kl    += kl.item()
            total_cond  += cond_loss.item()
            val_batches += 1

    avg_val_loss  = total_loss / val_batches
    avg_val_recon = total_recon / val_batches
    avg_val_kl    = total_kl / val_batches
    avg_val_cond  = total_cond / val_batches
    print(f"FT Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}, Cond: {avg_train_cond:.4f}) | \
          Val Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}, Cond: {avg_val_cond:.4f})")

    if epoch > warmup_epochs:
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"🔽 LR reduced from {old_lr:.2e} → {new_lr:.2e} at epoch {epoch}")

model_to_save = cvae_model.module if hasattr(cvae_model, "module") else cvae_model
torch.save(model_to_save.state_dict(), "finetuned_cvae.pt")
print("Model saved to finetuned_cvae.pt")

