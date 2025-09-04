import esm
import torch
import torch.nn as nn
import torch.nn.functional as F

esm_model_pretrained, ALPHABET = esm.pretrained.esm2_t12_35M_UR50D()

BOS_ID = ALPHABET.cls_idx
EOS_ID = ALPHABET.eos_idx
PAD_TOKEN_ID = ALPHABET.padding_idx
MAX_FASTA_LENGTH = 10
MAX_SEQ_LENGTH = MAX_FASTA_LENGTH + 2
CONDITION_LENGTH = 6


IDX_BIN  = [1, 3]
IDX_CONT = [0, 2, 4, 5]


batch_converter = ALPHABET.get_batch_converter(truncation_seq_length=MAX_FASTA_LENGTH)
idx_to_fasta = {v:k for k,v in ALPHABET.to_dict().items()}

def convert_and_pad(data, seq_length):
    """
    Uses the ESM batch converter to tokenize sequences and pads them to seq_length.
    Data is a list of (name, sequence) tuples.
    """
    _, _, tokens = batch_converter(data)
    current_len = tokens.size(1)
    if current_len < seq_length:
        pad_length = seq_length - current_len
        padding = torch.full((tokens.size(0), pad_length), ALPHABET.padding_idx, dtype=tokens.dtype)
        tokens = torch.cat([tokens, padding], dim=1)
    return tokens

def esm_collate_fn(batch):
    """
    batch: list of tuples (sequence, target)
    Returns:
      - tokens: a tensor of shape (batch_size, seq_len) ready for the ESM model
      - targets: a tensor of the corresponding targets
    """
    data = [(f"peptide_{i}", seq) for i, (seq, _) in enumerate(batch)]

    tokens = convert_and_pad(data, seq_length=MAX_SEQ_LENGTH)

    targets = torch.tensor([target for _, target in batch])
    return tokens, targets

def pretrain_collate_fn(batch):
    """
    Constructs a batch from the online pretraining dataset.
    Returns:
        - tokens: Tensor (batch, max_seq_length)
        - tgt_tokens: copy of tokens for reconstruction loss
        - conds: Tensor (batch, cond_dim); here cond_dim=2.
        - mask: Tensor (batch, cond_dim), where [1, 0] indicates that only the first (length) field is present.
    """
    data = [(f"peptide_{i}", seq) for i, (seq, _) in enumerate(batch)]
    tokens = convert_and_pad(data, seq_length=MAX_SEQ_LENGTH)

    tgt_tokens = tokens.clone()
    pad_col = torch.full((tgt_tokens.size(0), 1),
                         ALPHABET.padding_idx,
                         dtype=torch.long)
    tgt_tokens    = torch.cat([tgt_tokens[:, 1:], pad_col], dim=1)

    conds = torch.stack([cond for _, cond in batch], dim=0)
    mask = torch.ones_like(conds)
    mask[:, 1] = 0.0
    return tokens, tgt_tokens, conds, mask

def finetune_collate_fn(batch):
    """
    For fine-tuning, the condition vector is fully specified.
    The mask is all ones since every condition is available.
    """
    data = [(f"peptide_{i}", seq) for i, (seq, _, _) in enumerate(batch)]
    tokens = convert_and_pad(data, seq_length=MAX_SEQ_LENGTH)

    tgt_tokens = tokens.clone()
    pad_col = torch.full((tgt_tokens.size(0), 1),
                         ALPHABET.padding_idx,
                         dtype=torch.long)
    tgt_tokens    = torch.cat([tgt_tokens[:, 1:], pad_col], dim=1)

    conds = torch.stack([cond for _, cond, _ in batch], dim=0)
    mask = torch.stack([mask for _, _, mask in batch], dim=0)
    return tokens, tgt_tokens, conds, mask


def masked_regression_loss(pred, target, mask, reduction="mean", huber_delta=None):
    """
    pred, target, mask: (B, D)
    mask in {0,1}, computes loss only where mask==1
    """
    if huber_delta is None:
        per_elem = (pred - target) ** 2
    else:
        per_elem = F.smooth_l1_loss(pred, target, beta=huber_delta, reduction="none")
    per_elem = per_elem * mask
    per_sample = per_elem.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return per_sample.mean() if reduction == "mean" else per_sample

def cond_loss(bc_logit, cc_pred, mask_logit, cond, mask,
              lambda_mask=1.0, lambda_bin=2.0, lambda_cont=1.0,
              max_pos_w=50.0, huber_delta=None):
    device = cond.device

    # Mask reconstruction
    pos_frac = mask.mean(dim=0).clamp(1e-4, 1-1e-4)
    pos_w    = ((1 - pos_frac) / pos_frac).clamp(1.0, max_pos_w)
    loss_mask = F.binary_cross_entropy_with_logits(
        mask_logit, mask, pos_weight=pos_w, reduction="mean"
    )

    # Binary value masked supervision
    loss_bin_terms = []
    for j, col in enumerate(IDX_BIN):
        m = mask[:, col].bool()
        if m.any():
            tgt_j   = cond[:, col][m].float()
            logit_j = bc_logit[:, j][m]
            pos_frac_val = tgt_j.mean().clamp(1e-4, 1-1e-4)
            pos_w_val    = ((1 - pos_frac_val) / pos_frac_val).clamp(1.0, max_pos_w)
            w = torch.ones_like(tgt_j)
            w[tgt_j == 1] = pos_w_val
            loss_bin_terms.append(F.binary_cross_entropy_with_logits(logit_j, tgt_j, weight=w))
    loss_bin = torch.stack(loss_bin_terms).mean() if loss_bin_terms else torch.tensor(0., device=device)

    # Continuous dims masked supervision
    cc_tgt = cond[:, IDX_CONT]
    cc_m   = mask[:, IDX_CONT]
    loss_cont = masked_regression_loss(cc_pred, cc_tgt, cc_m, huber_delta=huber_delta)

    total = lambda_mask*loss_mask + lambda_bin*loss_bin + lambda_cont*loss_cont
    return total, loss_mask, loss_bin, loss_cont


def vae_loss_fn_with_cond(
    logits, tgt, mu, logvar, prior_mu, prior_logvar,
    bc_logit, cc_pred, mask_logit,
    cond, mask, pad_idx, kl_weight,
    lambda_mask=1.0, lambda_bin=2.0, lambda_cont=1.0,
):
    recon = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)(
        logits.view(-1, logits.size(-1)), tgt.view(-1)
    )

    q2, p2 = logvar.exp(), prior_logvar.exp()
    kl = 0.5 * torch.sum(
        prior_logvar - logvar + (q2 + (mu - prior_mu).pow(2)) / p2 - 1, dim=1
    ).mean()

    cond_term, _, _, _ = cond_loss(
        bc_logit, cc_pred, mask_logit, cond, mask,
        lambda_mask=lambda_mask, lambda_bin=lambda_bin, lambda_cont=lambda_cont
    )

    loss = recon + kl_weight*kl + cond_term
    return loss, recon, kl, cond_term


def kl_schedule(epoch, num_epochs, n_cycles=10, ratio=0.5):
    cycle_len = num_epochs / n_cycles
    t = (epoch % cycle_len) / cycle_len
    return float(min(1.0, t/ratio))