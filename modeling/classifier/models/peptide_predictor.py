import torch
import torch.nn as nn
import torch.nn.functional as F

class PeptidePredictor(nn.Module):
    """
    ESM last-layer -> masked mean over residues -> tiny shared MLP -> 2 heads.
    Returns: ap_pred (B,), cls_logit (B,)  [logits, not probs]
    """
    def __init__(self, esm_model, alphabet, hidden_dim=128, dropout=0.10):
        super().__init__()
        self.esm = esm_model
        self.alphabet = alphabet
        self.pad_idx = alphabet.padding_idx
        self.bos_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.embed_dim = self.esm.embed_dim
        self.num_layers = self.esm.num_layers

        self.shared = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ap_head  = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, 1)

    @torch.no_grad()
    def freeze_encoder(self):
        for p in self.esm.parameters():
            p.requires_grad = False
        self.esm.eval()

    def unfreeze_encoder(self):
        for p in self.esm.parameters():
            p.requires_grad = True
        self.esm.train()

    def _residue_mask(self, tokens):
        mask = (tokens != self.pad_idx) & (tokens != self.bos_idx) & (tokens != self.eos_idx)
        fallback = mask.any(dim=1, keepdim=True)
        return torch.where(fallback, mask, tokens != self.pad_idx)

    def _masked_mean(self, x, mask):
        mask_f = mask.float().unsqueeze(-1)
        summed  = (x * mask_f).sum(dim=1)
        denom   = mask_f.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def forward(self, tokens):
        out = self.esm(tokens, repr_layers=[self.num_layers], return_contacts=False)
        reps = out["representations"][self.num_layers]

        mask = self._residue_mask(tokens)
        seq_rep = self._masked_mean(reps, mask)

        h = self.shared(seq_rep)
        ap      = self.ap_head(h).squeeze(-1)
        cls_logit = self.cls_head(h).squeeze(-1)
        return ap, cls_logit
