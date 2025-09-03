import torch
import torch.nn as nn
import math
import esm

esm_model_pretrained, ALPHABET = esm.pretrained.esm2_t12_35M_UR50D()
PAD_TOKEN_ID = ALPHABET.padding_idx

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        # x: (B, T, H)  →  add positional (T,1,H) after swap
        return x + self.pe[: x.size(1)].transpose(0, 1)


class Decoder(nn.Module):
    def __init__(self, max_seq_length, hidden_dim, num_layers, nhead,
                 vocab_size, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN_ID)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_seq_length + 1)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            layer, num_layers, norm=nn.LayerNorm(hidden_dim)
        )

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_tokens, memory, memory_mask):
        B, T = tgt_tokens.size()
        H = self.embed.embedding_dim

        tgt = self.embed(tgt_tokens) * math.sqrt(H)  # (B, T, H)

        x = torch.cat([
            torch.zeros(B, 1, H, device=tgt.device), 
            tgt
        ], dim=1)  # (B, T+1, H)

        x = self.pos_enc(x)

        causal = nn.Transformer.generate_square_subsequent_mask(T + 1)\
                   .to(x.device)

        x = self.decoder(
            x, memory,
            tgt_mask=causal,
            memory_key_padding_mask=memory_mask
        )

        return self.output(x[:, 1:, :])  # (B, T, V)

class CVAESimpleEnc(nn.Module):
    def __init__(self, vocab_size,
                 latent_dim=64, cond_dim=6, cond_summary_dim=64,
                 max_seq_length=50, 
                 encoder_hidden_dim=256, num_encoder_layers=2,
                 decoder_hidden_dim=256, num_decoder_layers=2,
                 padding_idx=PAD_TOKEN_ID,
                 nhead=8, dropout=0.1):
        super().__init__()
        self.encoder_hidden = encoder_hidden_dim
        self.decoder_hidden = decoder_hidden_dim

        self.padding_idx = padding_idx

        self.enc_embed = nn.Embedding(vocab_size, encoder_hidden_dim, padding_idx=PAD_TOKEN_ID)
        self.enc_pos   = PositionalEncoding(encoder_hidden_dim, max_len=max_seq_length)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=encoder_hidden_dim, nhead=nhead, dim_feedforward=encoder_hidden_dim*nhead,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_summary_dim),
            nn.GELU(),
        )

        self.fc_mu     = nn.Linear(decoder_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(decoder_hidden_dim, latent_dim)
        
        self.prior_mu     = nn.Linear(cond_summary_dim, latent_dim)
        self.prior_logvar = nn.Linear(cond_summary_dim, latent_dim)

        self.cond_head_bin  = nn.Linear(cond_summary_dim, 2)
        self.cond_head_cont = nn.Linear(cond_summary_dim, 4)
        self.mask_head_logit = nn.Linear(cond_summary_dim, cond_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, decoder_hidden_dim)
        self.cond_to_hidden = nn.Linear(cond_summary_dim, decoder_hidden_dim)

        self.cond_reconstructor = nn.Sequential(
            nn.Linear(latent_dim, cond_dim),
        )

        self.decoder = Decoder(
            max_seq_length,
            hidden_dim=decoder_hidden_dim,
            num_layers=num_decoder_layers,
            nhead=nhead,
            vocab_size=vocab_size,
            #cond_summary_dim=cond_summary_dim,
            dropout=dropout
        )

    def compute_summary(self, cond, mask):
        cond_in = cond * mask
        x = torch.cat([cond_in, mask], dim=-1)      # (B, 2*cond_dim)
        summary = self.cond_encoder(x)           # (B, cond_summary_dim)

        summary = summary * math.sqrt(self.latent_to_hidden.out_features)
        return summary

    def compute_prior(self, summary):
        return self.prior_mu(summary), \
               self.prior_logvar(summary)

    def encode(self, tokens):
        x = self.enc_embed(tokens) * math.sqrt(self.encoder_hidden)
        x = self.enc_pos(x)

        key_pad = tokens.eq(self.padding_idx)

        x = self.encoder(x, src_key_padding_mask=key_pad)

        mask = ~tokens.eq(self.padding_idx)
        lengths = mask.sum(dim=1, keepdim=True)
        sum_emb = (x * mask.unsqueeze(-1)).sum(dim=1)
        seq_embed = sum_emb / lengths.clamp(min=1)

        return self.fc_mu(seq_embed), self.fc_logvar(seq_embed)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std

    def decode(self, z, tgt_tokens, summary):
        hid = self.latent_to_hidden(z)               # (B, H)
        latent_emb = hid.unsqueeze(1)                # (B, 1, H)

        cond_hid = self.cond_to_hidden(summary)      # (B, H)
        cond_emb = cond_hid.unsqueeze(1)         # (B, 1, H)
        memory = torch.cat([latent_emb, cond_emb], dim=1)
        mem_mask = torch.zeros(
            memory.size(0), memory.size(1),
            dtype=torch.bool, device=memory.device
        )
        return self.decoder(tgt_tokens, memory, mem_mask)

    def forward(self, tokens, cond, mask):
        mu_enc, logvar_enc = self.encode(tokens)
        z = self.reparameterize(mu_enc, logvar_enc)

        summary = self.compute_summary(cond, mask)
        p_mu, p_logvar = self.compute_prior(summary)

        bc_logit  = self.cond_head_bin(summary)
        cc_pred = self.cond_head_cont(summary)
        mask_logit  = self.mask_head_logit(summary)

        logits = self.decode(z, tokens, summary)
        return logits, mu_enc, logvar_enc, p_mu, p_logvar, bc_logit, cc_pred, mask_logit