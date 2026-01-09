# models/OracleAD.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class OracleAD(nn.Module):
    """
    OracleAD baseline: Per-variable temporal encoder + QKV spatial attention + per-variable decoder
    """

    def __init__(self, num_vars, window_size, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # 1) per-variable temporal encoder
        self.encoder_lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.attn_pool_w = nn.Linear(hidden_dim, 1)

        # 2) spatial interaction (QKV attention)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 3) per-variable decoder
        self.decoder_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.readout_recon = nn.Linear(hidden_dim, 1)
        self.readout_pred = nn.Linear(hidden_dim, 1)

    def get_causal_embedding(self, x):
        """
        x: [B, N, L]
        return c: [B, N, d]
        """
        B, N, L = x.shape
        x_flat = x.view(B * N, L, 1)

        enc_out, _ = self.encoder_lstm(x_flat)  # [B*N, L, d]
        attn_scores = F.softmax(self.attn_pool_w(enc_out), dim=1)  # [B*N, L, 1]
        c_flat = torch.sum(enc_out * attn_scores, dim=1)  # [B*N, d]

        c = c_flat.view(B, N, self.hidden_dim)
        return c

    def spatial_interaction(self, c):
        """
        c: [B, N, d]
        return c_star: [B, N, d], A: [B, N, N]
        """
        Q = self.W_Q(c)
        K = self.W_K(c)
        V = self.W_V(c)

        d_k = c.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [B,N,N]
        A = F.softmax(scores, dim=-1)  # [B,N,N]

        c_star = torch.matmul(self.dropout(A), V)  # [B,N,d]
        return c_star, A

    def decode(self, c_star, L):
        """
        c_star: [B,N,d]
        return recon: [B,N,L], pred: [B,N]
        """
        B, N, d = c_star.shape
        c_star_flat = c_star.view(B * N, 1, d)
        input_seq = c_star_flat.repeat(1, L, 1)  # [B*N, L, d]

        dec_out, _ = self.decoder_lstm(input_seq)  # [B*N, L, d]

        recon = self.readout_recon(dec_out).view(B, N, L)
        pred = self.readout_pred(dec_out[:, -1, :]).view(B, N)
        return recon, pred

    def forward(self, x):
        """
        x: [B, N, L]
        """
        L = x.shape[-1]
        c = self.get_causal_embedding(x)
        c_star, A = self.spatial_interaction(c)
        recon, pred = self.decode(c_star, L)
        return recon, pred, c_star, A
