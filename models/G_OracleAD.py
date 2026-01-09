import torch
import torch.nn as nn

# 这里直接贴你自己的 G_OracleAD（或 from .xxx import G_OracleAD）
import torch.nn.functional as F

class G_OracleAD(nn.Module):
    def __init__(self, num_vars, window_size, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        self.encoder_lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.attn_pool_w = nn.Linear(hidden_dim, 1)

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.decoder_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.readout_recon = nn.Linear(hidden_dim, 1)
        self.readout_pred = nn.Linear(hidden_dim, 1)

    def get_causal_embedding(self, x):
        B, N, L = x.shape
        x_flat = x.view(B * N, L, 1)
        enc_out, _ = self.encoder_lstm(x_flat)
        attn_scores = F.softmax(self.attn_pool_w(enc_out), dim=1)
        c_flat = torch.sum(enc_out * attn_scores, dim=1)
        c = c_flat.view(B, N, self.hidden_dim)
        return c

    def spatial_interaction(self, c):
        Q = self.W_Q(c)
        K = self.W_K(c)
        V = self.W_V(c)
        d_k = c.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        A = F.softmax(scores, dim=-1)
        c_star = torch.matmul(self.dropout(A), V)
        return c_star, A

    def decode(self, c_star, L):
        B, N, d = c_star.shape
        c_star_flat = c_star.view(B * N, 1, d)
        input_seq = c_star_flat.repeat(1, L, 1)
        dec_out, _ = self.decoder_lstm(input_seq)
        recon = self.readout_recon(dec_out).view(B, N, L)
        pred = self.readout_pred(dec_out[:, -1, :]).view(B, N)
        return recon, pred

    def forward(self, x):
        L = x.shape[-1]
        c = self.get_causal_embedding(x)
        c_star, A = self.spatial_interaction(c)
        recon, pred = self.decode(c_star, L)
        return recon, pred, c_star, A


class Model(nn.Module):
    """
    TSLib 适配器：
    - 输入 x_enc: [B, L, C]
    - 内部转成 [B, C, L] 给你原模型
    - 输出 recon: [B, L, C]
    同时把 A 等中间量也返回，方便你在 Exp 里加“梯度软约束”
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        hidden_dim = getattr(configs, "hidden_dim", 128)
        dropout = getattr(configs, "dropout", 0.1)

        self.core = G_OracleAD(
            num_vars=self.enc_in,
            window_size=self.seq_len,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x_enc, *args, **kwargs):
        # x_enc: [B, L, C] -> [B, C, L]
        x = x_enc.transpose(1, 2).contiguous()

        recon, pred, c_star, A = self.core(x)

        # recon: [B, C, L] -> [B, L, C]
        recon = recon.transpose(1, 2).contiguous()

        # pred: [B, C]（你要不要用看你的 loss 设计）
        return recon, pred, c_star, A
