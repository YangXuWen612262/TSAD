import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  
        self.pred_len = 1               
        self.enc_in = configs.enc_in    
        self.d_model = configs.d_model  
        self.n_heads = configs.n_heads  
        
        # 1. 独立时间编码器 (Per-variable Encoder)
        self.lstm_enc = nn.LSTM(
            input_size=1, 
            hidden_size=self.d_model, 
            batch_first=True
        )
        
        # 注意力池化
        self.att_pool_w = nn.Linear(self.d_model, 1)
        
        # 2. 潜在空间交互
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=self.d_model, 
            num_heads=self.n_heads, 
            batch_first=True
        )
        
        # 3. 独立解码器
        self.lstm_dec = nn.LSTM(
            input_size=self.d_model, 
            hidden_size=self.d_model, 
            batch_first=True
        )
        # 输出层
        self.projection = nn.Linear(self.d_model, 1)

    def forward(self, x):
        # x shape: [Batch, Length, Channel] (B, L, N)
        B, L, N = x.shape
        
        # --- Step 1: Temporal Modeling ---
        x_reshaped = x.permute(0, 2, 1).contiguous().view(B * N, L, 1)
        
        # Input to Encoder: past L-1 points
        enc_input = x_reshaped[:, :-1, :] # [B*N, L-1, 1]
        
        enc_out, _ = self.lstm_enc(enc_input) # [B*N, L-1, d]
        
        # Attention Pooling
        scores = self.att_pool_w(enc_out)     # [B*N, L-1, 1]
        alpha = F.softmax(scores, dim=1)
        c_i = torch.sum(alpha * enc_out, dim=1) # [B*N, d]
        
        # Reshape to [B, N, d]
        c_i = c_i.view(B, N, self.d_model)
        
        # --- Step 2: Spatial Interaction ---
        # c_star: [B, N, d]
        c_star, _ = self.spatial_attn(c_i, c_i, c_i) 
        
        # --- Step 3: Compute Distance Matrix ---
        c_expanded_1 = c_star.unsqueeze(2) # [B, N, 1, d]
        c_expanded_2 = c_star.unsqueeze(1) # [B, 1, N, d]
        dist_matrix = torch.norm(c_expanded_1 - c_expanded_2, p=2, dim=-1) # [B, N, N]
        
        # --- Step 4: Decoding ---
        # 【修改点】这里将 .view 改为 .reshape，解决报错
        dec_input = c_star.reshape(B * N, 1, self.d_model).repeat(1, L, 1) # [B*N, L, d]
        
        dec_out, _ = self.lstm_dec(dec_input) # [B*N, L, d]
        out_proj = self.projection(dec_out)   # [B*N, L, 1]
        
        # Reshape back to [B, L, N]
        output = out_proj.view(B, N, L).permute(0, 2, 1)
        
        recon = output[:, :-1, :]
        pred = output[:, -1:, :]
        
        return pred, recon, dist_matrix