import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.0):
        super(MixerBlock, self).__init__()
        # Token Mixing (Time mixing)
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mix = MLP(seq_len, seq_len * 2, seq_len, dropout)
        
        # Channel Mixing (Feature mixing)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mix = MLP(d_model, d_model * 2, d_model, dropout)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, D_Model]
        
        # Token Mixing (Time dimension)
        # Transpose for MLP on time dimension: [B, D, S]
        y = self.norm1(x).transpose(1, 2)
        y = self.token_mix(y).transpose(1, 2)
        x = x + y
        
        # Channel Mixing (Feature dimension)
        y = self.norm2(x)
        y = self.channel_mix(y)
        x = x + y
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len 
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.layer_num = configs.e_layers 

        # Embedding: Map input channels to d_model
        self.embedding = nn.Linear(self.enc_in, self.d_model)
        
        # Mixer Layers
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(self.seq_len, self.d_model, self.dropout)
            for _ in range(self.layer_num)
        ])
        
        # Projection to prediction (reconstruct original channels)
        self.projection = nn.Linear(self.d_model, self.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Channels]
        
        x = self.embedding(x_enc)
        
        for block in self.mixer_blocks:
            x = block(x)
            
        # GCAD predicts the feature at the last timestamp (or next step)
        # We take the last token representation
        # x: [Batch, Seq_Len, D_Model] -> [Batch, D_Model]
        output = self.projection(x[:, -1, :]) 
        
        # Output shape: [Batch, Channels]
        return output