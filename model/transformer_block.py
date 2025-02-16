import torch
from torch import nn

from model.linear import LoRALinear


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion,
                 lora_r=4, lora_alpha=8, lora_dropout=0.0,
                 enable_lora=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        hidden_dim = forward_expansion * embed_size
        self.ff1 = LoRALinear(
            in_features=embed_size,
            out_features=hidden_dim,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            bias=True,
            enable_lora=enable_lora
        )
        self.act = nn.SiLU()
        self.ff2 = LoRALinear(
            in_features=hidden_dim,
            out_features=embed_size,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            bias=True,
            enable_lora=enable_lora
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # x shape: (batch, seq_len, embed_size)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        x = self.drop(x)

        ff_out = self.ff2(self.act(self.ff1(x)))
        x = self.norm2(x + ff_out)
        x = self.drop(x)
        return x
