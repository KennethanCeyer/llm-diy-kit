import torch
from torch import nn

from model.transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, 
                 vocab_size, embed_size, num_layers, num_heads,
                 forward_expansion, dropout, max_length, device,
                 lora_r=4, lora_alpha=8, lora_dropout=0.0,
                 enable_lora=False):
        super().__init__()
        self.device = device
        self.embed_size = embed_size
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(max_length, embed_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_size=embed_size,
                num_heads=num_heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=enable_lora
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, x, key_padding_mask=None):
        B, S = x.shape
        pos_ids = torch.arange(0, S, device=self.device).unsqueeze(0).expand(B, S)
        x = self.token_emb(x) + self.pos_emb(pos_ids)

        attn_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=self.device), diagonal=1)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        logits = self.lm_head(x)
        return logits
