import torch
from torch import nn

from model.transformer_block import TransformerBlock


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, forward_expansion, dropout, max_length, device,
                 lora_r=4, lora_alpha=8, lora_dropout=0.0):
        super(Decoder, self).__init__()
        self.device = device
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, 
                    num_heads, 
                    dropout, 
                    forward_expansion, 
                    lora_r=lora_r, 
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length, device=self.device).unsqueeze(0).expand(batch_size, seq_length)

        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        attn_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=self.device, dtype=torch.bool),
            diagonal=1
        )

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return x
