import torch


class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=heads, batch_first=True
        )
        self.norm1 = torch.nn.LayerNorm(embed_size)
        self.norm2 = torch.nn.LayerNorm(embed_size)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_size, forward_expansion * embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(query, key, value, attn_mask=mask)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
