import torch


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        embed_size: int,
        heads: int,
        dropout: float,
        forward_expansion: int,
        rank: int,
    ):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=heads, batch_first=True
        )

        W0_size = self.attention.in_proj_weight.shape
        self.A = torch.nn.Parameter(torch.randn(W0_size[0], rank))
        self.B = torch.nn.Parameter(torch.randn(rank, W0_size[1]))
        self.norm1 = torch.nn.LayerNorm(embed_size)
        self.norm2 = torch.nn.LayerNorm(embed_size)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_size, forward_expansion * embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.register_parameter("W_prime", torch.nn.Parameter(torch.randn(W0_size)))

    def forward(
        self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        W = self.attention.in_proj_weight + self.A @ self.B
        self.W_prime.data = W
        self.attention.in_proj_weight = self.W_prime

        if mask is not None and mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        attention_output = self.attention(query, key, value, attn_mask=mask)[0]
        x = self.dropout(self.norm1(attention_output + query))
        forward_output = self.feed_forward(x)
        out = self.dropout(self.norm2(forward_output + x))

        return out
