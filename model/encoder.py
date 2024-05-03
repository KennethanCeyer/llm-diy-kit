from model.transformer_block import TransformerBlock
import torch


class Encoder(torch.nn.Module):

    def __init__(
        self,
        src_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        device: str,
        forward_expansion: int,
        dropout: float,
        max_length: int,
        prompt_length: int,
        rank: int,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = torch.nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = torch.nn.Embedding(
            max_length + prompt_length, embed_size
        )
        self.prompt_embeddings = torch.nn.Parameter(
            torch.randn(prompt_length, embed_size)
        )

        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    rank=rank,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        total_length = seq_length + self.prompt_embeddings.shape[0]
        positions = (
            torch.arange(0, total_length).expand(N, total_length).to(self.device).long()
        )
        input_embeddings = self.word_embedding(x)
        prompt_embeddings = self.prompt_embeddings.expand(N, -1, -1)
        x = torch.cat([prompt_embeddings, input_embeddings], dim=1)
        position_embeddings = self.position_embedding(positions)
        x = self.dropout(x + position_embeddings)

        for layer in self.layers:
            x = layer(x, x, x, mask)

        return x
