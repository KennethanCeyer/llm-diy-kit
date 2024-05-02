import torch
from model.mlm_head import MLMHead
from model.encoder import Encoder


class Transformer(torch.nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        self.head = MLMHead(embed_size, src_vocab_size)
        self.device = device

    def forward(self, x, mask):
        encoded = self.encoder(x, mask)
        return self.head(encoded)
