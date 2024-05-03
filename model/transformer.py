import torch
from transformers import PreTrainedModel, PretrainedConfig
from model.mlm_head import MLMHead
from model.encoder import Encoder
from model.utils import generate_square_subsequent_mask


class Transformer(torch.nn.Module):

    def __init__(
        self,
        src_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
        max_length: int,
        device: str,
        prompt_length: int,
        rank: int,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=device,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length,
            prompt_length=prompt_length,
            rank=rank,
        )
        self.head = MLMHead(embed_size, src_vocab_size)
        self.device = device

    def forward(self, x, mask):
        encoded = self.encoder(x, mask)
        return self.head(encoded)


class TransformerConfig(PretrainedConfig):
    model_type = "custom_transformer"

    def __init__(
        self,
        vocab_size=30522,
        embed_size=768,
        num_layers=12,
        heads=12,
        forward_expansion=4,
        dropout=0.1,
        max_length=512,
        device="cpu",
        prompt_length=512,
        rank=128,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        self.max_length = max_length
        self.device = device
        self.prompt_length = prompt_length
        self.rank = rank


class HuggingFaceTransformer(PreTrainedModel):
    config_class = TransformerConfig
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = Transformer(
            config.vocab_size,
            config.embed_size,
            config.num_layers,
            config.heads,
            config.forward_expansion,
            config.dropout,
            config.max_length,
            config.device,
            config.prompt_length,
            config.rank,
        )

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = generate_square_subsequent_mask(
                input_ids.size(1) + self.config.prompt_length
            ).to(input_ids.device)
        return self.transformer(input_ids, attention_mask)
