from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    num_epochs_pretrain: int = 5
    num_epochs_sft: int = 5
    vocab_size: int = 50257
    embed_size: int = 512
    num_layers: int = 4
    batch_size: int = 32
    num_heads: int = 4
    forward_expansion: int = 4
    dropout: float = 0.1
    max_length: int = 512


settings = Settings()
