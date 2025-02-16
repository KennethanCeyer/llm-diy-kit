from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    num_epochs_pretrain: int = 5
    num_epochs_sft: int = 3
    vocab_size: int = 50257
    embed_size: int = 1920
    num_layers: int = 8
    batch_size: int = 8
    num_heads: int = 12
    forward_expansion: int = 4
    dropout: float = 0.1
    max_length: int = 1024


settings = Settings()
