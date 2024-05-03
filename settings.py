from pydantic_settings import BaseSettings
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent


class Settings(BaseSettings):
    epochs: int = 5
    src_vocab_size: int = 250000
    embed_size: int = 768
    num_layers: int = 12
    batch_size: int = 2
    heads: int = 12
    forward_expansion: int = 4
    dropout: float = 0.1
    max_length: int = 512
    total_limit: int = 2
    save_steps: int = 100
    prompt_length: int = 512
    rank: int = 128


PROJECT_ROOT_DIR = get_project_root()
settings = Settings()
