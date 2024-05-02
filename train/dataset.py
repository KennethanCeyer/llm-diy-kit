import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer.tokenizer import Tokenizer
from texts import texts
from settings import settings


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> torch.Tensor:
        text = self.texts[idx]
        tokens = self.tokenizer.tokenize(
            text,
            return_tensors="pt",
            max_length=settings.max_length,
            padding="max_length",
            truncation=True,
        )
        return tokens["input_ids"].squeeze()


tokenizer = Tokenizer()
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True)
