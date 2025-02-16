from datasets import load_dataset
from torch.utils.data import DataLoader

from settings import settings
from tokenizer.tokenizer import Tokenizer

ds = load_dataset("wikimedia/wikipedia", "20231101.en")
tokenizer = Tokenizer()


def collate_fn(batch):
    texts = [item["text"][:settings.max_length] for item in batch]
    encoding = tokenizer.tokenize(
        texts,
        padding="max_length",
        truncation=True,
        max_length=settings.max_length,
        return_attention_mask=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    return input_ids, attention_mask


train_dataset = ds["train"]
train_dataloader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, collate_fn=collate_fn)
