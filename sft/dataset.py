from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from settings import settings
from tokenizer.tokenizer import Tokenizer

tokenizer = Tokenizer()

class InstructDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        instr = row["instruction"]
        inp = row["input"] or ""
        outp = row["output"] or ""

        if inp.strip():
            text = f"<|startoftext|>Human: {instr}\nInput: {inp}\nAssistant: {outp}<|endoftext|>"
        else:
            text = f"<|startoftext|>Human: {instr}\nAssistant: {outp}<|endoftext|>"

        return text
    

def instruct_collate_fn(batch, tokenizer, max_length=settings.max_length):
    enc = tokenizer.tokenize(
        batch,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]


fine_tuning_ds = load_dataset("hakurei/open-instruct-v1")
fine_tuning_training_data = fine_tuning_ds["train"]

instruct_dataset = InstructDataset(
    hf_dataset=fine_tuning_training_data,
    tokenizer=tokenizer,
    max_length=settings.max_length
)
instruct_dataloader = DataLoader(
    instruct_dataset,
    batch_size=settings.batch_size,
    shuffle=True,
    collate_fn=lambda x: instruct_collate_fn(x, tokenizer, max_length=settings.max_length)
)
