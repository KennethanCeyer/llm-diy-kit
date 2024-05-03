from transformers import Trainer, TrainingArguments
from model.transformer import TransformerConfig, HuggingFaceTransformer
from settings import settings
from dataset import train_dataset
import torch

config = TransformerConfig(
    vocab_size=settings.src_vocab_size,
    embed_size=settings.embed_size,
    num_layers=settings.num_layers,
    heads=settings.heads,
    forward_expansion=settings.forward_expansion,
    dropout=settings.dropout,
    max_length=settings.max_length,
    device=(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    ),
    prompt_length=settings.prompt_length,
    rank=settings.rank,
)

model = HuggingFaceTransformer(config)
training_args = TrainingArguments(
    output_dir="./model_save",
    overwrite_output_dir=True,
    num_train_epochs=settings.epochs,
    per_device_train_batch_size=settings.batch_size,
    save_steps=settings.save_steps,
    save_total_limit=settings.total_limit,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
