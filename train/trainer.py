import torch
from torch import nn, optim

from model.transformer import Transformer
from settings import settings
from train.dataset import train_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    vocab_size=settings.vocab_size,
    embed_size=settings.embed_size,
    num_layers=settings.num_layers,
    num_heads=settings.num_heads,
    forward_expansion=settings.forward_expansion,
    dropout=settings.dropout,
    max_length=settings.max_length,
    device=device,
    lora_r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    enable_lora=False
).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


if __name__ == "__main__":
    model.train()
    for epoch in range(settings.num_epochs_pretrain):
        epoch_loss = 0.0
        for batch_idx, (input_ids, attention_mask) in enumerate(train_dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:].clone()
            labels[:, -1] = -100
            
            key_padding_mask = (attention_mask == 0)
            
            optimizer.zero_grad()
            outputs = model(input_ids, key_padding_mask)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (batch_idx+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{settings.num_epochs_pretrain}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{settings.num_epochs_pretrain}] Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), "pretrained_model.pth")
        print("=== Pretrain Done. Saved => pretrained_model.pth ===")
