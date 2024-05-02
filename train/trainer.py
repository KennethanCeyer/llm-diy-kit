import torch.optim as optim
import torch
from settings import settings
from model.transformer import Transformer
from train.dataset import dataloader, tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    settings.src_vocab_size,
    settings.embed_size,
    settings.num_layers,
    settings.heads,
    settings.forward_expansion,
    settings.dropout,
    settings.max_length,
    device,
).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


for epoch in range(settings.epochs):
    for batch in dataloader:
        batch = batch.to(device)
        trg_mask = generate_square_subsequent_mask(batch.shape[1]).to(device)

        predictions = model(batch, trg_mask)
        loss = criterion(predictions.transpose(1, 2), batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{settings.epochs}], Loss: {loss.item():.4f}")
