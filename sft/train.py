import torch
from torch import nn, optim

from model.transformer import Transformer
from settings import settings
from sft.dataset import instruct_dataloader

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
    enable_lora=True,
).to(device)
state_dict = torch.load("pretrained_model.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
for name, param in model.named_parameters():
    if "A" in name or "B" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss(ignore_index=-100)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=1e-4)


if __name__ == "__main__":
    model.train()
    for epoch in range(settings.num_epochs_sft):
        total_loss = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(instruct_dataloader):
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

            total_loss += loss.item()

            if (batch_idx+1) % 100 == 0:
                print(f"[SFT] Epoch [{epoch+1}/{settings.num_epochs_sft}], Step [{batch_idx+1}/{len(instruct_dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(instruct_dataloader)
        print(f"[SFT] Epoch [{epoch+1}/{settings.num_epochs_sft}] Average Loss: {avg_loss:.4f}")

        model_path = f"sft_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"=== SFT Done for Epoch {epoch+1}. Saved => {model_path} ===")
