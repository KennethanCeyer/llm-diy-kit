import torch
from torch import nn, optim

from model.transformer import Transformer
from settings import settings

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
