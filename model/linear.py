import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, 
                 r=4, alpha=8, dropout=0.0, bias=True,
                 enable_lora=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.enable_lora = enable_lora

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))

        if not self.enable_lora:
            self.A.requires_grad = False
            self.B.requires_grad = False

        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.einsum("bsi,oi->bso", x, self.base_weight)

        if self.enable_lora:
            lora_out = torch.einsum("bsi,ri,or->bso", self.dropout(x), self.A, self.B)
            lora_out = lora_out * self.scaling
            out = out + lora_out

        if self.use_bias and self.bias is not None:
            out = out + self.bias
        return out
