import torch


class MLMHead(torch.nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(MLMHead, self).__init__()
        self.linear = torch.nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        return self.linear(x)
