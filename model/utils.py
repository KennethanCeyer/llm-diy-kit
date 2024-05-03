import torch


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf")).to(torch.bool)
    return mask
