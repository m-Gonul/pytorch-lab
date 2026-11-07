import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        ...

    def forward(self, x):
        ...


class GPT(nn.Module):
    def __init__(self, vocab, block_size, d, n_layer, n_head):
        super().__init__()
        ...

    def forward(self, idx, targets=None):
        ...


torch.manual_seed(0)
m = GPT(vocab=20, block_size=16, d=32, n_layer=2, n_head=4)
logits, loss = m(torch.randint(0, 20, (2, 10)), torch.randint(0, 20, (2, 10)))
print(logits.shape, loss)   # (2, 10, 20), ≈ log 20 = 3.0
