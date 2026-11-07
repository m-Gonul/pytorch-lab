import math
import torch


def iwae_bound(encode, decode, x, eps):
    """eps: (K, B, L) → skaler. Döngü yok."""
    ...


torch.manual_seed(0)
enc = torch.nn.Linear(6, 4); dec = torch.nn.Linear(2, 6)
encode = lambda x: enc(x).chunk(2, -1)
x = torch.randn(8, 6)
for K in (1, 10, 100):
    print(K, iwae_bound(encode, dec, x, torch.randn(K, 8, 2)).item())
