import torch
import torch.nn.functional as F
from torch.utils import benchmark


def moving_average_fast(x, w):
    """x: (n,) → (n-w+1,). Döngü yok."""
    ...


def time_fn(fn, *args):
    """Medyan süre (saniye), torch.utils.benchmark ile."""
    ...


x = torch.randn(100_000)
print(moving_average_fast(x, 5)[:3])
print(time_fn(moving_average_fast, x, 5))
