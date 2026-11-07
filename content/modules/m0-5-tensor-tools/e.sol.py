import torch
import torch.nn.functional as F
from torch.utils import benchmark


def moving_average_fast(x, w):
    """x: (n,) → (n-w+1,). Döngü yok."""
    c = torch.cat([x.new_zeros(1, dtype=torch.float64), x.double().cumsum(0)])
    return ((c[w:] - c[:-w]) / w).to(x.dtype)


def time_fn(fn, *args):
    """Medyan süre (saniye), torch.utils.benchmark ile."""
    t = benchmark.Timer(stmt="fn(*args)", globals={"fn": fn, "args": args})
    return t.blocked_autorange(min_run_time=0.2).median


x = torch.randn(100_000)
print(moving_average_fast(x, 5)[:3])
print(time_fn(moving_average_fast, x, 5))
