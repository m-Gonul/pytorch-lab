import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad


def dp_sgd_step(model, x, y, max_norm, noise_mult, lr, generator=None):
    """Örnekler üzerinde döngü yok."""
    ...


torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
x, y = torch.randn(64, 10), torch.randint(0, 2, (64,))
for _ in range(100):
    dp_sgd_step(model, x, y, max_norm=1.0, noise_mult=0.5, lr=0.5)
print(F.cross_entropy(model(x), y).item())
