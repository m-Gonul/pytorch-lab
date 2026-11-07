import torch
import torch.nn.functional as F


def alpha_sweep(Xtr, ytr, Xva, yva, C, alphas):
    """→ (A,) doğrulama doğrulukları. Döngü yok."""
    ...


g = torch.Generator().manual_seed(0)
Xtr = torch.randint(0, 3, (200, 50), generator=g).float()
ytr = torch.randint(0, 2, (200,), generator=g)
print(alpha_sweep(Xtr, ytr, Xtr, ytr, 2, torch.logspace(-3, 1, 5)))
