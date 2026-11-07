import torch
import torch.nn.functional as F


def alpha_sweep(Xtr, ytr, Xva, yva, C, alphas):
    """→ (A,) doğrulama doğrulukları. Döngü yok."""
    Y = F.one_hot(ytr, C).to(Xtr.dtype)
    N = Y.T @ Xtr                                                  # (C, V)
    a = alphas.to(Xtr.dtype)[:, None, None]                        # (A, 1, 1)
    log_theta = torch.log(N + a) - torch.log(N.sum(1, keepdim=True) + a * Xtr.shape[1])   # (A, C, V)
    log_prior = torch.log(Y.sum(0) / len(ytr))
    S = torch.einsum("nv,acv->anc", Xva, log_theta) + log_prior    # (A, n, C)
    return (S.argmax(2) == yva).to(Xtr.dtype).mean(1)


g = torch.Generator().manual_seed(0)
Xtr = torch.randint(0, 3, (200, 50), generator=g).float()
ytr = torch.randint(0, 2, (200,), generator=g)
print(alpha_sweep(Xtr, ytr, Xtr, ytr, 2, torch.logspace(-3, 1, 5)))
