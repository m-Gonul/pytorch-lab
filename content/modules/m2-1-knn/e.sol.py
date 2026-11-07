import torch
import torch.nn.functional as F


def predict_many_k(D, ytr, ks, C):
    """D: (nte, ntr), ks: (m,) uzun tensör → (m, nte) tahminler. Döngü yok."""
    kmax = int(ks.max())
    idx = torch.sort(D, dim=1, stable=True).indices[:, :kmax]   # (nte, kmax)
    V = F.one_hot(ytr[idx], C).cumsum(dim=1)                    # (nte, kmax, C)
    return V[:, ks - 1, :].argmax(-1).T                         # (nte, m) → (m, nte)


D = torch.tensor([[5.0, 1.0, 5.0, 13.0]])
ytr = torch.tensor([0, 1, 0, 1])
print(predict_many_k(D, ytr, torch.tensor([1, 2, 3, 4]), 2))   # [[1],[0],[0],[0]]
