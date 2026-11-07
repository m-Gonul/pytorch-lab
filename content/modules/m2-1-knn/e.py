import torch
import torch.nn.functional as F


def predict_many_k(D, ytr, ks, C):
    """D: (nte, ntr), ks: (m,) uzun tensör → (m, nte) tahminler. Döngü yok."""
    ...


D = torch.tensor([[5.0, 1.0, 5.0, 13.0]])
ytr = torch.tensor([0, 1, 0, 1])
print(predict_many_k(D, ytr, torch.tensor([1, 2, 3, 4]), 2))   # [[1],[0],[0],[0]]
