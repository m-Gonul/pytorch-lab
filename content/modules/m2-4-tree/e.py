import torch
import torch.nn.functional as F


def best_split_torch(X, y, C, w=None):
    """Örnek ağırlıklı Gini. → (j, t, skor) ya da None. Döngü yok."""
    ...


X = torch.tensor([[1.0], [2], [3], [4], [5], [6]])
y = torch.tensor([0, 0, 1, 1, 1, 0])
print(best_split_torch(X, y, 2))
print(best_split_torch(X, y, 2, torch.tensor([1.0, 1, 1, 1, 1, 10])))
