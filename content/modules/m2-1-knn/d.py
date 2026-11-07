import torch
import torch.nn.functional as F


def sqdist_torch(Xte, Xtr):
    ...


def knn_predict_torch(Xte, Xtr, ytr, k, C):
    ...


torch.manual_seed(0)
Xtr, Xte = torch.randn(100, 2), torch.randn(5, 2)
ytr = (Xtr[:, 0] > 0).long()
print(knn_predict_torch(Xte, Xtr, ytr, 5, 2), (Xte[:, 0] > 0).long())
