import torch
import torch.nn.functional as F


def sqdist_torch(Xte, Xtr):
    te = (Xte ** 2).sum(1, keepdim=True)          # (nte, 1)
    tr = (Xtr ** 2).sum(1)                        # (ntr,) → sağdan hizalanır
    return (te + tr - 2 * Xte @ Xtr.T).clamp_min(0)


def knn_predict_torch(Xte, Xtr, ytr, k, C):
    D = sqdist_torch(Xte, Xtr)
    idx = D.topk(k, dim=1, largest=False).indices            # (nte, k)
    return F.one_hot(ytr[idx], C).sum(1).argmax(1)


torch.manual_seed(0)
Xtr, Xte = torch.randn(100, 2), torch.randn(5, 2)
ytr = (Xtr[:, 0] > 0).long()
print(knn_predict_torch(Xte, Xtr, ytr, 5, 2), (Xte[:, 0] > 0).long())
