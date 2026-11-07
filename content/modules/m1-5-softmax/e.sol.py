import torch


def softmax_jacobian(Z):
    """Z: (n, K) → (n, K, K); J[i] = diag(p) − p pᵀ. Döngü yok."""
    P = torch.softmax(Z, dim=1)
    return torch.diag_embed(P) - P[:, :, None] * P[:, None, :]


def confusion_matrix(y_true, y_pred, K):
    """→ (K, K) tamsayı; satır gerçek, sütun tahmin. Döngü yok."""
    return torch.bincount(K * y_true + y_pred, minlength=K * K).view(K, K)


Z = torch.tensor([[2.0, 1.0, 0.0]])
print(softmax_jacobian(Z))
print(confusion_matrix(torch.tensor([0, 0, 1, 2, 2]), torch.tensor([0, 1, 1, 2, 0]), 3))
