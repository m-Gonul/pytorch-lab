import torch
import torch.nn.functional as F


def fit_mnb(X, y, C, alpha=1.0):
    """X: (n, V) sayımlar → (log_prior: (C,), log_theta: (C, V))"""
    ...


def predict_mnb(model, X):
    """→ (n, C) normalize log-olasılık"""
    ...


X = torch.tensor([[3, 0, 1, 0], [2, 0, 0, 1], [0, 2, 0, 3], [0, 3, 1, 2]], dtype=torch.float32)
y = torch.tensor([0, 0, 1, 1])
print(predict_mnb(fit_mnb(X, y, 2), X).exp())
