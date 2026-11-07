import torch
import torch.nn.functional as F


def fit_mnb(X, y, C, alpha=1.0):
    """X: (n, V) sayımlar → (log_prior: (C,), log_theta: (C, V))"""
    Y = F.one_hot(y, C).to(X.dtype)                   # (n, C)
    N = Y.T @ X                                       # (C, V)
    log_theta = torch.log(N + alpha) - torch.log(N.sum(1, keepdim=True) + alpha * X.shape[1])
    log_prior = torch.log(Y.sum(0) / len(y))
    return log_prior, log_theta


def predict_mnb(model, X):
    """→ (n, C) normalize log-olasılık"""
    log_prior, log_theta = model
    S = X @ log_theta.T + log_prior
    return S - torch.logsumexp(S, dim=1, keepdim=True)


X = torch.tensor([[3, 0, 1, 0], [2, 0, 0, 1], [0, 2, 0, 3], [0, 3, 1, 2]], dtype=torch.float32)
y = torch.tensor([0, 0, 1, 1])
print(predict_mnb(fit_mnb(X, y, 2), X).exp())
