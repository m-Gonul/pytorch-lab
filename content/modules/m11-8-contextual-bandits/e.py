import torch


def policy_actions(W, X):
    ...


def fit_reward_model(X, A, R, K, lam=1.0):
    ...


def estimates(W, X, A, R, p, Theta):
    ...


g = torch.Generator().manual_seed(0)
X = torch.randn(1000, 4, generator=g, dtype=torch.float64)
W = torch.randn(10, 3, 4, generator=g, dtype=torch.float64)
print(policy_actions(W, X).shape)
