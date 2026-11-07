import torch


def lambda_returns(states, rewards, V, gamma, lam):
    ...


def offline_forward(states, rewards, V, alpha, gamma, lam):
    ...


def offline_traces(states, rewards, V, alpha, gamma, lam):
    ...


def online_traces(states, rewards, V, alpha, gamma, lam):
    ...


V = torch.zeros(7, dtype=torch.float64)
print(offline_traces([3, 4, 5, 6], [0.0, 0.0, 1.0], V, 0.1, 1.0, 0.8))
