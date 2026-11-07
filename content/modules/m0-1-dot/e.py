import torch


def rowwise_dot(U, V):
    """(n, d), (n, d) → (n,)"""
    ...


def pairwise_cosine(Q, K):
    """(n, d), (m, d) → (n, m)"""
    ...


U, V = torch.randn(5, 3), torch.randn(5, 3)
print(rowwise_dot(U, V))
print(pairwise_cosine(U, V).shape)  # torch.Size([5, 5])
