import torch


def rowwise_dot(U, V):
    """(n, d), (n, d) → (n,)"""
    return torch.einsum("nd,nd->n", U, V)


def pairwise_cosine(Q, K):
    """(n, d), (m, d) → (n, m)"""
    Qn = Q / Q.norm(dim=1, keepdim=True)
    Kn = K / K.norm(dim=1, keepdim=True)
    return torch.einsum("nd,md->nm", Qn, Kn)


U, V = torch.randn(5, 3), torch.randn(5, 3)
print(rowwise_dot(U, V))
print(pairwise_cosine(U, V).shape)  # torch.Size([5, 5])
