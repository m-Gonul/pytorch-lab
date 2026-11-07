import torch


def kmeans_torch(X, C0, iters):
    ...


def lloyd_step_via_grad(X, C):
    """Atamaları sabitle, inertia'nın gradyanından Newton adımı."""
    ...


X = torch.tensor([[0.0, 0], [1, 0], [4, 4], [5, 4]])
print(kmeans_torch(X, X[:2].clone(), 2))
print(lloyd_step_via_grad(X, torch.tensor([[0.0, 0], [1, 0]])))   # [[0, 0], [3.33, 2.67]]
