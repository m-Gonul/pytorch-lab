import torch


def matmul_torch(A, B):
    ...


def matmul_grads(A, B):
    """L = (A @ B).sum() için (dL/dA, dL/dB) döndür — autograd ile."""
    A = A.detach().clone().requires_grad_(True)
    B = B.detach().clone().requires_grad_(True)
    # L'yi hesapla, backward çağır, gradyanları döndür
    ...


A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])
print(matmul_torch(A, B))
print(matmul_grads(A, B))
