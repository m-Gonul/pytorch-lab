import torch


def layout(t):
    ...


def same_storage(a, b):
    ...


def flatten_transposed(x):
    ...


def windows_torch(x, w):
    ...


A = torch.arange(6.0).reshape(2, 3)
print(layout(A), layout(A.T))
print(flatten_transposed(A))       # tensor([0., 3., 1., 4., 2., 5.])
print(windows_torch(torch.arange(6.0), 3))
