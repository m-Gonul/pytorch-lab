import torch


def layout(t):
    return tuple(t.shape), t.stride(), t.storage_offset(), t.is_contiguous()


def same_storage(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def flatten_transposed(x):
    return x.T.reshape(-1)          # contiguous değil → reshape kopyalar


def windows_torch(x, w):
    return x.unfold(0, w, 1)


A = torch.arange(6.0).reshape(2, 3)
print(layout(A), layout(A.T))
print(flatten_transposed(A))       # tensor([0., 3., 1., 4., 2., 5.])
print(windows_torch(torch.arange(6.0), 3))
