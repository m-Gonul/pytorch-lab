import torch


def unbroadcast_torch(G, shape):
    while G.dim() > len(shape):
        G = G.sum(0)
    dims = [i for i, s in enumerate(shape) if s == 1 and G.shape[i] != 1]
    return G.sum(dim=dims, keepdim=True) if dims else G


def manual_grads(a, b, W):
    """L = ReLU((a * b + a) @ W).sum() için (da, db, dW) — autograd YOK."""
    U = a * b + a                      # (n, d)
    Z = U @ W                          # (n, k)
    dZ = (Z > 0).to(Z.dtype)           # ∂ΣReLU/∂Z
    dW = U.T @ dZ
    dU = dZ @ W.T
    da = dU * (b + 1)                  # U = a·(b + 1)
    db = unbroadcast_torch(dU * a, b.shape)
    return da, db, dW
