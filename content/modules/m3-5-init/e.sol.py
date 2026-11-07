import torch


def std_vs_scale(X, Gs, scales, act):
    """Gs: (depth, width, width), scales: (S,) → (S, depth). Ölçekler üzerinde döngü yok."""
    width = X.shape[1]
    H = X.expand(scales.shape[0], *X.shape)                    # (S, N, width)
    c = (scales / width ** 0.5)[:, None, None]
    out = []
    for G in Gs:
        H = act(c * (H @ G.T))
        out.append(H.std(dim=(1, 2)))
    return torch.stack(out, 1)


g = torch.Generator().manual_seed(0)
X = torch.randn(256, 128, generator=g)
Gs = torch.randn(20, 128, 128, generator=g)
scales = torch.linspace(0.5, 2.5, 64)
print(std_vs_scale(X, Gs, scales, torch.relu)[:, -1])
