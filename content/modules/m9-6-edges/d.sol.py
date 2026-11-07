import torch
import torch.nn.functional as F


def make_batch(B, L, g, noise, n):
    """Gürültülü parça parça sabit sinyaller ve yumuşak basamak hedefi."""
    jumps = (torch.rand(B, L, generator=g) < 0.03).float() * torch.randn(B, L, generator=g)
    x = jumps.cumsum(1) + noise * torch.randn(B, L, generator=g)
    t = torch.arange(n) - n // 2
    gauss = torch.exp(-t.float() ** 2 / 8) / (2 * (2 * torch.pi) ** 0.5)
    target = F.conv1d(jumps[:, None], gauss[None, None], padding=n // 2)[:, 0]
    return x, target


def dog_kernel(sigma, n):
    x = (torch.arange(n) - n // 2).double()
    return x / sigma ** 2 * torch.exp(-x ** 2 / (2 * sigma ** 2))


def learn_edge_filter(n=21, steps=1500, noise=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([w], lr=1e-2)
    for _ in range(steps):
        x, t = make_batch(64, 256, g, noise, n)
        y = F.conv1d(x[:, None], w[None, None], padding=n // 2)[:, 0]
        loss = F.mse_loss(y[:, n:-n], t[:, n:-n])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach()


def sobel_torch(img):
    k = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=img.dtype)[:, None]
    out = F.conv2d(F.pad(img[None, None], (1, 1, 1, 1), mode="replicate"), k)[0]
    return out[0], out[1]


print(dog_kernel(2.0, 7))
