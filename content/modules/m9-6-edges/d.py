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
    ...


def learn_edge_filter(n=21, steps=1500, noise=0.3, seed=0):
    ...


def sobel_torch(img):
    ...


print(dog_kernel(2.0, 7))
