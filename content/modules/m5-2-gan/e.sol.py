import torch


def gradient_penalty(D, real, fake, alpha):
    """alpha: (B, 1). Döngü yok."""
    x_hat = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    g, = torch.autograd.grad(D(x_hat).sum(), x_hat, create_graph=True)
    return ((g.flatten(1).norm(dim=1) - 1) ** 2).mean()


torch.manual_seed(0)
D = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
print(gradient_penalty(D, torch.randn(8, 2), torch.randn(8, 2), torch.rand(8, 1)))
