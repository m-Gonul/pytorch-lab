import torch


def gradient_penalty(D, real, fake, alpha):
    """alpha: (B, 1). Döngü yok."""
    ...


torch.manual_seed(0)
D = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
print(gradient_penalty(D, torch.randn(8, 2), torch.randn(8, 2), torch.rand(8, 1)))
