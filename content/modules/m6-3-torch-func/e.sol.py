import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad


def dp_sgd_step(model, x, y, max_norm, noise_mult, lr, generator=None):
    """Örnekler üzerinde döngü yok."""
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_one(p, xi, yi):
        return F.cross_entropy(functional_call(model, p, (xi[None],)), yi[None])
    G = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)
    B = x.shape[0]
    norms = torch.sqrt(sum(g.reshape(B, -1).pow(2).sum(1) for g in G.values()))
    scale = (max_norm / (norms + 1e-12)).clamp(max=1.0)
    with torch.no_grad():
        for k, p in model.named_parameters():
            g = (G[k] * scale.view(-1, *[1] * (G[k].dim() - 1))).sum(0)
            g = g + noise_mult * max_norm * torch.randn(g.shape, generator=generator, dtype=g.dtype)
            p -= lr * g / B


torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
x, y = torch.randn(64, 10), torch.randint(0, 2, (64,))
for _ in range(100):
    dp_sgd_step(model, x, y, max_norm=1.0, noise_mult=0.5, lr=0.5)
print(F.cross_entropy(model(x), y).item())
