import math
import torch


def iwae_bound(encode, decode, x, eps):
    """eps: (K, B, L) → skaler. Döngü yok."""
    K, B, L = eps.shape
    mu, logvar = encode(x)
    std = torch.exp(0.5 * logvar)
    z = mu + std * eps
    xhat = decode(z.reshape(K * B, L)).reshape(K, B, -1)
    c = math.log(2 * math.pi)
    log_px = -0.5 * (((x - xhat) ** 2) + c).sum(-1)
    log_pz = -0.5 * (z ** 2 + c).sum(-1)
    log_qz = -0.5 * (eps ** 2 + logvar + c).sum(-1)
    log_w = log_px + log_pz - log_qz
    return (torch.logsumexp(log_w, 0) - math.log(K)).mean()


torch.manual_seed(0)
enc = torch.nn.Linear(6, 4); dec = torch.nn.Linear(2, 6)
encode = lambda x: enc(x).chunk(2, -1)
x = torch.randn(8, 6)
for K in (1, 10, 100):
    print(K, iwae_bound(encode, dec, x, torch.randn(K, 8, 2)).item())
