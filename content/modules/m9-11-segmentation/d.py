import math
import torch


def gmm_em(Z, mu0, iters=50, reg=1e-6):
    ...


Z = torch.randn(200, 2, dtype=torch.float64)
print(gmm_em(Z, Z[:3])[3])
