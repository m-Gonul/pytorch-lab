import numpy as np
import torch


class GMM:
    def __init__(self, shape, K=3, alpha=0.01, var0=225.0, T=0.7, w0=0.05):
        ...

    def apply(self, frame):
        ...


g = GMM((4, 5))
f = torch.full((4, 5), 100.0, dtype=torch.float64)
print(g.apply(f), g.apply(f + 120))
