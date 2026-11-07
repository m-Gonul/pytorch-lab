import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        ...

    def step(self, grads):
        ...


p = {"w": np.zeros(2)}
opt = Adam(p, lr=0.1)
for _ in range(100):
    opt.step({"w": 2 * (p["w"] - np.array([3.0, -1.0]))})
print(p["w"])
