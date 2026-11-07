import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        self.p, self.lr, (self.b1, self.b2), self.eps = params, lr, betas, eps
        self.wd, self.decoupled, self.t = weight_decay, decoupled, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, w in self.p.items():
            g = grads[k]
            if self.decoupled:
                w *= 1 - self.lr * self.wd
            else:
                g = g + self.wd * w
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            w -= self.lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)


p = {"w": np.zeros(2)}
opt = Adam(p, lr=0.1)
for _ in range(100):
    opt.step({"w": 2 * (p["w"] - np.array([3.0, -1.0]))})
print(p["w"])
