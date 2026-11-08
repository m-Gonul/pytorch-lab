import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.cap, self.data, self.pos = capacity, None, 0
        self.n = 0

    def add(self, batch):
        if self.data is None:
            self.data = np.zeros((self.cap,) + batch.shape[1:], batch.dtype)
        for row in batch:
            self.data[self.pos] = row
            self.pos = (self.pos + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, k, rng):
        return self.data[rng.choice(self.n, k, replace=False)]

    def __len__(self):
        return self.n


def mixed_fakes(current, buffer, frac, rng):
    k = int(round(frac * len(current))) if len(buffer) else 0
    k = min(k, len(buffer))
    out = np.concatenate([buffer.sample(k, rng), current[:len(current) - k]]) if k else current.copy()
    buffer.add(current)
    return out
