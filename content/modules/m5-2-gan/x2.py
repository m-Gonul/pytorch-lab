import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        ...

    def add(self, batch):
        ...

    def sample(self, k, rng):
        ...

    def __len__(self):
        ...


def mixed_fakes(current, buffer, frac, rng):
    ...
