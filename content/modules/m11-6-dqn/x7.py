import math


class PortableRNG:
    """Verilen Mulberry32 aracı: C++/Python/JS deney akışını eşler."""
    def __init__(self, seed):
        self.state = seed

    def random(self):
        self.state = (self.state + 0x6d2b79f5) & 0xffffffff
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xffffffff
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xffffffff)) & 0xffffffff
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296


def baird_q(steps=100000, alpha=.01, seed=1):
    # TODO: 16 parametre, sabit davranış ve gerçek Q-learning max hedefi.
    raise NotImplementedError


print(baird_q(10000)["norms"][-1])
