import math
import random


def walk_episode(rng):
    ...


def mc(episodes, alpha, V0=0.5):
    ...


def td0(episodes, alpha, V0=0.5):
    ...


def rms(V):
    ...


rng = random.Random(0)
eps = [walk_episode(rng) for _ in range(100)]
print(rms(td0(eps, 0.1)[0]), rms(mc(eps, 0.02)[0]))
