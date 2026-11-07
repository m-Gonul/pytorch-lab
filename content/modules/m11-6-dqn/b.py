import random


def walk_episode(rng, n=1000, jump=100):
    ...


def group(s, k=10):
    ...


def gradient_mc(episodes, alpha, k=10):
    ...


def semi_gradient_td(episodes, alpha, k=10):
    ...


rng = random.Random(0)
eps = [walk_episode(rng) for _ in range(200)]
print(semi_gradient_td(eps, 2e-3))
