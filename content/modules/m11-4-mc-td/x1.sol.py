import random
import numpy as np


def draw(rng):
    return min(rng.randint(1, 13), 10)


def _total(cards):
    s = sum(cards)
    usable = 1 in cards and s + 10 <= 21
    return (s + 10 if usable else s), usable


def play(rng):
    player = [draw(rng), draw(rng)]
    dealer = [draw(rng), draw(rng)]
    states = []
    while True:
        t, u = _total(player)
        if t > 21:
            return states, -1
        if t >= 12:
            states.append((int(u), t, dealer[0]))
        if t >= 20:
            break
        player.append(draw(rng))
    while _total(dealer)[0] < 17:
        dealer.append(draw(rng))
    p, d = _total(player)[0], _total(dealer)[0]
    return states, (1 if d > 21 or p > d else (0 if p == d else -1))


def mc_blackjack(n, seed=0):
    rng = random.Random(seed)
    S = np.zeros((2, 10, 10)); C = np.zeros((2, 10, 10))
    for _ in range(n):
        states, r = play(rng)
        for u, t, d in states:
            S[u, t - 12, d - 1] += r; C[u, t - 12, d - 1] += 1
    return S / np.maximum(C, 1)


print(mc_blackjack(1000)[0, 9, 0])
