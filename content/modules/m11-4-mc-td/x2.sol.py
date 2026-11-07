import random
import math


def run(alpha_fn, n_episodes, V0, seed=0):
    rng = random.Random(seed)
    V = [0.0] + [V0] * 5 + [0.0]
    cnt = [0] * 7
    out = []
    for _ in range(n_episodes):
        s = 3
        while s not in (0, 6):
            s2 = s + (1 if rng.random() < 0.5 else -1)
            cnt[s] += 1
            V[s] += alpha_fn(cnt[s]) * ((1.0 if s2 == 6 else 0.0) + V[s2] - V[s])
            s = s2
        out.append(math.sqrt(sum((V[i] - i / 6) ** 2 for i in range(1, 6)) / 5))
    return out


print(run(lambda n: 1 / n, 50, 0.0)[-1])
