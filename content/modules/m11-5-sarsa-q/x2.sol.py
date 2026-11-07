import math
import random
N = 10


def chain_step(s, a):
    if a == 0:
        return 0, (0.05 if s == 0 else 0.0)
    if s == N - 1:
        return 0, 1.0
    return s + 1, 0.0


def run(strategy, Q0=0.0, n_steps=5000, gamma=0.95, alpha=0.2, seed=0):
    rng = random.Random(seed)
    Q = [[Q0, Q0] for _ in range(N)]
    s, total = 0, 0.0
    for _ in range(n_steps):
        q = Q[s]
        greedy = rng.randrange(2) if q[0] == q[1] else int(q[1] > q[0])
        if strategy == "eps":
            a = rng.randrange(2) if rng.random() < 0.1 else greedy
        elif strategy == "boltzmann":
            z = max(min((q[1] - q[0]) / 0.1, 50.0), -50.0)
            a = int(rng.random() < 1 / (1 + math.exp(-z)))
        else:
            a = greedy
        s2, r = chain_step(s, a)
        Q[s][a] += alpha * (r + gamma * max(Q[s2]) - Q[s][a])
        total += r
        s = s2
    return total / n_steps, all(Q[i][1] > Q[i][0] for i in range(N))


print(run("eps"), run("greedy", Q0=2.0))
