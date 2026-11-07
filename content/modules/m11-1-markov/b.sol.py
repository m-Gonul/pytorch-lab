import random


def sample_index(probs, u):
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if u < c:
            return i
    return len(probs) - 1


def simulate_chain(T, s0, steps, seed=0):
    rng = random.Random(seed)
    s = [s0]
    for _ in range(steps):
        col = [T[i][s[-1]] for i in range(len(T))]
        s.append(sample_index(col, rng.random()))
    return s


def ar1(phi, sigma, n, seed=0):
    rng = random.Random(seed)
    x = [0.0]
    for _ in range(n - 1):
        x.append(phi * x[-1] + rng.gauss(0, sigma))
    return x


def autocov(x, lag):
    n = len(x)
    m = sum(x) / n
    return sum((x[t] - m) * (x[t + lag] - m) for t in range(n - lag)) / n


T = [[0.9, 0.5], [0.1, 0.5]]
s = simulate_chain(T, 0, 10000)
print(s.count(0) / len(s))
