import math
import random
import numpy as np


def ads(policy, ctr, T, seed=0):
    rng = random.Random(seed)
    K = len(ctr)
    n, s = [0] * K, [0] * K
    clicks = 0
    for t in range(1, T + 1):
        if policy == "ucb":
            a = t - 1 if t <= K else max(range(K), key=lambda i: s[i] / n[i] + math.sqrt(2 * math.log(t) / n[i]))
        else:
            a = max(range(K), key=lambda i: rng.betavariate(1 + s[i], 1 + n[i] - s[i]))
        r = 1 if rng.random() < ctr[a] else 0
        n[a] += 1
        s[a] += r
        clicks += r
    return clicks


def ab_test(clicks_a, n_a, clicks_b, n_b, n_samples=200000, seed=0):
    rng = np.random.default_rng(seed)
    pa = rng.beta(1 + clicks_a, 1 + n_a - clicks_a, n_samples)
    pb = rng.beta(1 + clicks_b, 1 + n_b - clicks_b, n_samples)
    return float((pb > pa).mean()), float(np.maximum(pa - pb, 0).mean())


print(ab_test(30, 1000, 45, 1000))
