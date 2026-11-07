import math
import random


def run(policy, probs, T, seed=0, eps=0.1, c=5.0):
    rng = random.Random(seed)
    K, best = len(probs), max(probs)
    n, s = [0] * K, [0.0] * K
    reg, total = [], 0.0
    for t in range(1, T + 1):
        if policy == "ucb":
            a = t - 1 if t <= K else max(range(K), key=lambda i: s[i] / n[i] + math.sqrt(2 * math.log(t) / n[i]))
        elif policy == "ts":
            a = max(range(K), key=lambda i: rng.betavariate(1 + s[i], 1 + n[i] - s[i]))
        else:
            e = eps if policy == "eps" else min(1.0, c / t)
            if rng.random() < e:
                a = rng.randrange(K)
            else:
                est = [s[i] / n[i] if n[i] else 0.0 for i in range(K)]
                m = max(est)
                a = rng.choice([i for i in range(K) if est[i] == m])
        r = 1 if rng.random() < probs[a] else 0
        n[a] += 1
        s[a] += r
        total += best - probs[a]
        reg.append(total)
    return reg


print(run("ts", [0.9, 0.8, 0.5], 1000)[-1])
