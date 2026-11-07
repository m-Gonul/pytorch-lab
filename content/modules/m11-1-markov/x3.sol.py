import math
import random
from collections import Counter, defaultdict


def fit(text, k):
    m = defaultdict(Counter)
    for i in range(len(text) - k):
        m[text[i:i + k]][text[i + k]] += 1
    return dict(m)


def log_prob(model, text, k, alpha=1.0, vocab=None):
    V = len(vocab) if vocab else len({c for cnt in model.values() for c in cnt})
    lp = 0.0
    for i in range(k, len(text)):
        c = model.get(text[i - k:i], Counter())
        lp += math.log((c[text[i]] + alpha) / (sum(c.values()) + alpha * V))
    return lp


def generate(model, seed_text, n, k, seed=0):
    rng = random.Random(seed)
    out = seed_text
    keys = sorted(model)
    for _ in range(n):
        ctx = out[-k:] if k else ""
        c = model.get(ctx)
        if not c:
            c = model[rng.choice(keys)]
        chars, w = zip(*sorted(c.items()))
        out += rng.choices(chars, weights=w)[0]
    return out


m = fit("abracadabra", 1)
print(m["a"])
