import math
import random


def block_params(d):
    return 12 * d * d + 13 * d


def gpt_params(V, T, d, n_layer):
    return V * d + T * d + n_layer * block_params(d) + 2 * d


def probs(logits, temperature=1.0, top_k=None):
    z = [x / temperature for x in logits]
    if top_k is not None:
        keep = set(sorted(range(len(z)), key=lambda i: -z[i])[:top_k])
        z = [x if i in keep else -math.inf for i, x in enumerate(z)]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def sample(logits, u, temperature=1.0, top_k=None):
    if temperature == 0:
        return max(range(len(logits)), key=lambda i: logits[i])
    cdf = 0.0
    p = probs(logits, temperature, top_k)
    for i, pi in enumerate(p):
        cdf += pi
        if u < cdf:
            return i
    return max(i for i, pi in enumerate(p) if pi > 0)


def generate(next_logits, prompt, n, temperature=1.0, top_k=None, rng=None):
    rng = rng or random.Random(0)
    toks = list(prompt)
    for _ in range(n):
        toks.append(sample(next_logits(toks), rng.random(), temperature, top_k))
    return toks


print(gpt_params(50257, 1024, 768, 12))        # 124439808
print(probs([2, 1, 0], top_k=2))               # [0.731, 0.269, 0]
bigram = lambda toks: [3.0 if i == (toks[-1] + 1) % 5 else 0.0 for i in range(5)]
print(generate(bigram, [0], 8, temperature=0.7, rng=random.Random(0)))
