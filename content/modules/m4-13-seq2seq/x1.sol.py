import math


def ngrams(tokens, n):
    out = {}
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i + n])
        out[g] = out.get(g, 0) + 1
    return out


def modified_precision(cand, refs, n):
    c = ngrams(cand, n)
    mx = {}
    for r in refs:
        for g, v in ngrams(r, n).items():
            mx[g] = max(mx.get(g, 0), v)
    return sum(min(v, mx.get(g, 0)) for g, v in c.items()), max(sum(c.values()), 0)


def brevity_penalty(c, r):
    return 1.0 if c > r else math.exp(1 - r / c)


def bleu(cand, refs, N=4):
    logs = []
    for n in range(1, N + 1):
        m, t = modified_precision(cand, refs, n)
        if m == 0 or t == 0:
            return 0.0
        logs.append(math.log(m / t))
    c = len(cand)
    r = min((abs(len(x) - c), len(x)) for x in refs)[1]
    return brevity_penalty(c, r) * math.exp(sum(logs) / N)
