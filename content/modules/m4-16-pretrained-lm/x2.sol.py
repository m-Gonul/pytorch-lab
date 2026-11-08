import math
import random


def poisson(lam, rng):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def text_infill(tokens, ratio, lam, rng, mask="<mask>"):
    n = len(tokens)
    target = int(round(ratio * n))
    masked = [False] * n
    starts = {}
    count = 0
    tries = 0
    while count < target and tries < 10 * n:
        tries += 1
        L = min(poisson(lam, rng), target - count)
        s = rng.randrange(n)
        if any(masked[s:s + L]) or s + L > n or s in starts:
            continue
        for i in range(s, s + L):
            masked[i] = True
        starts[s] = L
        count += L
    out = []
    i = 0
    while i < n:
        if i in starts:
            out.append(mask)
            if starts[i] == 0:
                out.append(tokens[i])
            i += max(starts[i], 1)
        else:
            out.append(tokens[i])
            i += 1
    return out


def token_deletion(tokens, p, rng):
    return [t for t in tokens if rng.random() >= p]


def sentence_permutation(sentences, rng):
    s = list(sentences)
    rng.shuffle(s)
    return s


def document_rotation(tokens, rng):
    k = rng.randrange(len(tokens))
    return tokens[k:] + tokens[:k]
