import math


def build_vocab(sentences, min_count=1):
    cnt = {}
    for s in sentences:
        for w in s:
            cnt[w] = cnt.get(w, 0) + 1
    words = sorted((w for w in cnt if cnt[w] >= min_count), key=lambda w: (-cnt[w], w))
    return {w: i for i, w in enumerate(words)}


def oov_hash(word):
    h = 0
    for ch in word:
        h = (31 * h + ord(ch)) % 2 ** 32
    return h


def encode(tokens, vocab, n_oov):
    return [vocab[w] if w in vocab else len(vocab) + oov_hash(w) % n_oov for w in tokens]


def skipgram_pairs(ids, window):
    out = []
    for i, c in enumerate(ids):
        for j in range(max(0, i - window), min(len(ids), i + window + 1)):
            if j != i:
                out.append((c, ids[j]))
    return out


def cbow_examples(ids, window):
    return [([ids[j] for j in range(max(0, i - window), min(len(ids), i + window + 1)) if j != i], c) for i, c in enumerate(ids)]


def _sig(x):
    return 1 / (1 + math.exp(-x))


def sgns_loss_grad(v, u_o, U_neg):
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    so = _sig(dot(u_o, v))
    sn = [_sig(dot(u, v)) for u in U_neg]
    J = -math.log(so) - sum(math.log(1 - s) for s in sn)
    dv = [(so - 1) * a for a in u_o]
    for s, u in zip(sn, U_neg):
        dv = [x + s * y for x, y in zip(dv, u)]
    du_o = [(so - 1) * a for a in v]
    du_k = [[s * a for a in v] for s in sn]
    return J, dv, du_o, du_k


def noise_dist(counts, power=0.75):
    p = [c ** power for c in counts]
    z = sum(p)
    return [x / z for x in p]


def discard_prob(f, t=1e-5):
    return max(0.0, 1 - math.sqrt(t / f))


print(skipgram_pairs([0, 1, 2, 3, 4], 2))
