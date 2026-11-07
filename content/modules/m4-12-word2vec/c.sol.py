import numpy as np


def toy_corpus(n_sent=6000, seed=0):
    rng = np.random.default_rng(seed)
    people = {"man": ("m", "common", "adult"), "woman": ("f", "common", "adult"), "king": ("m", "royal", "adult"), "queen": ("f", "royal", "adult"),
              "boy": ("m", "common", "young"), "girl": ("f", "common", "young"), "prince": ("m", "royal", "young"), "princess": ("f", "royal", "young")}
    ctx = {"m": ["he", "his", "him"], "f": ["she", "her", "hers"], "common": ["street", "market", "village"], "royal": ["crown", "throne", "palace"],
           "adult": ["work", "tax", "old"], "young": ["school", "toy", "play"]}
    places = [("france", "paris", "french"), ("spain", "madrid", "spanish"), ("italy", "rome", "italian"),
              ("germany", "berlin", "german"), ("japan", "tokyo", "japanese"), ("egypt", "cairo", "arabic")]
    pctx = {"country": ["nation", "border", "flag"], "city": ["mayor", "downtown", "metro"]}
    filler = ["the", "a", "of", "and", "is", "in", "to", "with"]
    sents = []
    for _ in range(n_sent):
        if rng.random() < 0.5:
            w = list(people)[rng.integers(8)]
            words = [w] + [str(rng.choice(ctx[a])) for a in people[w]]
        else:
            c, city, lang = places[rng.integers(len(places))]
            w, kind = (c, "country") if rng.random() < 0.5 else (city, "city")
            words = [w, lang, str(rng.choice(pctx[kind])), str(rng.choice(pctx[kind]))]
        words += [str(x) for x in rng.choice(filler, 2)]
        rng.shuffle(words)
        sents.append(words)
    return sents


def make_pairs(sentences, vocab, window):
    out = []
    for s in sentences:
        ids = [vocab[w] for w in s]
        for i in range(len(ids)):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    out.append((ids[i], ids[j]))
    return np.array(out)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def sgns_batch(Win, Wout, c, o, neg):
    v, uo, un = Win[c], Wout[o], Wout[neg]
    so = _sig(np.sum(v * uo, 1))
    sn = _sig(np.einsum("bd,bkd->bk", v, un))
    loss = float(-np.log(so).sum() - np.log(1 - sn).sum())
    gWin, gWout = np.zeros_like(Win), np.zeros_like(Wout)
    np.add.at(gWin, c, (so - 1)[:, None] * uo + np.einsum("bk,bkd->bd", sn, un))
    np.add.at(gWout, o, (so - 1)[:, None] * v)
    np.add.at(gWout, neg, sn[..., None] * v[:, None])
    return loss, gWin, gWout


def train_sgns(pairs, V, d=20, k=5, epochs=5, lr=0.05, batch=256, seed=0, counts=None):
    rng = np.random.default_rng(seed)
    if counts is None:
        counts = np.bincount(pairs[:, 0], minlength=V)
    p = counts.astype(float) ** 0.75
    p /= p.sum()
    Win = rng.normal(0, 0.1, (V, d))
    Wout = np.zeros((V, d))
    for ep in range(epochs):
        a = lr * (1 - ep / epochs)
        perm = rng.permutation(len(pairs))
        for b in range(0, len(pairs), batch):
            c, o = pairs[perm[b:b + batch]].T
            neg = rng.choice(V, (len(c), k), p=p)
            _, gi, go = sgns_batch(Win, Wout, c, o, neg)
            Win -= a * gi
            Wout -= a * go
    return Win


def analogy(E, vocab, a, b, c):
    inv = sorted(vocab, key=vocab.get)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    s = En @ (En[vocab[b]] - En[vocab[a]] + En[vocab[c]])
    s[[vocab[a], vocab[b], vocab[c]]] = -np.inf
    return inv[int(np.argmax(s))]


def most_similar(E, vocab, w, n=3):
    inv = np.array(sorted(vocab, key=vocab.get))
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    s = En @ En[vocab[w]]
    s[vocab[w]] = -np.inf
    return inv[np.argsort(-s)[:n]].tolist()
