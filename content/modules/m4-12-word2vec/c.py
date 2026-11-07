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
    ...


def sgns_batch(Win, Wout, c, o, neg):
    ...


def train_sgns(pairs, V, d=20, k=5, epochs=5, lr=0.05, batch=256, seed=0, counts=None):
    ...


def analogy(E, vocab, a, b, c):
    ...


def most_similar(E, vocab, w, n=3):
    ...
