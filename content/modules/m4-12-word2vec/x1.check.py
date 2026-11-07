import numpy as np
from labcheck import close, max_loops, need, ensure


def _corpus(n_sent=6000, seed=0):
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


TESTS = [("man", "king", "woman", "queen"), ("man", "woman", "boy", "girl"), ("king", "queen", "prince", "princess"), ("boy", "prince", "girl", "princess"),
         ("france", "paris", "spain", "madrid"), ("italy", "rome", "japan", "tokyo"), ("man", "boy", "queen", "princess"), ("germany", "berlin", "egypt", "cairo")]


def _acc(E, vocab):
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    ok = 0
    for a, b, c, t in TESTS:
        s = En @ (En[vocab[b]] - En[vocab[a]] + En[vocab[c]])
        s[[vocab[a], vocab[b], vocab[c]]] = -np.inf
        ok += int(np.argmax(s)) == vocab[t]
    return ok


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("cooccurrence", "ppmi", "svd_embeddings")])

    def t():
        C = need(ns, "cooccurrence")(np.array([[0, 1], [0, 1], [1, 2], [2, 0]]), 3)
        ensure(C[0, 1] == 2 and C[1, 2] == 1 and C.sum() == 4, "sayım")
        C2 = np.array([[0., 4, 1], [4, 0, 1], [1, 1, 0]])
        P = need(ns, "ppmi")(C2)
        close(P[0, 1], max(0, np.log(4 * 12 / (5 * 5))), what="PPMI")
        ensure(P[0, 0] == 0 and np.isfinite(P).all(), "sıfır sayım → 0")
        close(ns["ppmi"](C2, 2.0)[0, 1], max(0, np.log(4 * 12 / 25) - np.log(2)), what="SPPMI (kaydırma log k)")
        S = _corpus()
        vocab = {w: i for i, w in enumerate(sorted({w for s in S for w in s}))}
        V = len(vocab)
        pairs = np.array([(vocab[s[i]], vocab[s[j]]) for s in S for i in range(len(s)) for j in range(max(0, i - 3), min(len(s), i + 4)) if j != i])
        Cc = ns["cooccurrence"](pairs, V)
        rows = []
        accs = {}
        for name, M in (("ham sayım", Cc), ("log(1 + sayım)", np.log1p(Cc)), ("PPMI", ns["ppmi"](Cc))):
            for d in (5, 8, 20):
                E = need(ns, "svd_embeddings")(M, d)
                accs[(name, d)] = _acc(E, vocab)
                rows.append([name, d, f"{accs[(name, d)]}/8"])
        ensure(accs[("PPMI", 8)] >= 6 and accs[("PPMI", 8)] > accs[("ham sayım", 8)], "düşük boyutta PPMI-SVD ham sayımdan iyi")
        R_.table("Sayım tabanlı gömmeler: matris → SVD (eğitim yok)", ["matris", "d", "analoji"], rows,
                 note="Ham sayımlarda sık işlev kelimeleri (the, of…) ilk tekil vektörleri ele geçirir: az boyutla (d = 5, 8) anlam eksenlerine yer kalmaz; bu küçük derlemde d = 20 her şeyi kurtarır, gerçek sözlüklerde kurtarmaz. PMI 'beklenenden ne kadar sık birlikte' sorusunu sorar ve bu baskınlığı kaldırır. Levy & Goldberg: SGNS örtük olarak PMI − log k'yı çarpanlara ayırır — sayım tabanlı ve tahmin tabanlı yöntemler aynı nesnenin iki yüzüdür (GloVe ikisini birleştirir).")
    R_.test("PPMI + SVD", t)
