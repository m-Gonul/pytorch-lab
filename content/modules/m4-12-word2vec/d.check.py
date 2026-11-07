import time
import numpy as np
import torch
from labcheck import close, need, ensure


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
    En = E / E.norm(dim=1, keepdim=True)
    ok = 0
    for a, b, c, t in TESTS:
        s = En @ (En[vocab[b]] - En[vocab[a]] + En[vocab[c]])
        s[[vocab[a], vocab[b], vocab[c]]] = -1e9
        ok += int(s.argmax()) == vocab[t]
    return ok


def check(ns, R_):
    def t_eq():
        torch.manual_seed(0)
        emb = torch.nn.Embedding(7, 3)
        lin = need(ns, "embedding_as_linear")(emb)
        idx = torch.tensor([2, 5, 2])
        oh = torch.nn.functional.one_hot(idx, 7).float()
        close(lin(oh).detach().numpy(), emb(idx).detach().numpy(), atol=1e-6, what="one-hot × W = satır seçimi")
        (emb(idx) ** 2).sum().backward()
        (lin(oh) ** 2).sum().backward()
        close(lin.weight.grad.T.numpy(), emb.weight.grad.numpy(), atol=1e-6, what="gradyanlar aynı (yalnız seçilen satırlar)")
        m = need(ns, "SGNS")(7, 3, True)
        m(torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([[5, 6], [0, 6]])).backward()
        g = m.inp.weight.grad
        ensure(g.is_sparse and set(g.coalesce().indices()[0].tolist()) == {1, 2}, "sparse=True: yalnız dokunulan satırlar")
    R_.test("embedding = one-hot × W; seyrek gradyan", t_eq)

    def t_train():
        S = _corpus()
        vocab = {w: i for i, w in enumerate(sorted({w for s in S for w in s}))}
        V = len(vocab)
        pad = V
        sk, cb = [], []
        for s in S:
            ids = [vocab[w] for w in s]
            for i in range(len(ids)):
                ctx = [ids[j] for j in range(max(0, i - 3), min(len(ids), i + 4)) if j != i]
                sk += [(ids[i], c) for c in ctx]
                cb.append((ctx + [pad] * (6 - len(ctx)), ids[i]))
        sk = torch.tensor(sk)
        cbc = torch.tensor([c for c, _ in cb])
        cbo = torch.tensor([o for _, o in cb])
        cnt = torch.bincount(sk[:, 0], minlength=V).float() ** 0.75
        g = torch.Generator().manual_seed(0)

        def sk_batches():
            perm = torch.randperm(len(sk), generator=g)
            for b in range(0, len(sk), 256):
                c, o = sk[perm[b:b + 256]].T
                yield c, o, torch.multinomial(cnt, len(c) * 5, True, generator=g).view(-1, 5)

        def cb_batches():
            perm = torch.randperm(len(cbo), generator=g)
            for b in range(0, len(cbo), 64):
                i = perm[b:b + 64]
                yield cbc[i], cbo[i], torch.multinomial(cnt, len(i) * 5, True, generator=g).view(-1, 5)
        rows = []
        for name, mk, bt, sp in (("skip-gram, Adam", lambda: ns["SGNS"](V, 20), sk_batches, False), ("skip-gram, seyrek + SparseAdam", lambda: ns["SGNS"](V, 20, True), sk_batches, True),
                                 ("CBOW, Adam", lambda: need(ns, "CBOW")(V, 20, pad), cb_batches, False)):
            torch.manual_seed(0)
            m = mk()
            t0 = time.perf_counter()
            need(ns, "train")(m, bt, 3, 0.01, sp)
            acc = _acc(m.inp.weight.detach()[:V], vocab)
            rows.append([name, f"{acc}/8", f"{time.perf_counter() - t0:.1f} sn"])
            ensure(acc >= 6, f"{name}: analojiler {acc}/8")
        R_.table("Aynı derlem, d = 20, 3 epoch", ["model", "analoji", "süre"], rows,
                 note="CBOW cümle başına tek örnek üretir (bağlam ortalaması → merkez): daha hızlı, sık kelimelerde iyi; skip-gram her bağlam kelimesi için ayrı örnek üretir: nadir kelimelerde daha iyi (Mikolov). Seyrek gradyan, büyük sözlüklerde her adımda yalnız batch'teki satırları günceller — 10⁶ satırlık tabloda fark büyüktür.")
    R_.test("SGNS ve CBOW", t_train)
