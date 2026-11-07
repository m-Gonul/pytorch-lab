import numpy as np
from labcheck import close, max_loops, need, ensure

TESTS = [("man", "king", "woman", "queen"), ("man", "woman", "boy", "girl"), ("king", "queen", "prince", "princess"), ("boy", "prince", "girl", "princess"),
         ("france", "paris", "spain", "madrid"), ("italy", "rome", "japan", "tokyo"), ("man", "boy", "queen", "princess"), ("germany", "berlin", "egypt", "cairo")]


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("sgns_batch", "analogy", "most_similar")])
    S = ns["toy_corpus"](6000, 0)
    vocab = {w: i for i, w in enumerate(sorted({w for s in S for w in s}))}
    V = len(vocab)

    def t_grad():
        rng = np.random.default_rng(0)
        Win, Wout = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
        c, o, neg = np.array([0, 2, 0]), np.array([1, 3, 1]), np.array([[4, 5], [1, 4], [5, 5]])
        L, gi, go = need(ns, "sgns_batch")(Win, Wout, c, o, neg)
        h = 1e-6
        for (M, G) in ((Win, gi), (Wout, go)):
            for idx in [(0, 1), (2, 0), (5, 2), (1, 1), (3, 2)]:
                M[idx] += h
                Lp = ns["sgns_batch"](Win, Wout, c, o, neg)[0]
                M[idx] -= h
                close((Lp - L) / h, G[idx], atol=1e-4, what=f"sayısal gradyan {idx}")
        ensure(np.all(gi[[1, 3, 4, 5]] == 0), "Win'de yalnız merkez satırlar")
    R_.test("toplu SGNS gradyanı", t_grad)

    def t_train():
        P = need(ns, "make_pairs")(S, vocab, 3)
        ensure(P.shape[1] == 2 and len(P) == sum(sum(min(len(s), i + 4) - max(0, i - 3) - 1 for i in range(len(s))) for s in S), "çift sayısı")
        E = need(ns, "train_sgns")(P, V, 20, 5, 5, 0.05, 256, 0)
        res = [(a, b, c, t, need(ns, "analogy")(E, vocab, a, b, c)) for a, b, c, t in TESTS]
        acc = sum(r[3] == r[4] for r in res)
        ensure(acc >= 7, f"analoji doğruluğu {acc}/8")
        ensure("queen" in need(ns, "most_similar")(E, vocab, "princess", 3) or "girl" in ns["most_similar"](E, vocab, "princess", 3), "komşular")
        R_.table(f"{len(P)} çift, d = 20, 5 epoch — analojiler (b − a + c)", ["a : b", "c : ?", "beklenen", "bulunan"],
                 [[f"{a} : {b}", c, t, f] for a, b, c, t, f in res],
                 note="Derlemde kral−erkek ilişkisini gösteren hiçbir cümle yok: her kelime yalnız özelliklerine (cinsiyet, statü, yaş / ülke–şehir, dil) özgü bağlam kelimeleriyle geçiyor. Skip-gram bu bağlamları tahmin etmeye çalışırken vektörleri özelliklerin toplamı gibi düzenler — analojiler bu toplamsallığın sonucu.")
    R_.test("eğitim ve analojiler", t_train)
