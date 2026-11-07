import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("make_dataset/encode döngüsüz", lambda: (max_loops(src, 0, func="make_dataset"), max_loops(src, 0, func="encode")))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    V = "0123456789+=\n"

    def ds():
        D = need(ns, "make_dataset")(500, np.random.default_rng(3))
        ensure(D.shape == (500, 10), "şekil (n, 10)")
        for row in D[:50]:
            s = "".join(V[i] for i in row)
            a, b = int(s[:2]), int(s[3:5])
            ensure(s == f"{a:02d}+{b:02d}={str(a + b).zfill(3)[::-1]}\n", f"satır {s!r}")
    R.test("make_dataset = string biçimi", ds)

    def enc():
        text = "07+45=250\n99+99=891\n"
        ensure(np.array_equal(need(ns, "encode")(text), [[0, 7, 10, 4, 5, 11, 2, 5, 0, 12], [9, 9, 10, 9, 9, 11, 8, 9, 1, 12]]), "kodlama")
    R.test("encode", enc)

    def ent():
        D = ns["make_dataset"](50000, np.random.default_rng(0))
        H = need(ns, "answer_entropy")(D)
        a = D[:, 0] * 10 + D[:, 1]; b = D[:, 3] * 10 + D[:, 4]; s = a + b
        ref = 0.0
        for col in (s % 10, (s // 10) % 10, s // 100):
            p = np.bincount(col) / len(col); p = p[p > 0]
            ref -= (p * np.log(p)).sum()
        close(H, ref)
        return f"taban: {H:.3f} nat / satır (birler ve onlar ≈ log 10, yüzler ≈ 0.69). Model bunu ~0'a indirmeli."
    R.test("answer_entropy (taban kayıp)", ent)
