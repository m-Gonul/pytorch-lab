import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("interp_pos", "sincos_2d")])

    def t():
        rng = np.random.default_rng(0)
        pos = rng.normal(size=(14 * 14, 6))
        close(need(ns, "interp_pos")(pos, 14), pos, atol=1e-12, what="aynı boyut → değişmez")
        lin = np.add.outer(np.arange(14.), 2 * np.arange(14.))[..., None] * np.ones(3)
        up = ns["interp_pos"](lin.reshape(-1, 3), 24).reshape(24, 24, 3)
        tgt = (np.arange(24) + 0.5) * 14 / 24 - 0.5
        inner = (tgt >= 0) & (tgt <= 13)
        close(up[np.ix_(inner, inner)][..., 0], np.add.outer(tgt[inner], 2 * tgt[inner]), atol=1e-10, what="doğrusal alan tam interpole edilir")
        P = need(ns, "sincos_2d")(14, 64)
        ensure(P.shape == (196, 64), "şekil")
        S = P @ P.T
        a, b = S[14 * 5 + 5], S[14 * 5 + 6]
        ensure(np.argmax(a) == 14 * 5 + 5 and a[14 * 5 + 6] > a[14 * 9 + 9], "komşu konumlar daha benzer")
        close(S[14 * 2 + 3, 14 * 2 + 5], S[14 * 8 + 7, 14 * 8 + 9], atol=1e-10, what="benzerlik yalnız göreli yer değiştirmeye bağlı")
        R_.note("ViT'te konum embedding'leri öğrenilir ve 2B yapı yalnız ince ayarda interpolasyonla kullanılır; öğrenilmiş embedding'ler yine de 2B ızgarayı kendiliğinden kodlar (makale Şekil 7 orta: aynı satır/sütundaki yamalar benzer). Sabit 2B sin-cos (MAE, DeiT-III) bunu baştan verir ve göreli yer değiştirmeye göre değişmezdir.")
    R_.test("konum embedding'leri", t)
