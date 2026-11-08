import random
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        rng = random.Random(0)
        xs = [need(ns, "poisson")(3.0, rng) for _ in range(20000)]
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        close((m, v), (3.0, 3.0), atol=0.1, what="Poisson ortalama = varyans = λ")
        ensure(sum(x == 0 for x in xs) / len(xs) > 0.03, "sıfır uzunluk mümkün")
        toks = [f"w{i}" for i in range(400)]
        out = need(ns, "text_infill")(toks, 0.3, 3.0, random.Random(1))
        kept = [t for t in out if t != "<mask>"]
        nm = out.count("<mask>")
        ensure(kept == [t for t in toks if t in set(kept)], "kalan token'lar sırayı korur")
        dropped = 400 - len(kept)
        ensure(abs(dropped - 120) <= 3 and nm < dropped, f"%30 maskelendi ({dropped}), aralık başına tek mask ({nm})")
        d = need(ns, "token_deletion")(toks, 0.2, random.Random(2))
        ensure(abs(len(d) - 320) < 30 and all(t in toks for t in d), "silme")
        s = need(ns, "sentence_permutation")([["a"], ["b"], ["c"], ["d"]], random.Random(3))
        ensure(sorted(map(tuple, s)) == [("a",), ("b",), ("c",), ("d",)], "cümle karıştırma")
        r = need(ns, "document_rotation")(toks, random.Random(4))
        k = r.index("w0")
        ensure(r[k:] + r[:k] == toks, "döndürme")
        R_.note(f"Örnek: {' '.join(out[:18])} … — {dropped} token {nm} mask ile değiştirildi. BART metin doldurma (span'ler Poisson(λ = 3), %30) + cümle karıştırmayı en iyi bulur: kod çözücü yalnız eksik içeriği değil, eksik uzunluğu da tahmin etmek zorundadır (SpanBERT'ten farkı). Kod çözücü tüm orijinal metni üretir.")
    R_.test("BART gürültüleri", t)
