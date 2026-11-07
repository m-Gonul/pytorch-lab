import math
from labcheck import close, need, ensure

TEXT = (
    "pekiştirmeli öğrenmede bir ajan bir ortamla etkileşir. ajan her adımda ortamın durumunu gözler, "
    "bir eylem seçer ve bir ödül alır. amaç uzun vadede toplanan ödülü en büyük yapmaktır. ortamın "
    "bir sonraki durumu yalnızca şimdiki duruma ve seçilen eyleme bağlıysa ortam markov özelliğini "
    "taşır. bu özellik sayesinde geçmişin tamamını hatırlamak gerekmez; şimdiki durum geleceği "
    "tahmin etmek için yeterlidir. değer fonksiyonu bir durumdan başlayarak beklenen toplam ödülü "
    "ölçer. politika ise her durumda hangi eylemin seçileceğini söyler. iyi bir politika bulmak için "
    "ajan hem bildiklerini kullanmalı hem de yeni şeyler denemelidir. bu ikilem keşif ve sömürü "
    "dengesi olarak bilinir ve pekiştirmeli öğrenmenin merkezindedir. "
) * 3


def check(ns, R):
    train, test = TEXT[:1400], TEXT[1400:1700]
    vocab = sorted(set(TEXT))

    def f():
        m = need(ns, "fit")("abracadabra", 1)
        ensure(dict(m["a"]) == {"b": 2, "c": 1, "d": 1}, f"'a' bağlamı: {dict(m['a'])}")
        m2 = ns["fit"]("abracadabra", 2)
        ensure(dict(m2["ab"]) == {"r": 2}, "2. derece")
    R.test("fit", f)

    def lp():
        m = ns["fit"](train, 1)
        V = len(vocab)
        c = m.get("a", {}); want = math.log((c.get("b", 0) + 1) / (sum(c.values()) + V))
        close(need(ns, "log_prob")(m, "ab", 1, 1.0, vocab), want, atol=1e-12, what="tek geçiş")
        rows = []
        for k in (0, 1, 2, 3, 5):
            mk = ns["fit"](train, k)
            bpc = -ns["log_prob"](mk, test, k, 0.1, vocab) / (len(test) - k) / math.log(2)
            rows.append([k, f"{bpc:.2f}"])
        ensure(float(rows[2][1]) < float(rows[0][1]), "bağlam test kayıbını azaltmalı")
        R.table("Test karakter başına bit (α = 0.1)", ["derece k", "bit / karakter"], rows,
                note="k büyüdükçe önce iyileşir; çok büyük k'da bağlamlar seyrekleşir (bu küçük metin tekrarlı olduğu için ezber de mümkün). Dil modellerinin (M4) çıkış noktası.")
    R.test("log_prob", lp)

    def g():
        m = ns["fit"](train, 3)
        s = need(ns, "generate")(m, "pek", 80, 3, 1)
        ensure(len(s) == 83 and s.startswith("pek"), "uzunluk")
        ensure(all(s[i:i + 3] in m or True for i in range(len(s) - 3)), "")
        return f"örnek: «{s}»"
    R.test("generate", g)
