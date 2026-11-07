import math
import random
from labcheck import need, ensure


def check(ns, R_):
    def t():
        ensure(need(ns, "char_ngrams")("where", 3, 3) == ["<wh", "whe", "her", "ere", "re>", "<where>"], "fastText örneği (Bojanowski ve ark.)")
        ensure("<her>" in ns["char_ngrams"]("her", 3, 3) and ns["char_ngrams"]("her", 3, 6).count("<her>") == 1, "kısa kelime: tam dizge tek kez")
        rnd = random.Random(0)
        B = 5000
        Z = [[rnd.gauss(0, 1) for _ in range(64)] for _ in range(B)]
        cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
        wv = lambda w: need(ns, "word_vector")(w, Z, B)
        pairs = [("walking", "walked"), ("walking", "talking"), ("unhappiness", "happiness"), ("walking", "table"), ("tokyo", "kyoto")]
        rows = [[a, b, f"{cos(wv(a), wv(b)):.2f}", len(set(ns['char_ngrams'](a)) & set(ns['char_ngrams'](b)))] for a, b in pairs]
        ensure(float(rows[2][2]) > float(rows[3][2]) + 0.4 and float(rows[0][2]) > float(rows[3][2]), "ortak n-gramlar → benzer vektör")
        R_.table("Rastgele (eğitilmemiş) n-gram vektörleriyle OOV kelime vektörleri", ["kelime 1", "kelime 2", "kosinüs", "ortak n-gram"], rows,
                 note="Eğitimden önce bile, ortak alt dizgileri olan kelimeler ortak vektörleri topladığı için benzerdir; eğitim bu paylaşımı anlamla hizalar. Hiç görülmemiş bir kelimenin ('unhappiness' derlemde yoksa) vektörü n-gramlarından kurulur — OOV kovalarının 'tüm bilinmeyenler aynı' sorununa çözüm. Türkçe gibi eklemeli dillerde özellikle değerli.")
    R_.test("alt-kelime vektörleri", t)
