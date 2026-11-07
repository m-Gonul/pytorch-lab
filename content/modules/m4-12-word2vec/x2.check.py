import math
import random
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        codes, paths = need(ns, "huffman")([5, 1, 1, 2])
        ensure(len(codes[0]) == 1 and len(codes[1]) == 3 and len(codes[2]) == 3 and len(codes[3]) == 2, "kod uzunlukları (5, 1, 1, 2 → 1, 3, 3, 2)")
        ensure(all(paths[w][0] == paths[0][0] for w in range(4)), "tüm yollar kökten başlar")
        cs = ["".join(map(str, codes[w])) for w in range(4)]
        ensure(not any(a != b and b.startswith(a) for a in cs for b in cs), "önek-özgür kod")
        rnd = random.Random(0)
        for V in (7, 30):
            counts = [int(1000 / (r + 1)) + 1 for r in range(V)]
            codes, paths = ns["huffman"](counts)
            Wn = [[rnd.gauss(0, 1) for _ in range(5)] for _ in range(V - 1)]
            v = [rnd.gauss(0, 1) for _ in range(5)]
            close(sum(need(ns, "hs_prob")(v, w, codes, paths, Wn) for w in range(V)), 1.0, atol=1e-9, what=f"V = {V}: olasılıklar toplamı 1")
        rows = []
        for V in (10, 100, 1000, 10000):
            counts = [int(1e6 / (r + 1)) + 1 for r in range(V)]
            codes, _ = ns["huffman"](counts)
            tot = sum(counts)
            avg = sum(c * len(codes[w]) for w, c in enumerate(counts)) / tot
            H = -sum(c / tot * math.log2(c / tot) for c in counts)
            ensure(H <= avg < H + 1, "Huffman: H ≤ ort. uzunluk < H + 1")
            rows.append([V, f"{math.log2(V):.1f}", f"{avg:.2f}", f"{H:.2f}", max(len(c) for c in codes.values())])
        R_.table("Zipf dağılımında Huffman ağacı", ["V", "log₂ V", "ortalama yol (sıklık ağırlıklı)", "entropi", "en uzun yol"], rows,
                 note="Hiyerarşik softmax p(w | v)'yi kökten yaprağa ikili kararların çarpımı olarak yazar: V − 1 iç düğüm vektörü, örnek başına yol uzunluğu kadar iç çarpım. Huffman ağacı sık kelimelere kısa yol verir: ortalama maliyet entropiye yakın, log₂ V'den de küçük.")
    R_.test("hiyerarşik softmax", t)
