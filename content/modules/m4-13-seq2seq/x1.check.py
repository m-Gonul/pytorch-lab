import math
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        refs = ["the cat is on the mat".split(), "there is a cat on the mat".split()]
        ensure(need(ns, "modified_precision")("the the the the the the the".split(), refs, 1) == (2, 7), "Papineni örneği: 2/7")
        ensure(ns["modified_precision"]("the cat the cat on the mat".split(), refs, 2) == (4, 6), "bigram kırpma")
        close(need(ns, "brevity_penalty")(4, 6), math.exp(1 - 6 / 4), what="kısalık cezası")
        ensure(ns["brevity_penalty"](7, 6) == 1.0, "uzun aday cezasız")
        close(need(ns, "bleu")("the cat is on the mat".split(), refs), 1.0, what="birebir eşleşme")
        cands = {"birebir": "the cat is on the mat", "kelime sırası bozuk": "mat the on is cat the", "kısa": "the cat", "eşanlamlı": "the cat sits on the mat", "the × 7": "the the the the the the the"}
        rows = []
        for k, c in cands.items():
            cc = c.split()
            ps = [ns["modified_precision"](cc, refs, n) for n in range(1, 5)]
            rows.append([k, " / ".join(f"{a}/{b}" for a, b in ps), f"{ns['bleu'](cc, refs, 2):.3f}", f"{ns['bleu'](cc, refs):.3f}"])
        ensure(ns["bleu"]("the cat".split(), refs, 2) < 0.4, "kısa aday cezalandırılır")
        R_.table("BLEU (iki referans)", ["aday", "p₁ … p₄", "BLEU-2", "BLEU-4"], rows,
                 note="Kırpma 'the the the…' hilesini, kısalık cezası ('the cat' — yüksek kesinlik, düşük kapsama) kısa çıktıları engeller. BLEU eşanlamlıyı ödüllendirmez ve cümle düzeyinde gürültülüdür (tek bir 4-gram eksik → 0): derlem düzeyinde sayımlar toplanarak hesaplanır.")
    R_.test("BLEU", t)
