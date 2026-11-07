import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        res = [need(ns, "experiment")(150, s) for s in range(3)]
        a = np.mean([r[0] for r in res]); b = np.mean([r[1] for r in res])
        ensure(b < 0.85 * a, f"artırım test hatasını düşürmeli ({a:.3f} → {b:.3f})")
        R_.table("150 eğitim durumu, ~4370 test durumu (minimax değerleri), 3 tohum", ["veri", "test MSE"], [["yalnız 150 durum", f"{a:.3f}"], ["8 simetriyle 1200 örnek", f"{b:.3f}"]],
                 note="Değerin simetrilere değişmez olduğu bilgisi veriye gömülür. AlphaGo Zero artırımı eğitimde, AlphaGo ise ayrıca değerlendirmede (rastgele simetri) kullandı; AlphaZero satranç/shogi için kaldırdı (simetrik değiller).")
    R_.test("simetri artırımı", t)
