import numpy as np
import cv2
from labcheck import need, ensure


def check(ns, R):
    def t():
        Xtr, ytr = need(ns, "shapes")(400, 0)
        Xte, yte = ns["shapes"](200, 1)
        ensure(Xtr.shape == (400, 64, 64) and set(np.unique(ytr)) == {0, 1}, "veri")
        F = need(ns, "hog_features")(Xtr[:3])
        ensure(F.shape == (3, 1764), f"64 × 64 için 7 × 7 × 4 × 9 = 1764 boyut, {F.shape}")
        h, r = need(ns, "train_eval")(Xtr, ytr, Xte, yte)
        ensure(h > 0.9 and h > r + 0.15, f"HOG {h:.2f}, ham {r:.2f}")
        R.table("Daire mi kare mi? (rastgele konum, boyut, parlaklık)", ["öznitelik", "boyut", "test doğruluğu"], [["ham pikseller", 4096, f"{r:.2f}"], ["HOG", 1764, f"{h:.2f}"]],
                note="Yaya dedektörünün 64 × 128 penceresi aynı ayarlarla 3780 boyut verir (OpenCV HOGDescriptor varsayılanı).")
    R.test("HOG + doğrusal SVM", t)
