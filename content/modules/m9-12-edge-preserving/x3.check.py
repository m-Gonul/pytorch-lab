import numpy as np
import cv2
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    clean = data.camera()[128:384, 128:384]

    def mn():
        m = need(ns, "method_noise")(clean, lambda x: cv2.GaussianBlur(x, (5, 5), 0))
        close(m, clean.astype(float) - cv2.GaussianBlur(clean, (5, 5), 0), what="Gauss yöntem gürültüsü")
        close(ns["method_noise"](clean, lambda x: x), np.zeros(clean.shape), what="özdeşlik süzgeci")
    R.test("method_noise", mn)

    def cmp():
        t = need(ns, "compare")(clean)
        ensure(set(t) == {"gauss5", "median5", "bilateral", "nlm"}, "dört süzgeç")
        ensure(t["nlm"][1] < t["gauss5"][1] and t["bilateral"][1] < t["gauss5"][1], "NLM ve bilateral temiz görüntüye Gauss'tan az dokunmalı")
        R.table("camera 256², σ = 15", ["süzgeç", "gürültülüde PSNR", "yöntem gürültüsü std (temiz girdi)"], [[k, f"{v[0]:.2f}", f"{v[1]:.2f}"] for k, v in t.items()],
                note="Yöntem gürültüsü görüntüsünde Gauss'unki tüm kenarları gösterir; NLM'ninki çoğunlukla yapısız beyaz gürültüye benzer.")
    R.test("karşılaştırma tablosu", cmp)
