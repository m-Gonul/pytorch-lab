import numpy as np
from labcheck import need, ensure, max_loops


def check(ns, R_):
    src = ns["__source__"]
    R_.test("topk döngüsüz", lambda: max_loops(src, 0, func="topk_features"))

    def t():
        rng = np.random.default_rng(5)
        X = rng.normal(size=(40, 30))
        y = (X[:, 7] > 0).astype(int)
        ensure(7 in need(ns, "topk_features")(X, y, 3), "bilgili öznitelik seçilmeli")
        rows, w_all, r_all = [], [], []
        for s in range(5):
            w, r = need(ns, "cv_wrong_right")(s)
            w_all.append(w)
            r_all.append(r)
            rows.append([s, f"{w:.2f}", f"{r:.2f}"])
        ensure(np.mean(w_all) > 0.85 and abs(np.mean(r_all) - 0.5) < 0.15, "yanlış yol ~%100, doğru yol ~%50")
        R_.table("Saf gürültü (n = 50, p = 5000): 5 katlı CV doğruluğu", ["tohum", "seçim CV dışında (yanlış)", "seçim her katta (doğru)"], rows,
                 note="Etiketlerle ilişkili görünen 100 öznitelik şans eseri seçilir — ama doğrulama örneklerinin etiketleri de bu seçime katkı yapmıştır. Hastie, Tibshirani & Friedman (ESL §7.10.2) 'çapraz doğrulamanın yanlış ve doğru yolu'. Kural: etiket gören her adım (seçim, ölçekleme, hiperparametre) CV döngüsünün içinde.")
    R_.test("öznitelik seçimi sızıntısı", t)
