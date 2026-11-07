from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        close(need(ns, "flops_factor")(1.2, 1.1, 1.15, 1), 1.2 * 1.21 * 1.3225, what="B0 katsayıları ≈ 1.92")
        cands = need(ns, "grid_candidates")(0.05)
        ensure((1.2, 1.1, 1.15) in cands and all(abs(a * b * b * g * g - 2) <= 0.1 for a, b, g in cands), "kısıtı sağlayanlar")
        rows = []
        for phi in range(0, 8):
            d, w, r = need(ns, "scale_model")((18, 32, 224), 1.2, 1.1, 1.15, phi)
            rows.append([f"B{phi}", d, w, r, f"{ns['flops_factor'](1.2, 1.1, 1.15, phi):.1f}×"])
        R_.table("EfficientNet-B0'dan bileşik ölçekleme (α, β, γ) = (1.2, 1.1, 1.15)", ["model", "katman", "kök genişliği", "çözünürlük", "FLOP"], rows,
                 note="Yalnız derinliği ya da yalnız genişliği büyütmek çabuk doyar (makale Şekil 3); üçünü dengeli büyütmek daha büyük girdinin daha büyük receptive field'a ve daha çok kanala ihtiyaç duyduğu sezgisini kodlar. (Gerçek B1–B7 çözünürlükleri biraz farklı yuvarlanmıştır.)")
    R_.test("bileşik ölçekleme", t)
