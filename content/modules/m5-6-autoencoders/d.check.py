import time
import torch
import torch.nn.functional as F
from labcheck import need, ensure


def check(ns, R_):
    X, y = need(ns, "digits")()
    g = torch.Generator().manual_seed(0)
    p = torch.randperm(len(X), generator=g)
    te, pool = p[:497], p[497:]
    lab = pool[:50]

    def t():
        t00 = time.perf_counter()
        rows, M = [], {}
        for name, kw, mask in (("bağsız AE", dict(tied=False), 0.0), ("bağlı AE", dict(tied=True), 0.0), ("gürültü giderici AE (maske 0.5)", dict(tied=False), 0.5)):
            torch.manual_seed(0)
            m = need(ns, "AE")((64, 32, 12), **kw)
            need(ns, "train_ae")(m, X[pool], 100, 3e-3, mask)
            M[name] = m
        torch.manual_seed(0)
        M["açgözlü katman katman"] = need(ns, "greedy_pretrain")((64, 32, 12), X[pool], 100)
        mk = (torch.rand(X[te].shape, generator=torch.Generator().manual_seed(5)) >= 0.5).float()
        with torch.no_grad():
            for name, m in M.items():
                rows.append([name, sum(q.numel() for q in m.parameters()), f"{F.mse_loss(m(X[te]), X[te]).item():.4f}", f"{F.mse_loss(m(X[te] * mk), X[te]).item():.4f}"])
        cnt = {r[0]: r[1] for r in rows}
        ensure(cnt["bağlı AE"] < 0.6 * cnt["bağsız AE"], "bağlı ağırlık ≈ yarı parametre")
        ensure(float(rows[2][3]) < 0.7 * float(rows[0][3]), "DAE maskeli girdiyi daha iyi tamamlar")
        ensure(float(rows[1][2]) < 1.3 * float(rows[0][2]), "bağlı AE benzer hata")
        R_.table("Rakamlar 64 → 32 → 12 → 32 → 64 (1300 eğitim, 100 epoch)", ["model", "parametre", "test MSE", "yarısı silinmiş girdiden MSE"], rows,
                 note="Bağlı ağırlıklar parametreyi yarıya indirir, hata neredeyse aynı kalır. Gürültü giderici AE temiz girdide biraz kötü olabilir ama pikselleri yarısı silinmiş girdiden rakamı yeniden kurar: eksik ayrıntıyı tahmin etmeyi öğrenmiştir.")
        acc = {}
        rows2 = []
        for name, enc, fr in (("sıfırdan", None, False), ("AE ön eğitimi + ince ayar", M["bağsız AE"], False), ("AE kodlayıcı donuk", M["bağsız AE"], True),
                              ("DAE kodlayıcı donuk", M["gürültü giderici AE (maske 0.5)"], True), ("DAE ön eğitimi + ince ayar", M["gürültü giderici AE (maske 0.5)"], False)):
            acc[name] = need(ns, "classify")(enc, X[lab], y[lab], X[te], y[te], fr, 300)
            rows2.append([name, f"{acc[name]:.3f}"])
        ensure(acc["AE ön eğitimi + ince ayar"] > acc["sıfırdan"] + 0.02 and acc["DAE kodlayıcı donuk"] > acc["AE kodlayıcı donuk"], f"ön eğitim {acc}")
        R_.table(f"50 etiketle sınıflandırma ({time.perf_counter() - t00:.0f} sn toplam)", ["kodlayıcı", "test doğruluğu"], rows2,
                 note="Etiketsiz 1300 örnekten öğrenilen kodlayıcı az etiketle başlangıç noktası olarak işe yarar. Donuk kodlarda DAE'nin temsili düz AE'ninkinden belirgin iyi: gürültü giderme kriteri daha 'anlamlı' öznitelikler öğretir (Vincent ve ark.).")
    R_.test("AE ailesi ve ön eğitim", t)
