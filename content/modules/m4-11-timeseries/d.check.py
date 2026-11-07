import time
import torch
import torch.nn.functional as F
from labcheck import need, ensure


def check(ns, R_):
    S = need(ns, "generate_time_series")(5000, 48, 0)
    Xtr, Xte = S[:4000, :40], S[4000:, :40]
    y1tr, y1te = S[:4000, 40], S[4000:, 40]
    y8tr, y8te = S[:4000, 40:, 0], S[4000:, 40:, 0]

    def t_shapes():
        M = need(ns, "ex6_models")()
        shapes = {k: tuple(m(Xte[:3]).shape) for k, m in M.items()}
        ensure(shapes == {1: (3, 1), 2: (3, 1), 3: (3, 1), 4: (3, 8), 5: (3, 40, 8)}, f"çıkış şekilleri {shapes}")
        cnt = {k: sum(p.numel() for p in m.parameters()) for k, m in M.items()}
        ensure(cnt[1] == 41 and cnt[3] == 834 + 33 and cnt[4] == 952 + 32, "PyTorch sayımı (iki bias)")
        T = need(ns, "seq_targets")(S[:2])
        ensure(tuple(T.shape) == (2, 40, 8) and torch.equal(T[:, 39], S[:2, 40:, 0]) and torch.equal(T[:, 0, 0], S[:2, 1, 0]), "dizi→dizi hedefleri")
    R_.test("modeller", t_shapes)

    def t_train():
        t00 = time.perf_counter()
        torch.manual_seed(0)
        M = ns["ex6_models"]()
        tr = need(ns, "train")
        res = {"naif (1)": F.mse_loss(Xte[:, -1], y1te).item(), "naif (8)": F.mse_loss(Xte[:, -1].expand(-1, 8), y8te).item()}
        tr(M[1], Xtr, y1tr, 10, 5e-3)
        tr(M[2], Xtr, y1tr, 10, 5e-3)
        tr(M[3], Xtr, y1tr, 10, 1e-3)
        lin8 = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(40, 8))
        tr(lin8, Xtr, y8tr, 10, 5e-3)
        tr(M[4], Xtr, y8tr, 10, 5e-3)
        tr(M[5], Xtr, ns["seq_targets"](S[:4000]), 10, 5e-3)
        with torch.no_grad():
            for k in (1, 2, 3):
                res[k] = F.mse_loss(M[k](Xte), y1te).item()
            res["it"] = F.mse_loss(need(ns, "iterative_forecast")(M[3], Xte, 8), y8te).item()
            res["lin8"] = F.mse_loss(lin8(Xte), y8te).item()
            res[4] = F.mse_loss(M[4](Xte), y8te).item()
            res[5] = F.mse_loss(M[5](Xte)[:, -1], y8te).item()
        rows = [["naif: son değer", "1", f"{res['naif (1)']:.4f}"], ["model 1: doğrusal (41 parametre)", "1", f"{res[1]:.4f}"],
                ["model 2: tek rekürent nöron", "1", f"{res[2]:.4f}"], ["model 3: RNN 16-16-1", "1", f"{res[3]:.4f}"],
                ["naif: son değeri 8 kez", "8", f"{res['naif (8)']:.4f}"], ["model 3 yinelemeli", "8", f"{res['it']:.4f}"],
                ["doğrusal 40 → 8", "8", f"{res['lin8']:.4f}"], ["model 4: dizi→vektör", "8", f"{res[4]:.4f}"], ["model 5: dizi→dizi (son adım)", "8", f"{res[5]:.4f}"]]
        ensure(res[1] < res["naif (1)"] / 3 and res[3] < res["naif (1)"] / 3, "doğrusal ve derin RNN naif tahmini açık farkla geçer")
        ensure(res[2] > res[1], "tek nöronlu RNN doğrusal modeli geçemez")
        ensure(res[4] < res["lin8"] and res[5] < res["lin8"] and res["it"] > res[5], "8 adım: RNN'ler > doğrusal; yinelemeli < dizi→dizi")
        R_.table(f"Ders alıştırması 6 (4000 seri, 10 epoch, {time.perf_counter() - t00:.0f} sn) — test MSE", ["model", "ufuk", "MSE"], rows,
                 note="Tek nöronun gizli durumu tek bir tanh'lı sayıdır: hem bellek hem çıkış aralığı (−1, 1) kısıtlı. Yinelemeli tahminde hatalar birikir; 8 değeri doğrudan tahmin etmek daha iyidir. Dizi→dizi her adımda kayıp aldığı için aynı epoch'ta en iyi sonuca yaklaşır (Géron: 0.008 → 0.006).")
    R_.test("eğitim ve karşılaştırma", t_train)
