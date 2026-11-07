import time
import torch
import torch.nn.functional as F
from labcheck import need, ensure


def check(ns, R_):
    X, Y = need(ns, "dates")(12000, 0)
    Xt, Yt = ns["dates"](1000, 1)
    x, y = need(ns, "encode")(X, Y)
    xt, yt = ns["encode"](Xt, Yt)

    def t():
        res, rows, models = {}, [], {}
        for attn in (False, True):
            torch.manual_seed(0)
            m = need(ns, "Seq2Seq")(64, attn)
            out = m(x[:4], y[:4])
            ensure(tuple(out.shape) == (4, 11, len(ns["OUT"]) + 3), "logit şekli (B, T, V)")
            opt = torch.optim.Adam(m.parameters(), 3e-3)
            t0 = time.perf_counter()
            for ep in range(2):
                perm = torch.randperm(len(x))
                for i in range(0, len(x), 64):
                    idx = perm[i:i + 64]
                    loss = F.cross_entropy(m(x[idx], y[idx]).flatten(0, 1), y[idx].flatten())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            m.eval()
            p = m.greedy(xt)
            res[attn] = (p == yt).all(1).float().mean().item()
            tokacc = (p == yt).float().mean().item()
            models[attn] = m
            rows.append(["Bahdanau attention'lı" if attn else "attention'sız (yalnız son durum)", sum(q.numel() for q in m.parameters()), f"{tokacc:.3f}", f"{res[attn]:.3f}", f"{time.perf_counter() - t0:.0f} sn"])
        ensure(res[True] > 0.85 and res[True] > res[False] + 0.3, f"attention farkı {res}")
        A = models[True].attention_map(xt[:200])
        ensure(tuple(A.shape) == (200, 11, xt.shape[1]), "attention haritası (B, T_out, T_in)")
        ensure(torch.allclose(A.sum(-1), torch.ones(200, 11), atol=1e-5) and float(A.masked_select(~(xt[:200] != 0)[:, None].expand_as(A)).abs().max()) == 0.0, "satırlar 1'e toplanır; dolguya ağırlık yok")
        L = (xt[:200] != 0).sum(1)
        yearmask = (torch.arange(xt.shape[1])[None] >= (L - 4)[:, None]) & (torch.arange(xt.shape[1])[None] < L[:, None])
        mass = (A[:, :4] * yearmask[:, None]).sum(-1).mean().item()
        ensure(mass > 0.5, f"yıl rakamları üretilirken attention girdideki yıla bakar ({mass:.2f})")
        R_.table("Tarih çevirisi (12 000 örnek, 2 epoch, açgözlü çözümleme)", ["model", "parametre", "token doğruluğu", "tam eşleşme", "süre"], rows,
                 note=f"Attention'sız modelde kod çözücü tüm girdiyi tek bir 64 boyutlu vektörden çıkarmak zorunda; attention'lı model her çıkış karakterinde ilgili girdi karakterlerine bakar. Yıl rakamları (çıkış 1–4) üretilirken attention kütlesinin ortalama {mass:.0%}'i girdinin son dört karakterinde (yıl) — hizalama hiç denetim olmadan öğrenildi.")
    R_.test("attention'sız vs Bahdanau", t)
