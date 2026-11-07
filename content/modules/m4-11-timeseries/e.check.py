import time
import torch
import torch.nn.functional as F
from labcheck import need, ensure, fmt_time


def _series(n, steps, seed):
    g = torch.Generator().manual_seed(seed)
    f1, f2, o1, o2 = torch.rand(4, n, 1, generator=g)
    t = torch.linspace(0, 1, steps)
    s = 0.5 * torch.sin((t - o1) * (f1 * 10 + 10)) + 0.2 * torch.sin((t - o2) * (f2 * 20 + 20))
    return (s + 0.1 * (torch.rand(n, steps, generator=g) - 0.5))[..., None]


def check(ns, R_):
    def t_causal():
        torch.manual_seed(0)
        c = need(ns, "CausalConv1d")(1, 1, 3, 4)
        x = torch.randn(1, 1, 30)
        ensure(c(x).shape[-1] == 30, "uzunluk korunur")
        x2 = x.clone()
        x2[..., 20:] += 5
        ensure(torch.allclose(c(x)[..., :20], c(x2)[..., :20]), "gelecek geçmişi etkilemez")
        m = need(ns, "WaveNet")(16, (1, 2, 4, 8) * 2, 8)
        ensure(tuple(m(torch.randn(2, 50, 1)).shape) == (2, 50, 8), "çıkış (N, T, 8)")
        xi = torch.randn(1, 80, 1, requires_grad=True)
        m(xi)[0, 60].sum().backward()
        nz = torch.nonzero(xi.grad[0, :, 0])[:, 0]
        ensure(int(nz.max()) == 60 and int(nz.min()) == 29, f"t = 60 çıkışı tam [29, 60] girdilerine bağlı (RF 1 + 1 + 30 = 32), bulunan [{int(nz.min())}, {int(nz.max())}]")
    R_.test("nedensellik ve receptive field", t_causal)

    def t_train():
        S = _series(5000, 48, 0)
        Xtr, Xte, y8 = S[:4000, :40], S[4000:, :40], S[4000:, 40:, 0]
        Ytr = torch.stack([S[:4000, k:k + 40, 0] for k in range(1, 9)], -1)
        torch.manual_seed(0)
        m = ns["WaveNet"]()
        opt = torch.optim.Adam(m.parameters(), 3e-3)
        t0 = time.perf_counter()
        for _ in range(10):
            perm = torch.randperm(4000)
            for i in range(0, 4000, 32):
                idx = perm[i:i + 32]
                loss = F.mse_loss(m(Xtr[idx]), Ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            mse = F.mse_loss(m(Xte)[:, -1], y8).item()
        ensure(mse < 0.012, f"WaveNet 8 adım MSE {mse:.4f}")
        rows = []
        gru = torch.nn.GRU(1, 32, batch_first=True)
        wn = ns["WaveNet"](32, tuple(2 ** i for i in range(10)), 1)
        for T in (256, 1024, 4096):
            x = torch.randn(8, T, 1)
            r = [T]
            for mod in (gru, wn):
                best = 1e9
                for _ in range(2):
                    t1 = time.perf_counter()
                    out = mod(x)
                    (out[0] if isinstance(out, tuple) else out).sum().backward()
                    best = min(best, time.perf_counter() - t1)
                r.append(fmt_time(best))
            rows.append(r)
        R_.note(f"WaveNet dizi→dizi, 10 epoch ({time.perf_counter() - t0:.0f} sn): 8 adım MSE = {mse:.4f} (d aşamasındaki RNN'lerden iyi; RF = 32: 40'lık pencerenin çoğu).")
        R_.table("İleri + geri süresi, batch 8 (CPU): GRU-32 vs WaveNet-32 (1…512, RF 1024)", ["T", "GRU", "WaveNet"], rows,
                 note="GRU T adımı sırayla yürür (paralellik yok); nedensel konvolüsyonlar tüm zaman adımlarını aynı anda hesaplar. Bedeli: sabit, tasarımla belirlenen receptive field — ötesini hiç göremez.")
    R_.test("WaveNet eğitimi ve hız", t_train)
