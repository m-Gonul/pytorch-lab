import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def env_loop(r, v, d, v_last, gamma, lam):
    T, N = r.shape
    out = [[0.0] * N for _ in range(T)]
    rl, vl, dl, ll = r.tolist(), v.tolist(), d.tolist(), v_last.tolist()
    for n in range(N):
        last = 0.0
        for t in range(T - 1, -1, -1):
            nv = ll[n] if t == T - 1 else vl[t + 1][n]
            last = rl[t][n] + gamma * (1 - dl[t][n]) * nv - vl[t][n] + gamma * lam * (1 - dl[t][n]) * last
            out[t][n] = last
    return out


def check(ns, R_):
    src = ns["__source__"]
    R_.test("gae_matrix döngüsüz", lambda: (max_loops(src, 0, func="gae_matrix"), max_loops(src, 1, func="gae_scan")))
    g = torch.Generator().manual_seed(0)

    def eq():
        rows = []
        for T, N in ((16, 64), (128, 64), (512, 16)):
            r = torch.randn(T, N, generator=g, dtype=torch.float64)
            v = torch.randn(T, N, generator=g, dtype=torch.float64)
            d = (torch.rand(T, N, generator=g) < 0.05).double()
            vl = torch.randn(N, generator=g, dtype=torch.float64)
            a1 = need(ns, "gae_scan")(r, v, d, vl, 0.99, 0.95)
            a2 = need(ns, "gae_matrix")(r, v, d, vl, 0.99, 0.95)
            close(a1.numpy(), torch.tensor(env_loop(r, v, d, vl, 0.99, 0.95), dtype=torch.float64).numpy(), atol=1e-10, what="tarama = referans")
            close(a2.numpy(), a1.numpy(), atol=1e-10, what="matris = tarama")
            t0 = bench(lambda: env_loop(r, v, d, vl, 0.99, 0.95), repeat=1, number=1)
            t1 = bench(lambda: ns["gae_scan"](r, v, d, vl, 0.99, 0.95), repeat=3, number=1)
            t2 = bench(lambda: ns["gae_matrix"](r, v, d, vl, 0.99, 0.95), repeat=3, number=1)
            rows.append([f"{T} × {N}", fmt_time(t0), fmt_time(t1), fmt_time(t2)])
        R_.table("GAE süreleri", ["T × N", "saf Python (ortam × zaman)", "zaman taraması, ortamlar paralel", "(N, T, T) matris"], rows,
                 note="Matris sürümü O(T²N) iş yapar: kısa T'de Python döngüsünü yok ettiği için hızlı, uzun T'de gereksiz sıfırlarla yavaş. Doğru vektörleştirme bağımlılık zincirini (zaman) sıralı bırakıp bağımsız ekseni (ortamlar) paralelleştirir.")
    R_.test("gae_scan = gae_matrix", eq)

    def lam():
        S, R, D = need(ns, "walk_batch")(20000, 400, torch.Generator().manual_seed(1))
        ensure(S.shape == (401, 20000) and float(D[-1].mean()) > 0.98, "şekil ve bitiş")
        true = torch.linspace(-1, 1, 21, dtype=torch.float64)
        true[0] = true[-1] = 0
        Vh = true + 0.3 * torch.sin(torch.arange(21, dtype=torch.float64) * 1.3)
        Vh[0] = Vh[-1] = 0
        V = Vh[S]
        rows, mse = [], {}
        for l in (0.0, 0.5, 0.8, 0.9, 0.95, 1.0):
            a = ns["gae_scan"](R, V[:-1], D, V[-1], 1.0, l)
            ret0 = a[0] + V[0]
            mse[l] = float((ret0 ** 2).mean())
            rows.append([l, f"{float(ret0.mean()):+.4f}", f"{float(ret0.var()):.4f}", f"{mse[l]:.4f}"])
        best = min(mse, key=mse.get)
        ensure(0 < best < 1 and mse[best] < 0.5 * mse[1.0], f"ara λ en iyi (bulunan {best})")
        R_.table("19 durumlu yürüyüş, merkezde λ-getirisi hedefi (gerçek değer 0), V̂ = V + 0.3 sin(1.3 s), 20 000 bölüm", ["λ", "ortalama", "varyans", "MSE"], rows,
                 note="λ küçükken hedef yanlış V̂'ye dayanır (yanlılık ve V̂ farklarından gelen gürültü); λ = 1'de tüm bölüm rastgeleliği hedefe girer. GAE makalesindeki λ ≈ 0.95–0.99 seçimi bu ödünleşimin uzun ufuklu görevlerdeki yeridir.")
    R_.test("λ ödünleşimi", lam)
