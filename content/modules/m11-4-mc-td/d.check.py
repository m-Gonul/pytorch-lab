import random
import torch
from labcheck import close, need, ensure


def episode(rng, N=19):
    s = N // 2 + 1
    st, rw = [s], []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1)
        rw.append(1.0 if s2 == N + 1 else (-1.0 if s2 == 0 else 0.0))
        st.append(s2)
        if s2 in (0, N + 1):
            return st, rw
        s = s2


def check(ns, R_):
    rng = random.Random(0)
    V0 = torch.randn(21, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 0.3
    V0[0] = V0[20] = 0

    def lr():
        st, rw = [3, 4, 5, 6], [0.0, 0.0, 1.0]
        V = torch.tensor([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0], dtype=torch.float64)
        G = need(ns, "lambda_returns")(st, rw, V, 1.0, 0.0)
        close(G.numpy(), torch.tensor([0.4, 0.5, 1.0]).numpy(), what="λ = 0: TD(0) hedefleri")
        G1 = ns["lambda_returns"](st, rw, V, 1.0, 1.0)
        close(G1.numpy(), torch.tensor([1.0, 1.0, 1.0]).numpy(), what="λ = 1: MC getirileri")
    R_.test("lambda_returns", lr)

    def eq():
        for lam in (0.0, 0.4, 0.9, 1.0):
            for k in range(5):
                st, rw = episode(rng)
                a = need(ns, "offline_forward")(st, rw, V0, 0.1, 0.95, lam)
                b = need(ns, "offline_traces")(st, rw, V0, 0.1, 0.95, lam)
                close(a.numpy(), b.numpy(), atol=1e-12, what=f"λ = {lam}: ileri = geri (çevrimdışı)")
    R_.test("çevrimdışı ileri bakış = geriye bakış izleri", eq)

    def on():
        st, rw = episode(random.Random(5))
        a = ns["offline_traces"](st, rw, V0, 0.1, 1.0, 0.8)
        b = need(ns, "online_traces")(st, rw, V0, 0.1, 1.0, 0.8)
        d = float((a - b).abs().max())
        ensure(d > 0, "çevrimiçi sürüm farklı olmalı")
        rows = []
        true = torch.linspace(-1, 1, 21, dtype=torch.float64)[1:20] * 1.0
        true = torch.arange(1, 20, dtype=torch.float64) / 10 - 1
        for lam in (0.0, 0.4, 0.8, 0.95, 1.0):
            err = 0
            for seed in range(20):
                r2 = random.Random(seed)
                V = torch.zeros(21, dtype=torch.float64)
                for _ in range(10):
                    st, rw = episode(r2)
                    V = ns["online_traces"](st, rw, V, 0.2, 1.0, lam)
                err += float(((V[1:20] - true) ** 2).mean().sqrt())
            rows.append([lam, f"{err / 20:.3f}"])
        R_.table("19 durumlu yürüyüş, 10 bölüm, α = 0.2, çevrimiçi izler", ["λ", "RMS"], rows,
                 note=f"Çevrimiçi ve çevrimdışı sonuç aynı bölümde {d:.2e} farklı. Ara λ en iyidir; λ = 1'de büyük α ile kararsızlaşır (Sutton Şekil 12.3).")
    R_.test("online_traces ve λ eğrisi", on)
