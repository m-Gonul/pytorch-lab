import random
import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def loop_collect(k, n_envs, T, sigma):
    rng = random.Random(0)
    out = []
    for _ in range(n_envs):
        s = rng.gauss(0, 1)
        for _ in range(T):
            u = k * s + sigma * rng.gauss(0, 1)
            s2 = 1.1 * s + 0.5 * u + 0.1 * rng.gauss(0, 1)
            out.append((s, u, -(s * s + 0.5 * u * u), s2))
            s = max(min(s2, 5), -5)
    return out


def check(ns, R_):
    src = ns["__source__"]
    R_.test("yalnız adım döngüsü", lambda: max_loops(src, 1, func="collect"))
    P, kstar = ns["riccati"]()

    def q():
        g = torch.Generator().manual_seed(0)
        S, U, Rw, S2 = need(ns, "collect")(kstar, 2000, 30, 0.5, g)
        ensure(S.shape == (60000,) and torch.allclose(Rw, -(S ** 2 + 0.5 * U ** 2)), "geçişler")
        w = need(ns, "lstdq")(S, U, Rw, S2, kstar)
        # eylem-değer: Q(s,u) = -(s² + 0.5u²) - γ P (1.1 s + 0.5 u)² + sabit
        ref = torch.tensor([-(1 + 0.95 * P * 1.21), -(0.95 * P * 2 * 1.1 * 0.5), -(0.5 + 0.95 * P * 0.25)], dtype=torch.float64)
        close(w[:3].numpy(), ref.numpy(), atol=0.02, what="LSTD-Q, optimal politikada Riccati Q fonksiyonuna eşit")
    R_.test("collect ve lstdq", q)

    def d():
        rows = []
        for name, a in [("tam iyileştirme (k ← −w₁/2w₂)", None), ("DPG α = 0.3", 0.3), ("DPG α = 0.1", 0.1)]:
            k, h = need(ns, "dpg")(40, -0.5, a, 256, 20, 0.5, 1)
            rows.append([name] + [f"{h[i]:.3f}" for i in (0, 1, 5, 20, 40)])
            if a != 0.1:
                ensure(abs(k - kstar) < 0.03, f"{name}: k* = {kstar:.3f}'e yakınsamalı (bulunan {k:.3f})")
        t_vec = bench(lambda: ns["collect"](-1.0, 256, 20, 0.5, torch.Generator().manual_seed(0)), repeat=3, number=1)
        t_loop = bench(lambda: loop_collect(-1.0, 256, 20, 0.5), repeat=3, number=1)
        rows.append(["süre: 256 ortam toplu toplama", fmt_time(t_vec), "", "", "", ""])
        rows.append(["süre: ortam döngüsü", fmt_time(t_loop), "", "", "", ""])
        R_.table(f"LQR, k* = {kstar:.4f} (Riccati)", ["yöntem", "k₀", "k₁", "k₅", "k₂₀", "k₄₀"], rows,
                 note="Q bu problemde tam olarak karesel olduğundan doğrusal eleştirmen yanlılıksızdır; ∂Q/∂u = w₁s + 2w₂u eleştirmenin aktöre verdiği yön. Tam iyileştirme politika yinelemesidir (birkaç adımda yakınsar); gradyan adımları DDPG'nin aktörüne karşılık gelir.")
    R_.test("dpg", d)
