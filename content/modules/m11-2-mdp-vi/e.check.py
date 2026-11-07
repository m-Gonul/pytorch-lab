import numpy as np
from scipy.stats import poisson
from labcheck import close, need, ensure, bench, fmt_time


def check(ns, R_):
    def mdp():
        P, R = need(ns, "inventory_mdp")(C=20, max_order=8, lam=4.0)
        ensure(P.shape == (9, 21, 21) and R.shape == (21, 9), f"şekiller {P.shape}, {R.shape}")
        close(P.sum(-1), np.ones((9, 21)), atol=1e-12, what="satırlar olasılık")
        s, a = 5, 3
        y = 8; d = np.arange(29); pd = poisson.pmf(d, 4.0); pd[-1] += poisson.sf(28, 4.0)
        want = (pd * (10 * np.minimum(y, d) - 0.5 * np.maximum(y - d, 0) - 2 * np.maximum(d - y, 0))).sum() - 4 * a
        close(R[s, a], want, atol=1e-10, what="beklenen ödül")
        close(P[a, s, 2], pd[6], atol=1e-12, what="geçiş: 8 − 6 = 2")
    R_.test("inventory_mdp", mdp)

    def solve():
        P, R = ns["inventory_mdp"]()
        V, pi = need(ns, "vi_vectorized")(P, R, 0.95)
        close(V, ns["vi_loop"](P, R, 0.95), atol=1e-6, what="döngülü VI ile aynı")
        post = np.minimum(np.arange(51) + pi, 50)
        low = np.where(pi > 0)[0]
        ensure(len(low) > 0 and (post[low] == post[low][0]).mean() > 0.8, "sipariş verilen durumlarda hedef düzey sabit olmalı")
        ensure((pi[low.max() + 1:] == 0).all(), "eşiğin üstünde sipariş yok")
        t1 = bench(lambda: ns["vi_loop"](P, R, 0.95, 1e-6), repeat=1, number=1)
        t2 = bench(lambda: ns["vi_vectorized"](P, R, 0.95, 1e-6), repeat=3)
        R_.table("Stok MDP (51 durum × 21 eylem), γ = 0.95", ["ölçü", "değer"], [["sipariş eşiği s", int(low.max())], ["hedef düzey S", int(post[low][0])], ["döngülü VI", fmt_time(t1)], ["vektörel VI", fmt_time(t2)]],
                 note="(s, S) politikası: stok s'nin altına inince S'ye tamamla. Scarf (1960) bu yapının sabit sipariş maliyetinde en iyi olduğunu kanıtladı; burada sabit maliyet yok, yine de eşik yapısı çıkıyor.")
    R_.test("vi_vectorized ve (s, S) yapısı", solve)
