import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngü kuralları", lambda: (max_loops(src, 0, func="augment"), max_loops(src, 0, func="az_loss"), max_loops(src, 1, func="fit_elo")))

    def sym():
        P = need(ns, "symmetry_perms")()
        ensure(P.shape == (8, 9) and len({tuple(p) for p in P}) == 8, "8 farklı permütasyon")
        ensure(all(sorted(p) == list(range(9)) for p in P) and all(p[4] == 4 for p in P), "permütasyon, merkez sabit")
        corners = {0, 2, 6, 8}
        ensure(all({int(p[i]) for i in corners} == corners for p in P), "köşeler köşelere")
        rng = np.random.default_rng(0)
        X = rng.integers(0, 2, (5, 18)).astype(float)
        PI = rng.dirichlet(np.ones(9), 5)
        Xa, PIa, Za = need(ns, "augment")(X, PI, np.arange(5.0))
        ensure(Xa.shape == (40, 18) and PIa.shape == (40, 9) and Za.shape == (40,), "şekiller")
        close(np.sort(Xa[Za == 3][:, :9].sum(1)), np.full(8, X[3, :9].sum()), what="taş sayısı korunur")
        close(PIa.sum(1), np.ones(40), what="π hâlâ dağılım")
    R_.test("simetriler", sym)

    def loss():
        v = need(ns, "az_loss")(np.array([1.0]), np.array([0.5]), np.array([[0.5, 0.5]]), np.array([[0.0, 0.0]]), 0.0)
        close(v, 0.25 + np.log(2), what="el hesabı")
        big = ns["az_loss"](np.array([1.0]), np.array([1.0]), np.array([[1.0, 0.0]]), np.array([[1000.0, 0.0]]), 0.0)
        ensure(np.isfinite(big) and abs(big) < 1e-9, "kararlı log-softmax")
    R_.test("az_loss", loss)

    def elo():
        close(need(ns, "elo_expected")(1200, 1000), 1 / (1 + 10 ** -0.5), what="beklenen skor")
        rng = np.random.default_rng(1)
        true = np.array([0.0, 150, 300, 600, 900])
        K = len(true)
        Wm = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i != j:
                    Wm[i, j] = rng.binomial(400, ns["elo_expected"](true[i], true[j]))
        Wm = (Wm + (400 - Wm.T)) / 2
        np.fill_diagonal(Wm, 0)
        r = need(ns, "fit_elo")(Wm)
        close(r - r[0], true - true[0], atol=40, what="dereceler geri kazanılır")
        R_.table("Bradley–Terry ile Elo (her çift 400 oyun)", ["oyuncu", "gerçek", "kestirim"], [[i, f"{true[i] - true[0]:.0f}", f"{r[i] - r[0]:.0f}"] for i in range(K)],
                 note="Elo yalnız farklar için anlamlıdır; AlphaGo makaleleri bir oyuncuyu (ör. GnuGo = 0 ya da rastgele oyuncu) sabitleyerek ölçekler.")
    R_.test("Elo / Bradley–Terry", elo)
