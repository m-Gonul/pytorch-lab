import numpy as np
from labcheck import close, need, ensure


def check(ns, R):
    def gr():
        rows = []
        for N, p in [(10, 0.5), (10, 0.45), (20, 0.55)]:
            got = need(ns, "gamblers_ruin")(N, p)
            i = np.arange(1, N)
            q = 1 - p
            want = i / N if p == 0.5 else (1 - (q / p) ** i) / (1 - (q / p) ** N)
            close(got, want, atol=1e-10, what=f"N = {N}, p = {p}")
            rows.append([N, p, f"{got[N // 2 - 1]:.4f}"])
        R.table("Kumarbazın iflası: yarı servetle başlayıp hedefe ulaşma olasılığı", ["N", "p", "P(hedef)"], rows, note="p = 0.45'te bile olasılık hızla düşer: küçük bir dezavantaj uzun oyunda belirleyici.")
    R.test("gamblers_ruin (soğurulma olasılıkları)", gr)

    def ab():
        P = np.zeros((7, 7)); P[0, 0] = P[6, 6] = 1
        for i in range(1, 6):
            P[i, i - 1] = P[i, i + 1] = 0.5
        B, t = need(ns, "absorption")(P, [0, 6])
        close(t, np.array([i * (6 - i) for i in range(1, 6)], float), atol=1e-10, what="beklenen adım i(N − i)")
    R.test("absorption (beklenen adım)", ab)

    def mf():
        rng = np.random.default_rng(0)
        P = rng.random((5, 5)); P /= P.sum(1, keepdims=True)
        M = need(ns, "mean_first_passage")(P)
        w, V = np.linalg.eig(P.T); pi = np.real(V[:, np.argmin(abs(w - 1))]); pi /= pi.sum()
        close(np.diag(M), 1 / pi, atol=1e-9, what="Kac: m_jj = 1/π_j")
        s, total, cnt = 0, 0, 0
        steps = 0
        for _ in range(20000):
            nxt = rng.choice(5, p=P[s]); steps += 1
            if nxt == 3:
                total += steps; cnt += 1; steps = 0
            s = nxt
        close(total / cnt, M[3, 3] if True else 0, rtol=0.05, what="simülasyonla dönüş zamanı")
    R.test("mean_first_passage", mf)
