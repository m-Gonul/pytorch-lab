import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: max_loops(src, 0, func="rope"))

    def t():
        rng = np.random.default_rng(0)
        d = 32
        q, k = rng.normal(size=(1, d)), rng.normal(size=(1, d))
        r = need(ns, "rope")
        close(np.linalg.norm(r(q, [7])), np.linalg.norm(q), what="normu korur")
        x = np.array([[1.0, 0.0]])
        close(r(x, [1], 10000), np.array([[np.cos(1), np.sin(1)]]), what="2B döndürme")
        rows = []
        for m_, n_ in ((3, 1), (50, 48), (1000, 998), (5, 7), (1005, 1007)):
            sr = float((r(q, [m_]) @ r(k, [n_]).T)[0, 0])
            P = np.zeros((2000, d))
            a = np.arange(2000)[:, None] * 10000.0 ** (-np.arange(0, d, 2) / d)
            P[:, 0::2], P[:, 1::2] = np.sin(a), np.cos(a)
            sa = float(((q + P[m_]) @ (k + P[n_]).T)[0, 0])
            rows.append([m_, n_, m_ - n_, f"{sr:+.4f}", f"{sa:+.4f}"])
        ensure(abs(float(rows[0][3]) - float(rows[1][3])) < 1e-3 and abs(float(rows[1][3]) - float(rows[2][3])) < 1e-3 and abs(float(rows[3][3]) - float(rows[4][3])) < 1e-3, "RoPE skoru yalnız m − n'ye bağlı")
        R_.table("q·k skoru: RoPE vs toplamsal sinüzoidal PE", ["m", "n", "m − n", "RoPE", "toplamsal PE"], rows,
                 note="RoPE sorgu ve anahtarı konumlarına göre döndürür: ⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩ — skor yalnız göreli uzaklığa bağlıdır, mutlak konum skora hiç girmez. Toplamsal PE'de (q + P_m)·(k + P_n) çapraz terimler mutlak konuma bağlıdır. LLaMA ve çoğu modern LLM RoPE kullanır; değerlere (V) uygulanmaz.")
    R_.test("RoPE", t)
