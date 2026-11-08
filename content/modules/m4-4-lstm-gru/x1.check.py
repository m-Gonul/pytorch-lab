import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("peephole_step", "cifg_step")])

    def t():
        rng = np.random.default_rng(0)
        B, d, H = 3, 4, 5
        x, h, c = rng.normal(size=(B, d)), rng.normal(size=(B, H)), rng.normal(size=(B, H))
        W, U, b = rng.normal(size=(4 * H, d)), rng.normal(size=(4 * H, H)), rng.normal(size=4 * H)
        s = lambda z: 1 / (1 + np.exp(-z))
        a = x @ W.T + h @ U.T + b
        i, f, g, o = s(a[:, :H]), s(a[:, H:2 * H]), np.tanh(a[:, 2 * H:3 * H]), s(a[:, 3 * H:])
        c_ref = f * c + i * g
        h2, c2 = need(ns, "peephole_step")(x, h, c, W, U, b, np.zeros((3, H)))
        close((h2, c2), (o * np.tanh(c_ref), c_ref), atol=1e-12, what="P = 0 → standart LSTM")
        P = rng.normal(size=(3, H))
        h3, c3 = ns["peephole_step"](x, h, c, W, U, b, P)
        i2, f2 = s(a[:, :H] + P[0] * c), s(a[:, H:2 * H] + P[1] * c)
        cc = f2 * c + i2 * g
        close((h3, c3), (s(a[:, 3 * H:] + P[2] * cc) * np.tanh(cc), cc), atol=1e-12, what="peephole: i, f c_(t−1)'i, o c_t'yi görür")
        W3, U3, b3 = rng.normal(size=(3 * H, d)), rng.normal(size=(3 * H, H)), rng.normal(size=3 * H)
        h4, c4 = need(ns, "cifg_step")(x, h, c, W3, U3, b3)
        a3 = x @ W3.T + h @ U3.T + b3
        ff = s(a3[:, :H])
        close(c4, ff * c + (1 - ff) * np.tanh(a3[:, H:2 * H]), atol=1e-12, what="CIFG: i = 1 − f")
        ensure(np.all(np.abs(c4) <= np.maximum(np.abs(c), 1) + 1e-12), "CIFG: c dışbükey karışım — |c| büyüyemez")
        R_.note("Peephole bağlantıları kapılara hücrenin kendi içeriğini gösterir — kesin zamanlama gerektiren görevlerde (ör. aralık sayma) yardımcı. Greff ve ark.'nın 5400 deneylik çalışmasında ne peephole ne de CIFG anlamlı fark yaratmadı; en kritik bileşenler unutma kapısı ve çıkış aktivasyonu çıktı. CIFG parametrenin dörtte birini kazandırır ve GRU'nun güncelleme kapısıyla aynı fikirdir (z ve 1 − z).")
    R_.test("peephole ve CIFG", t)
