import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def hand():
        xs = [[1.0], [0.0], [0.0]]
        hs = need(ns, "rnn_forward")(xs, [0.0], [[1.0]], [[0.5]], [0.0])
        close([h[0] for h in hs], [0.761594, 0.363399, 0.179726], atol=1e-5)
        dWx, dWh, db, dh0 = need(ns, "rnn_backward")(xs, hs, [0.0], [[1.0]], [[0.5]], [[0.0], [0.0], [1.0]])
        close(dWx[0][0], 0.088185, atol=1e-5, what="∂L/∂u")
    R.test("elle örnek", hand)

    def num():
        rng = random.Random(0)
        D, H, T = 2, 3, 4
        xs = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(T)]
        h0 = [rng.gauss(0, 0.5) for _ in range(H)]
        Wx = [[rng.gauss(0, 0.7) for _ in range(D)] for _ in range(H)]
        Wh = [[rng.gauss(0, 0.7) for _ in range(H)] for _ in range(H)]
        b = [rng.gauss(0, 0.3) for _ in range(H)]
        C = [[rng.gauss(0, 1) for _ in range(H)] for _ in range(T)]
        L = lambda Wx_, Wh_, h0_: sum(c * v for ct, ht in zip(C, need(ns, "rnn_forward")(xs, h0_, Wx_, Wh_, b)) for c, v in zip(ct, ht))
        hs = need(ns, "rnn_forward")(xs, h0, Wx, Wh, b)
        dWx, dWh, db, dh0 = need(ns, "rnn_backward")(xs, hs, h0, Wx, Wh, C)
        e = 1e-6
        for i in range(H):
            for k in range(H):
                P = [r[:] for r in Wh]; M = [r[:] for r in Wh]; P[i][k] += e; M[i][k] -= e
                close(dWh[i][k], (L(Wx, P, h0) - L(Wx, M, h0)) / (2 * e), rtol=1e-5, atol=1e-7, what=f"dWh[{i}][{k}]")
        for k in range(H):
            P, M = h0[:], h0[:]; P[k] += e; M[k] -= e
            close(dh0[k], (L(Wx, Wh, P) - L(Wx, Wh, M)) / (2 * e), rtol=1e-5, atol=1e-7, what=f"dh0[{k}]")
    R.test("D = 2, H = 3, T = 4: sayısal gradyanla aynı", num)
