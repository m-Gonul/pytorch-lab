import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def hand():
        L, g = need(ns, "mlp_loss")([1, 2], 1, [[1, 2], [-1, 1]], [0, 0], [1, -2], 0)
        close(L, 2); close(g["W1"], [[2, 4], [-4, -8]]); close(g["w2"], [10, 2]); close(g["x"], [6, 0]); close(g["b1"], [2, -4]); close(g["b2"], 2)
    R.test("elle örnek: tüm gradyanlar", hand)

    def num():
        rng = random.Random(0)
        x = [rng.gauss(0, 1) for _ in range(3)]
        W1 = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(4)]
        b1 = [rng.gauss(0, 0.5) for _ in range(4)]
        w2 = [rng.gauss(0, 1) for _ in range(4)]
        f = lambda W1_, b1_, w2_, b2_, x_: need(ns, "mlp_loss")(x_, 0.7, W1_, b1_, w2_, b2_)[0]
        L, g = need(ns, "mlp_loss")(x, 0.7, W1, b1, w2, 0.1)
        h = 1e-6
        for i in range(4):
            for j in range(3):
                Wp = [r[:] for r in W1]; Wm = [r[:] for r in W1]
                Wp[i][j] += h; Wm[i][j] -= h
                close(g["W1"][i][j], (f(Wp, b1, w2, 0.1, x) - f(Wm, b1, w2, 0.1, x)) / (2 * h), rtol=1e-5, atol=1e-7, what=f"dW1[{i}][{j}]")
        for j in range(3):
            xp, xm = x[:], x[:]; xp[j] += h; xm[j] -= h
            close(g["x"][j], (f(W1, b1, w2, 0.1, xp) - f(W1, b1, w2, 0.1, xm)) / (2 * h), rtol=1e-5, atol=1e-7, what=f"dx[{j}]")
    R.test("3-4-1 ağ: sayısal gradyanla aynı", num)
