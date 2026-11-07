import numpy as np
from labcheck import close, no_imports, need, ensure


def numgrad(f, arrs, h=1e-6):
    out = []
    for a in arrs:
        g = np.zeros_like(a)
        it = np.nditer(a, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = a[i]; a[i] = old + h; fp = f(); a[i] = old - h; fm = f(); a[i] = old
            g[i] = (fp - fm) / (2 * h)
        out.append(g)
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    T = need(ns, "Tensor", "sınıf") if "Tensor" in ns else None
    if T is None:
        R.test("`Tensor` tanımlı", lambda: need(ns, "Tensor", "sınıf"))
        return
    rng = np.random.default_rng(0)

    def ub():
        f = need(ns, "unbroadcast")
        G = rng.normal(size=(4, 2, 3))
        close(f(G, (3,)), G.sum((0, 1))); close(f(G, (2, 1)), G.sum(0).sum(1, keepdims=True)); close(f(G, (4, 1, 3)), G.sum(1, keepdims=True)); close(f(G, (4, 2, 3)), G)
    R.test("unbroadcast genel kural", ub)

    def expr():
        A, B, c = rng.normal(size=(4, 3)), rng.normal(size=(3, 5)), rng.normal(size=(5,))
        def f():
            a, b, cc = T(A), T(B), T(c)
            z = ((a @ b + cc) * 0.5 - cc ** 2).relu() + (a @ b) / 3.0
            return z.sum(), (a, b, cc)
        L, ts = f()
        L.backward()
        num = numgrad(lambda: float(f()[0].data), [A, B, c])
        for t, gnum, name in zip(ts, num, "ABc"):
            close(t.grad, gnum, rtol=1e-5, atol=1e-6, what=f"d{name}")
    R.test("matmul + broadcast + relu + pow + div: sayısal gradyanla aynı", expr)

    def lse():
        Z0 = rng.normal(size=(6, 4))
        def f():
            z = T(Z0)
            return ((z.exp().sum(axis=1, keepdims=True)).log() - z.sum(axis=1, keepdims=True) * 0.25).mean(), z
        L, z = f(); L.backward()
        close(z.grad, numgrad(lambda: float(f()[0].data), [Z0])[0], rtol=1e-5, atol=1e-7, what="dZ")
    R.test("exp / log / sum(axis, keepdims) / mean", lse)

    def acc():
        x = T(np.array([1.0, -2.0, 3.0]))
        (x * x).sum().backward()
        close(x.grad, [2, -4, 6], what="x·x")
        y = T(np.array(2.0)); (y + y * y + 1 - y).backward()
        close(y.grad, 4.0, what="y + y² + 1 − y")
    R.test("paylaşılan düğümde gradyan birikiyor", acc)

    def train():
        Xn = np.concatenate([rng.normal([0, 3], 1, (40, 2)), rng.normal([-3, -2], 1, (40, 2)), rng.normal([3, -2], 1, (40, 2))])
        yn = np.repeat([0, 1, 2], 40)
        Y = np.eye(3)[yn]
        W, b = T(np.zeros((2, 3))), T(np.zeros(3))
        for _ in range(150):
            W.grad[...] = 0; b.grad[...] = 0
            Z = T(Xn) @ W + b
            L = ((Z.exp().sum(axis=1, keepdims=True)).log() - (Z * Y).sum(axis=1, keepdims=True)).mean()
            L.backward()
            W.data -= 0.5 * W.grad; b.data -= 0.5 * b.grad
        a = ((Xn @ W.data + b.data).argmax(1) == yn).mean()
        ensure(a >= 0.9, f"doğruluk {a:.2f}")
        return f"senin motorunla softmax regression: doğruluk {a:.3f}, son kayıp {float(L.data):.3f}"
    R.test("motorla softmax regression eğitimi", train)
