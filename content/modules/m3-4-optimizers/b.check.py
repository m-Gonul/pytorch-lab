import math
from labcheck import close, no_imports, need, ensure


def rosen_grad(w):
    x, y = w
    return [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)]


def ref_traj(kind, steps, **kw):
    w, st = [-1.2, 1.0], {}
    out = []
    for _ in range(steps):
        g = rosen_grad(w)
        if kind == "sgd":
            u = st.get("u"); u = list(g) if u is None else [kw["beta"] * a + b for a, b in zip(u, g)]; st["u"] = u
            w = [a - kw["lr"] * b for a, b in zip(w, u)]
        elif kind == "rms":
            s = st.get("s", [0.0, 0.0]); s = [0.99 * a + 0.01 * b * b for a, b in zip(s, g)]; st["s"] = s
            w = [a - kw["lr"] * b / (math.sqrt(c) + 1e-8) for a, b, c in zip(w, g, s)]
        else:
            t = st.get("t", 0) + 1; m = st.get("m", [0.0, 0.0]); v = st.get("v", [0.0, 0.0])
            if kw.get("decoupled"):
                w = [a * (1 - kw["lr"] * kw["wd"]) for a in w]
            else:
                g = [a + kw["wd"] * b for a, b in zip(g, w)]
            m = [0.9 * a + 0.1 * b for a, b in zip(m, g)]; v = [0.999 * a + 0.001 * b * b for a, b in zip(v, g)]
            st.update(t=t, m=m, v=v)
            w = [a - kw["lr"] * (b / (1 - 0.9 ** t)) / (math.sqrt(c / (1 - 0.999 ** t)) + 1e-8) for a, b, c in zip(w, m, v)]
        out.append(list(w))
    return out


def run(fn, steps, **kw):
    w, st, out = [-1.2, 1.0], {}, []
    for _ in range(steps):
        w = fn(w, rosen_grad(w), st, **kw); out.append(list(w))
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def hand():
        w, st = [0.0], {}
        ws = []
        for _ in range(2):
            w = need(ns, "adam")(w, [2 * (w[0] - 3)], st, lr=0.1, eps=0.0); ws.append(w[0])
        close(ws, [0.1, 0.199897], atol=2e-6)
    R.test("Adam elle örnek: 0.1, 0.1999", hand)
    R.test("SGD momentum: Rosenbrock yörüngesi", lambda: close(run(need(ns, "sgd_momentum"), 30, lr=1e-4, beta=0.9), ref_traj("sgd", 30, lr=1e-4, beta=0.9)))
    R.test("RMSProp: Rosenbrock yörüngesi", lambda: close(run(need(ns, "rmsprop"), 30, lr=1e-3), ref_traj("rms", 30, lr=1e-3)))
    R.test("Adam (L2, wd = 0.1)", lambda: close(run(need(ns, "adam"), 30, lr=1e-2, wd=0.1), ref_traj("adam", 30, lr=1e-2, wd=0.1)))
    R.test("AdamW (decoupled, wd = 0.1)", lambda: close(run(need(ns, "adam"), 30, lr=1e-2, wd=0.1, decoupled=True), ref_traj("adam", 30, lr=1e-2, wd=0.1, decoupled=True)))
