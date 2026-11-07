import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def ref(params, grad_fn, steps, lr, wd, decoupled, b1=0.9, b2=0.999, eps=1e-8):
    p = {k: v.copy() for k, v in params.items()}
    m = {k: np.zeros_like(v) for k, v in p.items()}; v_ = {k: np.zeros_like(v) for k, v in p.items()}
    for t in range(1, steps + 1):
        g = grad_fn(p)
        for k in p:
            gk = g[k] if decoupled else g[k] + wd * p[k]
            if decoupled:
                p[k] = p[k] * (1 - lr * wd)
            m[k] = b1 * m[k] + (1 - b1) * gk; v_[k] = b2 * v_[k] + (1 - b2) * gk * gk
            p[k] = p[k] - lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v_[k] / (1 - b2 ** t)) + eps)
    return p


def check(ns, R):
    src = ns["__source__"]
    R.test("step: en fazla bir (parametre) döngüsü", lambda: max_loops(src, 1, func="step"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(20, 5)), rng.normal(size=20)
    gf = lambda p: {"W": 2 * A.T @ (A @ p["W"] + p["b"] - B) / 20, "b": np.array([2 * (A @ p["W"] + p["b"] - B).mean()])}
    init = {"W": rng.normal(size=5), "b": np.array([0.5])}
    Adam = need(ns, "Adam", "sınıf") if "Adam" in ns else None

    def run(decoupled):
        p = {k: v.copy() for k, v in init.items()}
        opt = ns["Adam"](p, lr=0.05, weight_decay=0.1, decoupled=decoupled)
        for _ in range(40):
            opt.step(gf(p))
        return p

    def traj():
        need(ns, "Adam", "sınıf")
        for dec in (False, True):
            p = run(dec); want = ref(init, gf, 40, 0.05, 0.1, dec)
            close(p["W"], want["W"], what=f"decoupled={dec} W"); close(p["b"], want["b"], what=f"decoupled={dec} b")
        pa, pw = run(False), run(True)
        return f"40 adım sonra ‖W‖: Adam (L2) {np.linalg.norm(pa['W']):.4f}, AdamW {np.linalg.norm(pw['W']):.4f}"
    R.test("Adam ve AdamW referansla aynı; sonuçları farklı", traj)

    def ill():
        need(ns, "Adam", "sınıf")
        D = np.array([1.0, 100.0])
        loss = lambda w: (D * (w - 1) ** 2).sum()
        w_a = {"w": np.zeros(2)}; opt = ns["Adam"](w_a, lr=0.05)
        w_s = np.zeros(2)
        for _ in range(200):
            opt.step({"w": 2 * D * (w_a["w"] - 1)})
            w_s = w_s - 0.009 * 2 * D * (w_s - 1)
        ensure(loss(w_a["w"]) < 0.1 * loss(w_s), f"Adam {loss(w_a['w']):.3g}, SGD {loss(w_s):.3g}")
        return f"κ = 100 kuadratik, 200 adım: Adam kaybı {loss(w_a['w']):.2e}, SGD (kararlı en büyük lr) {loss(w_s):.2e}"
    R.test("kötü koşullu problemde Adam SGD'den hızlı", ill)
