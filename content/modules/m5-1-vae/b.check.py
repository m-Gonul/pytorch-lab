import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    mu = [1.0, -0.5, 0.0]; lv = [math.log(0.25), 0.3, 0.0]
    want = sum(0.5 * (m * m + math.exp(l) - 1 - l) for m, l in zip(mu, lv))
    R.test("kl_gauss", lambda: close(need(ns, "kl_gauss")(mu, lv), want))

    def gr():
        dmu, dlv = need(ns, "kl_grad")(mu, lv)
        h = 1e-6
        f = ns["kl_gauss"]
        num_mu = [(f(mu[:i] + [mu[i] + h] + mu[i + 1:], lv) - f(mu[:i] + [mu[i] - h] + mu[i + 1:], lv)) / (2 * h) for i in range(3)]
        num_lv = [(f(mu, lv[:i] + [lv[i] + h] + lv[i + 1:]) - f(mu, lv[:i] + [lv[i] - h] + lv[i + 1:])) / (2 * h) for i in range(3)]
        close(dmu, num_mu, atol=1e-5, what="dmu"); close(dlv, num_lv, atol=1e-5, what="dlogvar")
    R.test("kl_grad = sayısal türev", gr)
    R.test("reparam", lambda: close(need(ns, "reparam")([1.0, 2.0], [math.log(0.25), 0.0], [0.4, -1.0]), [1.2, 1.0]))
    R.test("log_normal", lambda: close(need(ns, "log_normal")([0.5, 1.0], [0.0, 2.0], [0.0, math.log(4)]),
                                         -0.5 * (2 * math.log(2 * math.pi) + math.log(4) + 0.25 + 0.25)))

    def mc():
        rng = random.Random(0)
        eps = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(20000)]
        est = need(ns, "kl_mc")(mu, lv, eps)
        ensure(abs(est - want) < 0.05, f"MC {est:.4f}, kapalı form {want:.4f}")
        return f"20000 örnek: MC {est:.4f} ≈ kapalı form {want:.4f}"
    R.test("Monte Carlo KL → kapalı form", mc)
