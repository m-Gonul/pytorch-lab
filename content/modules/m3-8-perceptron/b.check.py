import math
import random
from labcheck import no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def logic():
        t = need(ns, "tlu")
        ensure([t((1, 1), -1.5, x) for x in ((0, 0), (0, 1), (1, 0), (1, 1))] == [0, 0, 0, 1], "VE")
        ensure([t((1, 1), -0.5, x) for x in ((0, 0), (0, 1), (1, 0), (1, 1))] == [0, 1, 1, 1], "VEYA")
        ensure([need(ns, "xor_net")(x) for x in ((0, 0), (0, 1), (1, 0), (1, 1))] == [0, 1, 1, 0], "XOR ağı")
        X = [(0, 0), (0, 1), (1, 0), (1, 1)]
        *_, conv = need(ns, "train_perceptron")(X, [0, 1, 1, 0], 1.0, 50)
        ensure(not conv, "XOR'da perceptron yakınsamamalı")
    R_.test("mantık birimleri ve XOR", logic)

    def conv():
        rng = random.Random(0)
        rows = []
        for gamma in (0.4, 0.2, 0.1, 0.05):
            X, y = [], []
            while len(X) < 200:
                p = (rng.uniform(-1, 1), rng.uniform(-1, 1))
                s = (p[0] + 2 * p[1] - 0.3) / math.sqrt(5)
                if abs(s) >= gamma:
                    X.append(p)
                    y.append(int(s > 0))
            w, b, err, ok = ns["train_perceptron"](X, y, 1.0, 1000)
            ensure(ok and all(ns["tlu"](w, b, x) == t for x, t in zip(X, y)), f"γ = {gamma}: ayrılabilir veride yakınsamalı")
            R2 = max(1 + x[0] ** 2 + x[1] ** 2 for x in X)
            bound = R2 / gamma ** 2
            ensure(err <= bound, "Novikoff sınırı (bias'ı ek bir girdi 1 olarak say)")
            rows.append([gamma, err, f"{bound:.0f}", f"{need(ns, 'margin')(w, b, X, y):.3f}"])
        R_.table("Doğrusal ayrılabilir 200 örnek, gerçek kenar payı γ", ["γ", "perceptron hata sayısı", "Novikoff (R/γ)²", "bulunan çözümün kenar payı"], rows,
                 note="Sınır kötümserdir ama doğru ölçeklenir: γ yarıya inince hata sayısı kabaca 4 katına çıkar. Perceptron herhangi bir ayırıcı bulur, en geniş kenarlıyı değil (SVM'nin farkı, x1).")
    R_.test("yakınsama", conv)
