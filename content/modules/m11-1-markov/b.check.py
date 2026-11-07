import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "scipy", "torch"}))

    def si():
        f = need(ns, "sample_index")
        ensure([f([0.2, 0.5, 0.3], u) for u in (0.0, 0.19, 0.2, 0.69, 0.7, 0.999)] == [0, 0, 1, 1, 2, 2], "ters CDF")
    R.test("sample_index", si)

    def ch():
        T = [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]]
        s = need(ns, "simulate_chain")(T, 0, 60000, 1)
        ensure(len(s) == 60001 and s[0] == 0, "uzunluk ve başlangıç")
        Tn = np.array(T)
        w, V = np.linalg.eig(Tn); pi = np.real(V[:, np.argmin(abs(w - 1))]); pi /= pi.sum()
        freq = np.bincount(s, minlength=3) / len(s)
        close(freq, pi, atol=0.015, what="uzun dönem frekanslar ≈ durağan dağılım")
        a = np.array(s)
        emp = np.array([[((a[:-1] == j) & (a[1:] == i)).sum() for j in range(3)] for i in range(3)], float)
        emp /= emp.sum(0, keepdims=True)
        close(emp, Tn, atol=0.02, what="ampirik geçiş matrisi")
        return f"frekanslar {np.round(freq, 3).tolist()}, durağan {np.round(pi, 3).tolist()}"
    R.test("simulate_chain", ch)

    def ar():
        x = need(ns, "ar1")(0.8, 1.0, 50000, 3)
        xs = np.array(x[1000:])
        close(xs.var(), 1 / 0.36, rtol=0.08, what="durağan varyans")
        for lag in (1, 2, 5):
            close(need(ns, "autocov")(x, lag), np.mean((np.array(x[:-lag]) - np.mean(x)) * (np.array(x[lag:]) - np.mean(x))) * (len(x) - lag) / len(x), atol=1e-9, what=f"autocov {lag}")
        r = [ns["autocov"](x[1000:], k) / ns["autocov"](x[1000:], 0) for k in range(4)]
        close(np.array(r), 0.8 ** np.arange(4), atol=0.05, what="otokorelasyon ≈ φ^τ")
    R.test("ar1 ve autocov", ar)
