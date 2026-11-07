import numpy as np
from labcheck import close, no_loops, no_imports, need


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü/comprehension yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(3)
    X = rng.normal(size=(9, 5))
    Xn = X / np.sqrt((X ** 2).sum(1))[:, None]
    ref = Xn @ Xn.T

    def cm():
        S = need(ns, "cosine_matrix")(X)
        close(S, ref, what="benzerlik matrisi")
        close(np.diag(S), np.ones(9), what="köşegen (kendisiyle benzerlik)")
        close(S, np.asarray(S).T, what="simetri")
    R.test("cosine_matrix doğru, köşegen 1, simetrik", cm)
    R.test("elle örnek u, v için S[0,1] = 0.3655", lambda: close(need(ns, "cosine_matrix")(np.array([[1., 2, 3], [4, -5, 6]]))[0, 1], 12 / np.sqrt(1078)))

    def ms():
        f = need(ns, "most_similar")
        for i in range(9):
            r = ref[i].copy(); r[i] = -np.inf
            got = f(X, i)
            assert int(got) == int(np.argmax(r)), f"i={i}: {got} bulundu, {int(np.argmax(r))} bekleniyordu"
        T = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
        assert int(f(T, 0)) == 1, "satır 0 kendisini seçmemeli"
    R.test("most_similar kendisini dışarıda bırakıyor", ms)
