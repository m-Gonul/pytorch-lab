import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def ref_mha(x, Wq, Wk, Wv, Wo, h, causal):
    B, T, d = x.shape; dk = d // h
    out = np.zeros((B, T, d))
    for b in range(B):
        heads = []
        for i in range(h):
            sl = slice(i * dk, (i + 1) * dk)
            q, k, v = x[b] @ Wq[:, sl], x[b] @ Wk[:, sl], x[b] @ Wv[:, sl]
            S = q @ k.T / np.sqrt(dk)
            if causal:
                S[np.triu_indices(T, 1)] = -np.inf
            A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
            heads.append(A @ v)
        out[b] = np.concatenate(heads, 1) @ Wo
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    R.test("softmax: -inf ve büyük değer", lambda: close(need(ns, "softmax")(np.array([[1000.0, float("-inf"), 1000.0]])), [[0.5, 0, 0.5]]))
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 6, 12)); W = [rng.normal(size=(12, 12)) / 3 for _ in range(4)]

    def t(h, c):
        return lambda: close(need(ns, "mha")(x, *W, h=h, causal=c), ref_mha(x, *W, h, c))
    R.test("h = 3, tam", t(3, False))
    R.test("h = 3, causal", t(3, True))
    R.test("h = 1 (tek head)", t(1, False))

    def causal_prop():
        y = need(ns, "mha")(x, *W, h=4, causal=True)
        x2 = x.copy(); x2[:, 4:] = rng.normal(size=x2[:, 4:].shape)
        y2 = need(ns, "mha")(x2, *W, h=4, causal=True)
        close(y[:, :4], y2[:, :4], what="ilk 4 konum")
        ensure(not np.allclose(y[:, 4:], y2[:, 4:]), "son konumlar değişmeliydi")
    R.test("causal: gelecek değişince geçmiş çıktılar değişmiyor", causal_prop)
