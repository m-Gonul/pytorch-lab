import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("masked_softmax", "additive_attention", "luong_attention", "masked_ce")])
    rng = np.random.default_rng(0)

    def t():
        e = np.e
        a, c = need(ns, "luong_attention")(np.array([[1., 0]]), np.array([[[1., 0], [0, 1], [1, 1]]]), np.ones((1, 3), bool))
        close(a[0], [e / (2 * e + 1), 1 / (2 * e + 1), e / (2 * e + 1)], what="el örneği α")
        close(c[0, 0], 2 * e / (2 * e + 1), what="bağlam")
        B, T, dh, ds, da = 3, 6, 5, 4, 7
        s, H = rng.normal(size=(B, ds)), rng.normal(size=(B, T, dh))
        W1, W2, v = rng.normal(size=(dh, da)), rng.normal(size=(ds, da)), rng.normal(size=da)
        mask = np.arange(T)[None] < np.array([6, 3, 1])[:, None]
        a, c = need(ns, "additive_attention")(s, H, W1, W2, v, mask)
        for b in range(B):
            L = mask[b].sum()
            ee = np.array([v @ np.tanh(W1.T @ H[b, i] + W2.T @ s[b]) for i in range(L)])
            ref = np.exp(ee - ee.max()) / np.exp(ee - ee.max()).sum()
            close(a[b, :L], ref, atol=1e-12, what=f"toplamsal α (örnek {b})")
            ensure(np.all(a[b, L:] == 0), "dolgu konumları 0 ağırlık")
            close(c[b], ref @ H[b, :L], atol=1e-12, what="bağlam")
        Wg = rng.normal(size=(ds, dh))
        a2, _ = ns["luong_attention"](s, H, mask, Wg)
        sc = np.einsum("bd,btd->bt", s @ Wg, H)
        close(a2[0], np.exp(sc[0] - sc[0].max()) / np.exp(sc[0] - sc[0].max()).sum(), atol=1e-12, what="Luong general")
        logits = rng.normal(size=(2, 4, 6))
        tg = np.array([[3, 4, 2, 0], [5, 0, 0, 0]])
        lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        close(need(ns, "masked_ce")(logits, tg, 0), -(lp[0, 0, 3] + lp[0, 1, 4] + lp[0, 2, 2] + lp[1, 0, 5]) / 4, what="maskeli CE")
    R_.test("attention ve maskeler", t)
