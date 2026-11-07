import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("pe", "offset_rotation", "mha", "decoder_masks", "ls_ce")])
    rng = np.random.default_rng(0)

    def t_pe():
        P = need(ns, "pe")(100, 16)
        close(P[3, 4], np.sin(3 / 10000 ** (4 / 16)), what="PE")
        for k in (1, 5, 17):
            R = need(ns, "offset_rotation")(k, 16)
            close(P[:100 - k] @ R.T, P[k:], atol=1e-10, what=f"P[p + {k}] = R_k P[p]")
            close(R @ R.T, np.eye(16), atol=1e-12, what="R ortogonal")
        G = P @ P.T
        close(G[10, 15], G[40, 45], atol=1e-10, what="P_p · P_{p+k} yalnız k'ya bağlı")
    R_.test("sinüzoidal kodlama", t_pe)

    def t_mha():
        B, T, S, d, h = 2, 4, 5, 8, 2
        xq, xkv = rng.normal(size=(B, T, d)), rng.normal(size=(B, S, d))
        W = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4)]
        sm, cm = need(ns, "decoder_masks")([4, 2], [5, 3], T, S)
        ensure(sm.shape == (2, 4, 4) and cm.shape == (2, 4, 5), "şekiller")
        ensure(sm[0].sum() == 10 and not sm[1, 3, 2] and sm[1, 3, 1] and cm[1, 0].tolist() == [True] * 3 + [False] * 2, "maskeler")
        out = need(ns, "mha")(xq, xkv, *W, h, cm)
        for b in range(B):
            heads = []
            for j in range(h):
                sl = slice(j * d // h, (j + 1) * d // h)
                q, k, v = (xq[b] @ W[0])[:, sl], (xkv[b] @ W[1])[:, sl], (xkv[b] @ W[2])[:, sl]
                L = [5, 3][b]
                s = q @ k[:L].T / np.sqrt(d // h)
                a = np.exp(s - s.max(1, keepdims=True))
                a /= a.sum(1, keepdims=True)
                heads.append(a @ v[:L])
            close(out[b], np.concatenate(heads, 1) @ W[3], atol=1e-10, what="cross-attention (dolgu anahtarları dışarıda)")
        xkv2 = xkv.copy()
        xkv2[1, 3:] += 100
        close(ns["mha"](xq, xkv2, *W, h, cm)[1], out[1], atol=1e-10, what="dolgu içeriği çıktıyı etkilemez")
    R_.test("maskeli multi-head cross-attention", t_mha)

    def t_ls():
        logits = rng.normal(size=(2, 3, 5))
        t = np.array([[1, 4, 0], [2, 0, 0]])
        lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        ref = []
        for (b, i) in ((0, 0), (0, 1), (1, 0)):
            q = np.full(5, 0.02)
            q[t[b, i]] = 0.92
            ref.append(-(q * lp[b, i]).sum())
        close(need(ns, "ls_ce")(logits, t, 0.1, 0), np.mean(ref), what="yumuşatılmış CE")
        close(ns["ls_ce"](logits, t, 0.0, 0), -np.mean([lp[0, 0, 1], lp[0, 1, 4], lp[1, 0, 2]]), what="ε = 0 → normal CE")
    R_.test("etiket yumuşatma", t_ls)
