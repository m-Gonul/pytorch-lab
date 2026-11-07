import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def ref_forward(idx, P, H):
    def ln(x, g, b):
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * g + b
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    B, T = idx.shape
    out = []
    for bi in range(B):
        x = P["wte"][idx[bi]] + P["wpe"][:T]
        d = x.shape[1]; dk = d // H
        for p in P["blocks"]:
            h = ln(x, p["ln1_g"], p["ln1_b"])
            qkv = h @ p["attn_w"] + p["attn_b"]
            heads = []
            for i in range(H):
                q = qkv[:, i * dk:(i + 1) * dk]; k = qkv[:, d + i * dk:d + (i + 1) * dk]; v = qkv[:, 2 * d + i * dk:2 * d + (i + 1) * dk]
                S = q @ k.T / np.sqrt(dk)
                S[np.triu_indices(T, 1)] = -np.inf
                A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
                heads.append(A @ v)
            x = x + np.concatenate(heads, 1) @ p["proj_w"] + p["proj_b"]
            x = x + gelu(ln(x, p["ln2_g"], p["ln2_b"]) @ p["fc_w"] + p["fc_b"]) @ p["fc2_w"] + p["fc2_b"]
        out.append(ln(x, P["lnf_g"], P["lnf_b"]) @ P["wte"].T)
    return np.stack(out)


def params(rng, V, T, d, L):
    def blk():
        r = lambda *s: rng.normal(size=s) * 0.3
        return dict(ln1_g=1 + r(d), ln1_b=r(d), attn_w=r(d, 3 * d), attn_b=r(3 * d), proj_w=r(d, d), proj_b=r(d),
                    ln2_g=1 + r(d), ln2_b=r(d), fc_w=r(d, 4 * d), fc_b=r(4 * d), fc2_w=r(4 * d, d), fc2_b=r(d))
    return dict(wte=rng.normal(size=(V, d)) * 0.3, wpe=rng.normal(size=(T, d)) * 0.3,
                blocks=[blk() for _ in range(L)], lnf_g=1 + rng.normal(size=d) * 0.1, lnf_b=rng.normal(size=d) * 0.1)


def check(ns, R):
    src = ns["__source__"]
    R.test("gpt_forward'da tek döngü (bloklar)", lambda: max_loops(src, 1, func="gpt_forward", why="head ve konumlar vektörel"))
    R.test("layer_norm, gelu döngüsüz", lambda: (max_loops(src, 0, func="layer_norm"), max_loops(src, 0, func="gelu")))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(3)

    def ln():
        x = rng.normal(size=(2, 3, 6)); g = rng.normal(size=6); b = rng.normal(size=6)
        y = need(ns, "layer_norm")(x, g, b)
        close((y - b) / g, (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5))
    R.test("layer_norm", ln)
    R.test("gelu (tanh)", lambda: close(need(ns, "gelu")(np.array([-1.0, 0, 1, 3])), [-0.158808, 0, 0.841192, 2.996363], atol=1e-5))
    P = params(rng, 13, 10, 12, 3)
    idx = rng.integers(0, 13, (2, 7))
    R.test("3 blok, 3 head: başvuru ile aynı", lambda: close(need(ns, "gpt_forward")(idx, P, 3), ref_forward(idx, P, 3)))

    def causal():
        idx2 = idx.copy(); idx2[:, 5:] = (idx2[:, 5:] + 1) % 13
        a = need(ns, "gpt_forward")(idx, P, 3); b = need(ns, "gpt_forward")(idx2, P, 3)
        close(a[:, :5], b[:, :5], what="ilk 5 konum")
    R.test("nedensellik", causal)
