import numpy as np


def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gpt_forward(idx, P, n_head):
    B, T = idx.shape
    x = P["wte"][idx] + P["wpe"][:T]
    d = x.shape[-1]
    dk = d // n_head
    mask = np.triu(np.ones((T, T), bool), 1)
    for p in P["blocks"]:
        h = layer_norm(x, p["ln1_g"], p["ln1_b"])
        qkv = h @ p["attn_w"] + p["attn_b"]
        q, k, v = np.split(qkv.reshape(B, T, 3, n_head, dk).transpose(2, 0, 3, 1, 4), 3)
        S = q[0] @ k[0].transpose(0, 1, 3, 2) / np.sqrt(dk)
        S = np.where(mask, -np.inf, S)
        A = np.exp(S - S.max(-1, keepdims=True))
        A /= A.sum(-1, keepdims=True)
        o = (A @ v[0]).transpose(0, 2, 1, 3).reshape(B, T, d)
        x = x + o @ p["proj_w"] + p["proj_b"]
        h = layer_norm(x, p["ln2_g"], p["ln2_b"])
        x = x + gelu(h @ p["fc_w"] + p["fc_b"]) @ p["fc2_w"] + p["fc2_b"]
    return layer_norm(x, P["lnf_g"], P["lnf_b"]) @ P["wte"].T


rng = np.random.default_rng(0)
V, T, d, L = 11, 8, 8, 2
def blk():
    r = lambda *s: rng.normal(size=s) * 0.2
    return dict(ln1_g=np.ones(d), ln1_b=np.zeros(d), attn_w=r(d, 3 * d), attn_b=r(3 * d), proj_w=r(d, d), proj_b=r(d),
                ln2_g=np.ones(d), ln2_b=np.zeros(d), fc_w=r(d, 4 * d), fc_b=r(4 * d), fc2_w=r(4 * d, d), fc2_b=r(d))
P = dict(wte=rng.normal(size=(V, d)) * 0.2, wpe=rng.normal(size=(T, d)) * 0.2, blocks=[blk(), blk()], lnf_g=np.ones(d), lnf_b=np.zeros(d))
print(gpt_forward(np.array([[1, 2, 3, 4]]), P, n_head=2).shape)   # (1, 4, 11)
